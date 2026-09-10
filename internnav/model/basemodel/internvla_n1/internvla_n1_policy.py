import copy
import itertools
import re
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.evidence_history import (
    build_causal_history_metadata,
    select_causal_history_ids,
)
from internnav.model.basemodel.internvla_n1.evidence_sequence import prepare_conditioned_sequences
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    EVIDENCE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.model.utils.vln_utils import (
    S1Output,
    S2Output,
    chunk_token,
    split_and_clean,
    traj_to_actions,
)


class InternVLAN1Net(PreTrainedModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Union[InternVLAN1ModelConfig, ModelCfg]):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])

        self.model = InternVLAN1ForCausalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.model_config.device},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.init_prompts()

        self.num_frames = self.model_config.num_frames
        self.num_history = self.model_config.num_history
        self.num_future_steps = self.model_config.num_future_steps
        self.continuous_traj = self.model_config.continuous_traj
        self.resize_w = self.model_config.resize_w
        self.resize_h = self.model_config.resize_h

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0  # S2's episode idx is different from the system's idx
        self.conversation_history = []  # Multi-turn conversation exists when looking down
        self.llm_output = ""
        self.last_history_ids = []

    def init_prompts(self):
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        # For absolute pixel goal
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.last_history_ids = []

    def _append_observation(self, image, depth, pose):
        planar_pose = np.asarray(pose, dtype=np.float32)
        if planar_pose.shape != (3,):
            raise ValueError("pose must contain planar x, y, yaw")
        self.rgb_list.append(image)
        self.depth_list.append(depth)
        self.pose_list.append(planar_pose)

    def _prepare_evidence_inputs(self, inputs, history_ids):
        if not getattr(self.model.config, "use_evidence_memory", False):
            return inputs, {}
        input_ids = inputs.input_ids[0]
        image_mask = input_ids.eq(self.model.config.image_token_id)
        image_block_starts = torch.nonzero(image_mask & ~torch.roll(image_mask, 1), as_tuple=False).flatten()
        if image_mask[0]:
            image_block_starts[0] = 0
        if len(image_block_starts) <= len(history_ids):
            raise ValueError("cannot locate current observation image block during evidence prefill")
        insert_position = max(0, int(image_block_starts[len(history_ids)].item()) - 1)
        conditioned = prepare_conditioned_sequences(
            input_ids=(input_ids,),
            labels=(torch.full_like(input_ids, -100),),
            evidence_insert_positions=(insert_position,),
            evidence_token_id=EVIDENCE_TOKEN_INDEX,
            trajectory_token_id=151667,
            num_evidence_tokens=self.model.config.num_evidence_tokens,
            num_trajectory_tokens=0,
        )
        inputs["input_ids"] = conditioned.input_ids[0].unsqueeze(0)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        planar_poses = np.asarray(self.pose_list, dtype=np.float32)
        metadata = build_causal_history_metadata(
            history_ids,
            planar_poses[:, :2],
            planar_poses[:, 2],
            current_frame_id=len(self.pose_list) - 1,
        )
        evidence_kwargs = {
            "evidence_relative_poses": torch.from_numpy(metadata.relative_poses).unsqueeze(0).to(self.device),
            "evidence_ages": torch.from_numpy(metadata.ages).unsqueeze(0).to(self.device),
            "evidence_qualities": torch.from_numpy(metadata.qualities).unsqueeze(0).to(self.device),
            "evidence_valid_mask": torch.ones((1, len(history_ids)), dtype=torch.bool, device=self.device),
            "evidence_history_counts": torch.tensor([len(history_ids)], device=self.device),
            "evidence_image_counts": torch.tensor([len(self.input_images)], device=self.device),
            "evidence_prompt_lengths": torch.tensor([inputs["input_ids"].shape[1]], device=self.device),
        }
        return inputs, evidence_kwargs

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self._append_observation(image, depth, pose)
        self.episode_idx += 1

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        # Need to be careful: look_down images are not added to rgb_list and won't be selected as history
        # 1. Preprocess input
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:  # Don't add look_down images to rgb_list
            image = image.resize((self.resize_w, self.resize_h))
            self._append_observation(image, depth, pose)

        # 2. Prepare input for the model
        if not look_down:
            # Clear conversation history when not looking down, provide normal image history and instruction
            self.conversation_history = []
            # 2.1 instruction
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            # 2.2 images
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = select_causal_history_ids(self.episode_idx, self.num_history)
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.last_history_ids = history_id
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1  # Only increment when not looking down to maintain correspondence with rgb_list idx
        else:
            # Continue conversation based on previous when looking down
            self.input_images.append(image)  # This image should be the look_down image
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )
            history_id = self.last_history_ids

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        inputs, evidence_kwargs = self._prepare_evidence_inputs(inputs, history_id)

        # 3. Model inference
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
                **evidence_kwargs,
            ).sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")
        output = S2Output()

        # 4. Post-process results
        if bool(re.search(r'\d', self.llm_output)):  # Output pixel goal
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            output.output_pixel = np.array(pixel_goal)

            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            with torch.no_grad():
                traj_latents = self.model.generate_latents(
                    output_ids,
                    inputs.pixel_values,
                    image_grid_thw,
                    attention_mask=torch.ones_like(output_ids),
                    **evidence_kwargs,
                )
            output.output_latent = traj_latents

        else:  # Output action
            action_seq = self.parse_actions(self.llm_output)
            output.output_action = action_seq

        return output

    def s1_step_latent(self, rgb, depth, latent):
        with torch.no_grad():
            dp_actions = self.model.generate_traj(
                traj_latents=latent, images_dp=rgb, depths_dp=depth
            )  # use_aysnc based on MODEL

        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            random_choice = np.random.choice(dp_actions.shape[0])
            action_list = chunk_token(dp_actions[random_choice])

        action_list = [x for x in action_list if x != 0]

        output = S1Output(idx=action_list[:4])
        return output
