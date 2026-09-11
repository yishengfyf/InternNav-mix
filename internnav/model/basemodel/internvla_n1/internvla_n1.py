from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .evidence_conditioning import gather_visual_evidence, pool_visual_features
from .evidence_sequence import replace_evidence_embeddings
from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

TRAJ_TOKEN_INDEX = 151667
EVIDENCE_TOKEN_INDEX = 151668
IMAGE_TOKEN_INDEX = 151655
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


@dataclass
class DualVLNCausalLMOutput(CausalLMOutputWithPast):
    s2_loss: Optional[torch.FloatTensor] = None
    trajectory_loss: Optional[torch.FloatTensor] = None
    stage_loss: Optional[torch.FloatTensor] = None


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    model_type = "internvla_n1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def get_model(self):
        return self.model

    def _record_numeric(self, name, tensor):
        if not getattr(self.config, "numeric_diagnostics", False) or tensor is None:
            return
        if not hasattr(self, "numeric_diagnostics"):
            self.numeric_diagnostics = {}
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite].float()
        self.numeric_diagnostics[name] = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "finite": bool(finite.all()),
            "nonfinite": int(detached.numel() - finite.sum().item()),
            "min": float(finite_values.min()) if finite_values.numel() else None,
            "max": float(finite_values.max()) if finite_values.numel() else None,
            "max_abs": float(finite_values.abs().max()) if finite_values.numel() else None,
        }

    @staticmethod
    def _late_evidence_residual(hidden_states, evidence_tokens):
        """Read evidence per output position without backpropagating through the frozen VLM."""
        evidence_tokens = evidence_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        queries = F.normalize(hidden_states.detach().float(), dim=-1)
        keys = F.normalize(evidence_tokens.float(), dim=-1)
        weights = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)), dim=-1)
        return torch.matmul(weights.to(evidence_tokens.dtype), evidence_tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        evidence_keys = (
            "evidence_relative_poses",
            "evidence_ages",
            "evidence_qualities",
            "evidence_valid_mask",
            "evidence_history_counts",
            "evidence_image_counts",
            "evidence_prompt_lengths",
        )
        evidence_kwargs = {key: kwargs.get(key) for key in evidence_keys if kwargs.get(key) is not None}
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs.update(evidence_kwargs)
        return model_inputs

    def _inject_evidence_embeddings(
        self,
        input_ids,
        inputs_embeds,
        image_embeds,
        image_grid_thw,
        attention_mask,
        labels,
        evidence_relative_poses,
        evidence_ages,
        evidence_qualities,
        evidence_valid_mask,
        evidence_history_counts,
        evidence_image_counts,
        evidence_prompt_lengths=None,
    ):
        if not getattr(self.config, "use_evidence_memory", False):
            return inputs_embeds, None
        required = {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "evidence_relative_poses": evidence_relative_poses,
            "evidence_ages": evidence_ages,
            "evidence_qualities": evidence_qualities,
            "evidence_valid_mask": evidence_valid_mask,
            "evidence_history_counts": evidence_history_counts,
            "evidence_image_counts": evidence_image_counts,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"evidence conditioning is enabled but inputs are missing: {', '.join(missing)}")

        merge_size = self.config.vision_config.spatial_merge_size
        pooled_images = pool_visual_features(image_embeds, image_grid_thw, merge_size)
        visual_batch = gather_visual_evidence(
            pooled_images,
            evidence_image_counts,
            evidence_history_counts,
            max_history=evidence_valid_mask.shape[1],
        )
        prompt_mask = (
            attention_mask.bool()
            if attention_mask is not None and attention_mask.ndim == 2
            else torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        )
        prompt_mask &= input_ids.ne(IMAGE_TOKEN_INDEX)
        prompt_mask &= input_ids.ne(EVIDENCE_TOKEN_INDEX)
        prompt_mask &= input_ids.ne(TRAJ_TOKEN_INDEX)
        if labels is not None:
            prompt_mask &= labels.eq(-100)
        if evidence_prompt_lengths is not None:
            prompt_lengths = evidence_prompt_lengths.to(device=input_ids.device)
            prompt_mask &= torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < prompt_lengths[:, None]

        task_state = self.get_model().task_state_estimator(
            visual_batch.current_features,
            inputs_embeds,
            prompt_mask,
        )
        self._record_numeric("task_state", task_state)
        ablation = getattr(self.config, "evidence_ablation", "task_spatial")
        task_state, evidence_relative_poses, evidence_ages, evidence_qualities, evidence_valid_mask = (
            self._apply_evidence_ablation(
                ablation,
                task_state,
                evidence_relative_poses,
                evidence_ages,
                evidence_qualities,
                evidence_valid_mask,
            )
        )
        evidence_output = self.get_model().evidence_memory(
            visual_batch.history_features,
            evidence_relative_poses,
            evidence_ages,
            evidence_qualities,
            evidence_valid_mask,
            task_state,
        )
        self._record_numeric("evidence_tokens", evidence_output.tokens)
        evidence_tokens = evidence_output.tokens
        if getattr(self.config, "numeric_diagnostics", False) and evidence_tokens.requires_grad:
            evidence_tokens.retain_grad()
            self.numeric_gradient_tensors["evidence_tokens"] = evidence_tokens
        if getattr(self.config, "evidence_detach_tokens", False):
            evidence_tokens = evidence_tokens.detach()
        inputs_embeds = replace_evidence_embeddings(
            input_ids,
            inputs_embeds,
            evidence_tokens,
            EVIDENCE_TOKEN_INDEX,
        )
        self._record_numeric("inputs_embeds_after_evidence", inputs_embeds)
        if getattr(self.config, "numeric_diagnostics", False) and inputs_embeds.requires_grad:
            inputs_embeds.retain_grad()
            self.numeric_gradient_tensors["inputs_embeds_after_evidence"] = inputs_embeds
        return inputs_embeds, evidence_output

    @staticmethod
    def _apply_evidence_ablation(ablation, task_state, relative_poses, ages, qualities, valid_mask):
        if ablation not in {"null", "content", "spatial", "task_spatial"}:
            raise ValueError(f"unknown evidence ablation: {ablation}")
        if ablation != "task_spatial":
            task_state = torch.zeros_like(task_state)
        if ablation in {"null", "content"}:
            relative_poses = torch.zeros_like(relative_poses)
            ages = torch.zeros_like(ages)
            qualities = torch.zeros_like(qualities)
        if ablation == "null":
            valid_mask = torch.zeros_like(valid_mask)
        return task_state, relative_poses, ages, qualities, valid_mask

    def _apply_late_evidence_adapter(self, hidden_states, evidence_output):
        bypass_mode = getattr(self.config, "evidence_gradient_bypass_mode", "mean")
        if bypass_mode == "cross_attention":
            evidence_residual = self._late_evidence_residual(hidden_states, evidence_output.tokens)
        elif bypass_mode == "mean":
            evidence_residual = evidence_output.tokens.mean(dim=1, keepdim=True)
            evidence_residual = evidence_residual.to(device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            raise ValueError(f"unknown evidence_gradient_bypass_mode: {bypass_mode}")
        residual_scale = float(getattr(self.config, "evidence_gradient_bypass_scale", 0.1))
        return hidden_states + residual_scale * evidence_residual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
        evidence_frame_ids: Optional[torch.Tensor] = None,
        evidence_relative_poses: Optional[torch.Tensor] = None,
        evidence_ages: Optional[torch.Tensor] = None,
        evidence_qualities: Optional[torch.Tensor] = None,
        evidence_valid_mask: Optional[torch.Tensor] = None,
        evidence_history_counts: Optional[torch.Tensor] = None,
        evidence_image_counts: Optional[torch.Tensor] = None,
        evidence_prompt_lengths: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        evidence_output = None
        if getattr(self.config, "numeric_diagnostics", False):
            self.numeric_diagnostics = {}
            self.numeric_gradient_tensors = {}
            self._record_numeric("pixel_values", pixel_values)
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            self._record_numeric("text_embeddings", inputs_embeds)
            image_embeds = None
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                self._record_numeric("image_embeddings", image_embeds)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                self._record_numeric("inputs_embeds_before_evidence", inputs_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
            traj_idx = input_ids == TRAJ_TOKEN_INDEX
            if n_traj_tokens != 0:
                latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
                H = latent_queries.shape[-1]
                latent_queries = latent_queries.contiguous().view(-1, H)
                if n_traj_tokens != latent_queries.shape[0]:
                    raise ValueError(
                        f"trajectory placeholder count {n_traj_tokens} does not match latent queries {latent_queries.shape[0]}"
                    )
                inputs_embeds[traj_idx] = latent_queries.to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            if input_ids.eq(EVIDENCE_TOKEN_INDEX).any():
                inputs_embeds, evidence_output = self._inject_evidence_embeddings(
                    input_ids,
                    inputs_embeds,
                    image_embeds,
                    image_grid_thw,
                    attention_mask,
                    labels,
                    evidence_relative_poses,
                    evidence_ages,
                    evidence_qualities,
                    evidence_valid_mask,
                    evidence_history_counts,
                    evidence_image_counts,
                    evidence_prompt_lengths,
                )

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Optional training-safe path: keep evidence placeholders in the VLM
        # sequence, but route adapter gradients through a late residual instead
        # of the numerically unstable frozen VLM backward graph.
        gradient_bypass = bool(getattr(self.config, "evidence_gradient_bypass", False) and evidence_output is not None)
        if gradient_bypass:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
        else:
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        hidden_states = outputs[0]
        self._record_numeric("qwen_hidden_states", hidden_states)
        if gradient_bypass:
            hidden_states = hidden_states.detach()
            hidden_states = self._apply_late_evidence_adapter(hidden_states, evidence_output)
        logits = self.lm_head(hidden_states)
        self._record_numeric("logits", logits)

        loss = None
        s2_loss = None
        trajectory_loss = None
        stage_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            s2_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            self._record_numeric("s2_loss", s2_loss)

        if traj_poses is not None:
            traj_hidden_states = []
            for b in range(hidden_states.shape[0]):
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            self._record_numeric("trajectory_hidden_states", traj_hidden_states)
            if gradient_bypass:
                latent_queries = self.get_model().latent_queries.to(
                    device=traj_hidden_states.device, dtype=traj_hidden_states.dtype
                )
                residual_scale = float(getattr(self.config, "evidence_gradient_bypass_scale", 0.1))
                traj_hidden_states = traj_hidden_states + residual_scale * latent_queries
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            # In a dispatched model, the trajectory head may live on a different
            # GPU from the input batch. Keep all trajectory supervision together
            # with the latent/NextDiT branch so sharded training does not mix devices.
            trajectory_device = traj_hidden_states.device
            loss_mask = torch.arange(traj_images.size(1), device=trajectory_device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.to(trajectory_device).unsqueeze(1)

            if 'nextdit' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std

                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = torch.cat([memory_tokens, traj_hidden_states], dim=1)
                else:
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = traj_hidden_states

                relative_poses = traj_poses.flatten(0, 1).to(trajectory_device)
                bsz = relative_poses.shape[0]
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                target = noise - relative_poses
                trajectory_element_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                masked_loss = trajectory_element_loss * mask
                trajectory_loss = (
                    masked_loss.sum()
                    / mask.sum()
                    / (trajectory_element_loss.shape[1] * trajectory_element_loss.shape[2])
                )
                self._record_numeric("trajectory_loss", trajectory_loss)
            elif 'navdp' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    trajectory_loss = (
                        masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])
                    )

            else:
                raise NotImplementedError

        if evidence_output is not None and hasattr(self.config, "evidence_stage_loss_weight"):
            # Stage supervision is optional and never required by the runtime inference API.
            stage_loss = None
        weighted_losses = []
        if s2_loss is not None:
            weighted_losses.append(getattr(self.config, "s2_loss_weight", 1.0) * s2_loss)
        if trajectory_loss is not None:
            weighted_losses.append(getattr(self.config, "trajectory_loss_weight", 1.0) * trajectory_loss)
        if stage_loss is not None:
            weighted_losses.append(getattr(self.config, "evidence_stage_loss_weight", 1.0) * stage_loss)
        if weighted_losses:
            loss = sum(weighted_losses)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return DualVLNCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s2_loss=s2_loss,
            trajectory_loss=trajectory_loss,
            stage_loss=stage_loss,
        )

    def generate_latents(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        attention_mask=None,
        evidence_relative_poses=None,
        evidence_ages=None,
        evidence_qualities=None,
        evidence_valid_mask=None,
        evidence_history_counts=None,
        evidence_image_counts=None,
        evidence_prompt_lengths=None,
    ):
        input_ids = input_ids.to(self.get_model().device)
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1).to(text_embeds.dtype)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        evidence_output = None
        if input_ids.eq(EVIDENCE_TOKEN_INDEX).any():
            text_embeds, evidence_output = self._inject_evidence_embeddings(
                input_ids,
                text_embeds,
                image_embeds,
                image_grid_thw,
                attention_mask,
                None,
                evidence_relative_poses,
                evidence_ages,
                evidence_qualities,
                evidence_valid_mask,
                evidence_history_counts,
                evidence_image_counts,
                evidence_prompt_lengths,
            )

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        with torch.no_grad():
            gradient_bypass = bool(
                getattr(self.config, "evidence_gradient_bypass", False) and evidence_output is not None
            )
            outputs = self.model(
                inputs_embeds=text_embeds.detach() if gradient_bypass else text_embeds,
                position_ids=position_ids,
                # attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]
        if gradient_bypass:
            hidden_states = hidden_states.detach()
            residual_scale = float(getattr(self.config, "evidence_gradient_bypass_scale", 0.1))
            hidden_states = hidden_states + residual_scale * latent_queries.to(
                hidden_states.device, hidden_states.dtype
            )
            if evidence_output is not None:
                hidden_states = self._apply_late_evidence_adapter(hidden_states, evidence_output)

        return hidden_states

    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        if 'nextdit' in self.get_system1_type():
            scheduler = FlowMatchEulerDiscreteScheduler()
            device = traj_latents.device
            dtype = traj_latents.dtype

            traj_latents = self.get_model().cond_projector(traj_latents)
            if 'async' in self.get_system1_type():
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
                hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
            else:
                hidden_states = traj_latents
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
            batch_size = traj_latents.shape[0]
            latent_size = predict_step_nums
            latent_channels = 3

            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

            for t in scheduler.timesteps:
                latent_features = self.get_model().action_encoder(latents)
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
                latent_model_input = latent_features.repeat(2, 1, 1)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict noise model_output
                noise_pred = self.get_model().traj_dit(
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)

                # perform guidance
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous: x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        elif 'navdp' in self.get_system1_type():
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
