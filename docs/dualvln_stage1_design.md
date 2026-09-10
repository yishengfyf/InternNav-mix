# DualVLN Stage 1: Task-Conditioned Spatial Evidence Memory

Status: implementation design, not an experimental result.

Baseline: `upstream/main@7a5c62400ac45b313d9b709c740b64191556a242` on branch
`dualvln-spatial-memory-v1`.

## 1. Question and scope

Stage 1 tests whether causal spatial history becomes more useful when retrieval is conditioned on an estimated task
state. It must separate that effect from extra training, extra history images, and spatial metadata alone.

The first implementation keeps the native S2 outputs (pixel target, turn, look-down, and STOP) and the native S1
NextDiT trajectory interface. Full OCC, a recovery agent, a new backbone, and reliability learning are out of scope.

## 2. Current code constraints

- `NavPixelGoalDataset` samples up to `num_history` RGB frames uniformly from the causal prefix, but exposes them to
  S2 only as an ordinary image list. It currently does not return history frame ids, relative poses, ages, or quality.
- `DataCollatorForSupervisedDataset` appends four trajectory tokens and records their start in `t_s_pos`. Any evidence
  insertion must update sequence truncation, labels, attention masks, position ids, and `t_s_pos` together.
- `InternVLAN1ForCausalLM.forward` replaces image and trajectory-token embeddings, then reads S2 hidden states at
  `t_s_pos:t_s_pos+n_query` for NextDiT conditioning.
- When trajectory labels are present, the current forward computes trajectory MSE but not token cross entropy for the
  S2 answer. Stage 1 must expose `s2_loss`, `trajectory_loss`, and their configured weighted sum separately.
- `generate_latents` repeats the S2 pass after generation. Evidence injection must be shared by training, generation
  prefill, and latent extraction; changing only one path is invalid.
- `set_model` currently enables every NextDiT, RGB, resampler, projector, and latent-query parameter by substring.
  Stage 1 uses a fail-closed parameter allowlist and reports exact trainable counts.
- The policy declares `pose_list` but does not append poses alongside RGB history. The online evidence store must make
  RGB, pose, timestamp, and quality updates atomic.

## 3. Evidence contract

All runtime fields are available at or before the current decision. Training-only labels are kept separate.

| Field | Shape | Meaning |
| --- | --- | --- |
| `history_features` | `B x N x Dv` | pooled causal observation features |
| `relative_poses` | `B x N x 4` | current-frame `dx`, `dy`, `sin(dyaw)`, `cos(dyaw)` |
| `ages` | `B x N x 1` | non-negative decision-step age |
| `qualities` | `B x N x 2` | observable pose/measurement quality; no GT success signal |
| `valid_mask` | `B x N` | real entries versus padding |
| `task_state` | `B x Dt` | estimate from instruction, current observation, and causal history |
| output `tokens` | `B x K x H` | fixed-count evidence tokens in the S2 hidden dimension |

The writer combines observation features and spatial metadata without the task state. The reader adds the task-state
query to learned evidence queries and cross-attends over stored entries. An always-valid null entry makes empty history
well defined. Read weights, null weights, stage logits, and empty-history flags are returned for audits.

Task-state supervision may use training annotations or soft/masked pseudo-labels, but the inference API never accepts
GT progress. Negative age on a valid entry is rejected. The caller remains responsible for supplying a causal prefix.

## 4. S2 integration sequence

1. Pool each history image's visual tokens using its `image_grid_thw`; do not mix current-image tokens into storage.
2. Estimate `task_state` from instruction embeddings, current pooled visual features, and the previous recurrent task
   state. The initial state is learned and reset per episode.
3. Retrieve `K` evidence tokens with `TaskConditionedEvidenceMemory`.
4. Insert `K` registered evidence placeholder tokens immediately before the current observation in both training and
   inference prompts, and replace their embeddings with retrieved evidence.
5. Build a single helper for embedding replacement and sequence bookkeeping. Use it in training forward, generation
   prefill, and `generate_latents`.
6. Keep evidence labels at `IGNORE_INDEX`. Recompute multimodal RoPE after insertion and assert image/evidence/traj token
   counts per sample. Never infer evidence positions from padded batch offsets.
7. Compute masked S2 token CE and trajectory flow-matching loss independently. Log the latent mean, standard deviation,
   norm, and cosine shift relative to the frozen baseline.

The placeholder route is chosen over appending anonymous embeddings because Qwen multimodal RoPE and generation cache
bookkeeping depend on an explicit aligned sequence. It also makes train/inference parity testable.

## 5. Initial trainable policy

The default first run trains only:

- evidence writer, task-state estimator, conditioned reader, stage head, and evidence-to-S2 projection;
- S2 LoRA parameters after `peft` is available;
- latent queries and `cond_projector` as the initial bridge.

The vision tower, RGB model, NextDiT action encoder/decoder, and NextDiT transformer are frozen initially. Gradients still
flow through frozen NextDiT to the S2 bridge. Expansion is evidence-driven: if fixed-target S1 controls pass but full
S2-to-S1 trajectories regress, unfreeze the smallest bridge component first.

## 6. Comparisons

| ID | History path | Spatial metadata | Task-conditioned read | Purpose |
| --- | --- | --- | --- | --- |
| B0 | checkpoint default | no | no | clean reproduction |
| B1 | checkpoint default | no | no | same-data continuation control |
| B2 | equal `K` history tokens | no | no | extra-context/token control |
| M1 | evidence memory | yes | no | spatial relation effect |
| M2 | evidence memory | yes | yes | task-conditioning effect |

Use the same new training data, updates, history frames, token count, optimizer budget, and seeds for B1/B2/M1/M2.
Validation-only historical audits never become training samples.

## 7. Verification gates

1. CPU protocol tests: shapes, padding invariance, empty history, causal-age validation, task conditioning, gradients,
   explicit trainable allowlist, and unmatched-pattern failure.
2. Dataset fixture: prove frame ids and metadata are prefix-only and train/inference selectors agree.
3. Tiny model integration: prove evidence positions, image positions, labels, RoPE, and `t_s_pos` remain aligned.
4. Overfit 8-32 authorized training samples: both S2 and trajectory losses decline; evidence and bridge gradients are
   nonzero; frozen parameters have no gradients.
5. Closed-loop smoke on 1-4 development episodes, then 200-500 update profiling after GPUs are free.

No larger run starts until the dataset source and split manifest are fixed, required packages exist in an isolated
environment, and the shared GPUs are available.
