# tacfore

tacfore is a research codebase for tactile-aware robot policy learning.
The current tracked project focuses on a Diffusion Policy stack with tactile
latent foresight, quality scoring, gradient guidance, and real-robot serving.

The repository has been pruned to the active DP + foresight + scoring +
guidance workflow. Older CLIP pretraining code, legacy TFAC variants, working
notes, and the root `scripts/` directory are not part of the tracked source.

## Features

- Tactile VAE pretraining for marker-offset latent representations.
- Multi-step tactile foresight models that predict future tactile consequences
  from current observations and action chunks.
- Diffusion Policy training with visual observations, proprioception, and frozen
  tactile VAE features.
- Board-wiping and insertion quality scorers for classifier/energy guidance.
- WebSocket/TCP serving utilities for baseline DP, foresight-guided DP, and
  TacQuality-guided DP rollouts.
- Robot data collection tools for trajectory generation, replay, and HDF5
  recording.
- Vendored references for DETR backbones and the official Diffusion Policy
  implementation.

## Repository Layout

```text
TFAC_V5/
  tactile_vae.py                         Tactile VAE model
  pretrain_tactile_vae.py                tactile VAE training
  pretrain_latent_foresight*.py          latent foresight training
  foresight_multistep.py                 multi-step foresight model
  board_chunk_energy/                    board marker/action scorer
  board_latent_energy/                   board latent/action scorer
  tac_quality_energy/                    reusable TacQuality guidance modules

diffusion/
  train_dp_tac_concat.py                 DP + frozen TactileVAE training
  train_dp_tac_vae.py                    tactile VAE DP baseline
  train_dp_foresight_joint.py            DP/foresight joint training experiments
  network.py, dataset.py, utils.py       DP model and data utilities

for_show_xiaomi/
  serve_dp_policy.py                     baseline DP policy server
  serve_board_dp_foresight_guided.py     board foresight + latent-energy server
  serve_dp_tac_quality_guided.py         TacQuality guided server
  ws_server.py, ws_client.py             robot communication utilities
  eval_*_rollouts.py                     rollout evaluation utilities

data_collection/
  generate_trajectories.py               programmatic EEF trajectory generation
  auto_replay.py                         robot replay and HDF5 recording
  simple_record_pih_v8l_forceviz.py      force-visualized recording utility

diffusion_policy_official/               upstream Diffusion Policy source copy
detr/                                    DETR backbone dependency copy
utils.py                                 shared helpers
```

## Installation

This repository currently does not provide a single pinned root environment
file. The working setup is a Python/PyTorch research environment named
`TactileACT`.

```bash
conda create -n TactileACT python=3.8 -y
conda activate TactileACT
pip install torch torchvision
pip install diffusers h5py numpy opencv-python matplotlib tqdm einops pyyaml
pip install scipy scikit-learn msgpack msgpack-numpy
cd detr && pip install -e . && cd ..
```

If you use the vendored official Diffusion Policy code directly, install its
extra dependencies from `diffusion_policy_official/conda_environment*.yaml` or
install it in editable mode:

```bash
cd diffusion_policy_official
pip install -e .
cd ..
```

Robot serving requires the local robot, camera, force/tactile, and network setup
used in the Xiaomi deployment environment. Paths in several configs point to
`/home/chenshuai/...`; update them for a different machine.

## Data Format

Training and rollout code expects episode HDF5 files. The data collection tools
write the same schema used by the DP and foresight loaders, including:

```text
actions/joint_abs
actions/eef_abs
observations/images/global
observations/images/wrist
observations/proprio_joint
observations/proprio_eef
observations/tac/left/marker_offset
observations/tac/right/marker_offset
observations/tac/left/force6d
observations/tac/right/force6d
ft
joint_current
```

Generated datasets, model checkpoints, rollout logs, image caches, and output
directories are intentionally not tracked in git.

## Training

### 1. Train a Tactile VAE

```bash
python TFAC_V5/pretrain_tactile_vae.py \
  --data_dirs /path/to/hdf5_dataset_a /path/to/hdf5_dataset_b \
  --output_dir /path/to/output/tactile_vae \
  --sides left right \
  --latent_dim 16 \
  --temporal_window 8 \
  --epochs 200 \
  --batch_size 128
```

### 2. Train Multi-step Latent Foresight

Use one of the tracked config files under `TFAC_V5/`, then edit dataset and
checkpoint paths as needed.

```bash
python TFAC_V5/pretrain_latent_foresight_multistep.py \
  --config TFAC_V5/config_pretrain_foresight_board_multistep16_boardvae_marker_only.json
```

### 3. Train Diffusion Policy with Tactile Concatenation

```bash
python diffusion/train_dp_tac_concat.py \
  --dataset_dir /path/to/hdf5_dataset \
  --save_dir /path/to/output/dp_tac_concat \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --vae_checkpoint /path/to/best_tactile_vae.pt \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --epochs 600 \
  --batch_size 64 \
  --gpu 0
```

For large image datasets, prefer `--lazy_images` and optionally
`--image_cache_dir /path/to/cache`.

### 4. Train Board Quality/Energy Scorers

Audit the built-in board data index before training:

```bash
python -m TFAC_V5.board_chunk_energy.train --audit_only
python -m TFAC_V5.board_latent_energy.train --audit_only \
  --tactile_vae_ckpt /path/to/best_tactile_vae.pt
```

Train a marker/action chunk scorer:

```bash
python -m TFAC_V5.board_chunk_energy.train \
  --output_dir /path/to/output/board_chunk_energy \
  --epochs 40 \
  --batch_size 128
```

Train a latent/action scorer:

```bash
python -m TFAC_V5.board_latent_energy.train \
  --output_dir /path/to/output/board_latent_energy \
  --tactile_vae_ckpt /path/to/best_tactile_vae.pt \
  --epochs 40 \
  --batch_size 128
```

## Serving and Deployment

Baseline DP server:

```bash
python -m for_show_xiaomi.serve_dp_policy \
  --ckpt_dir /path/to/dp_checkpoint_dir \
  --ckpt_name dp_best.pth \
  --host 0.0.0.0 \
  --port 8766 \
  --gpu 0
```

Board foresight + latent-energy guided server:

```bash
python -m for_show_xiaomi.serve_board_dp_foresight_guided \
  --arm latent_energy_guided \
  --dp_ckpt /path/to/dp_best.pth \
  --foresight_dir /path/to/foresight_dir \
  --foresight_ckpt /path/to/foresight_best.ckpt \
  --scorer_ckpt /path/to/board_latent_energy_best.pt \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --guidance_path latent_only \
  --score_mode expert_margin \
  --guidance_steps 5 \
  --guidance_scale 0.003 \
  --send_guidance_report
```

TacQuality guided server:

```bash
python -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --ckpt_dir /path/to/dp_checkpoint_dir \
  --ckpt_name dp_best.pth \
  --foresight_dir /path/to/foresight_dir \
  --foresight_ckpt /path/to/foresight_best.ckpt \
  --rollout_arm_config /path/to/rollout_arm_config.json \
  --host 0.0.0.0 \
  --port 8766 \
  --gpu 0 \
  --send_guidance_report
```

The tracked `for_show_xiaomi/guide_forshow.sh` is a command runbook for the
current board rollout setup. It is intended for copying individual commands, not
for executing the whole file as a script.

## Data Collection

Generate candidate end-effector trajectories:

```bash
python data_collection/generate_trajectories.py \
  --task wipe \
  --type positive \
  --pattern straight \
  --batch 10 \
  --randomize \
  --save_dir /path/to/trajectories
```

Dry-run safety checks and replay on the robot:

```bash
python data_collection/auto_replay.py \
  --traj_dir /path/to/trajectories \
  --save_dir /path/to/hdf5_output \
  --dry_run

python data_collection/auto_replay.py \
  --traj_dir /path/to/trajectories \
  --save_dir /path/to/hdf5_output
```

See `data_collection/README.md` for task-specific positive/negative trajectory
generation details and safety checks.

## Verification

Useful lightweight checks:

```bash
python -m TFAC_V5.board_chunk_energy.train --audit_only
python -m TFAC_V5.board_latent_energy.train --audit_only --tactile_vae_ckpt /path/to/best_tactile_vae.pt
python -m for_show_xiaomi.serve_dp_tac_quality_guided --help
python diffusion/train_dp_tac_concat.py --help
```

Full training and serving tests require local HDF5 data, trained checkpoints,
GPU resources, and the robot deployment environment.

## Git-tracked Scope

The current tracked source contains only the active project modules:

- `TFAC_V5/`
- `diffusion/`
- `for_show_xiaomi/`
- `data_collection/`
- `diffusion_policy_official/`
- `detr/`
- `utils.py`

The root `scripts/` directory, older TFAC versions, CLIP pretraining assets, and
working-record files are intentionally not tracked.

## License and Third-party Code

This repository does not currently include a root project license file. Add one
before public redistribution.

Third-party or vendored code keeps its own license information:

- `diffusion_policy_official/` follows the license and attribution in that
  directory.
- `detr/` follows the license and attribution in that directory.

## Citation

If you use the vendored Diffusion Policy or DETR components, cite their original
projects as appropriate. Add the tacfore paper/project citation here when
the public citation is finalized.
