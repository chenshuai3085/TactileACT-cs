# Five-task TacVAE retraining

Date: 2026-08-03

## Objective

Train a new 45D TacVAE from scratch on all compatible Board, Vase, Card,
Chip, and Socket marker data. The model uses left tactile marker sequences,
an 8-frame temporal window, five latent channels, and a `5x3x3=45D` latent
representation per aligned frame.

## Canonical dataset

Exact duplicates, mirrored disks, derived action relabels, malformed HDF5
files, and episodes without marker offsets are excluded before splitting.

| Task | Episodes | Marker frames | Train / val / test episodes |
|---|---:|---:|---:|
| Board | 979 | 832,659 | 783 / 98 / 98 |
| Vase | 238 | 96,136 | 190 / 24 / 24 |
| Card | 393 | 172,828 | 315 / 39 / 39 |
| Chip | 42 | 20,517 | 34 / 4 / 4 |
| Socket | 5,109 | 1,765,805 | 4,085 / 512 / 512 |
| Total | 6,761 | 2,887,945 | 5,407 / 677 / 677 |

The usable Chip data are the 42 episodes from `260710_v8j_jiashupian`.
All have left/right `marker_offset` with shape `T x 9 x 9 x 2`, left/right
`force6d`, and `ft`. The 200 episodes from 260702/260703 contain tactile
images but no marker offsets, so they cannot enter this marker TacVAE.

## Reproducibility contract

- Source configuration: `TFAC_V5/config_multitask_tacvae_5task_sources.json`
- Manifest builder: `TFAC_V5/build_multitask_tacvae_5task_manifest.py`
- Manifest: `outputs/multitask_tacvae_5task_20260803/manifest.jsonl`
- Manifest file SHA256: `31c566c63ebdbca655d51977f70dec98aa1b5da22b9d00dc3a2392c232aea96b`
- Training canonical manifest hash: `0da3ca17844f1d8f4ff30ab7bb9e83bf03de4aadbf982aee9161033e6e8d9783`
- Split seed: 42; splitting is episode-level within each canonical source.
- Every episode is checked for readable HDF5, both finite marker streams,
  matching lengths, exact shape, minimum length, and unique full-sequence
  marker fingerprint.

## Training protocol

Each balanced training sample follows:

```text
uniform task -> uniform condition -> uniform episode -> uniform stride-2 window
```

Each batch therefore contains exactly 20% samples from each task. Global
normalization uses train episodes only and equal weights at task, condition,
and episode levels. Validation covers every held-out episode with non-overlap
stride-8 windows. Checkpoint selection and early stopping use the equal-task
macro MSE; micro and per-task MSE are also recorded.

The entry point saves atomic `latest.pt` and `best_macro.pt` checkpoints and
supports epoch-boundary resume with optimizer, RNG, history, manifest hash,
normalization, and protocol signature checks.

## Verification

- Five focused `unittest` checks passed.
- Python compilation and `git diff --check` passed.
- The full manifest validated all 6,761 episodes with zero unexpected invalid
  files after configured exclusions.
- A full-data CPU smoke run completed two balanced train batches plus all
  36,036 non-overlapping validation windows. Initial macro MSE was `2.248121`;
  per-task MSE was Board `2.308872`, Card `2.045069`, Chip `1.604532`, Socket
  `2.823233`, and Vase `2.458899`. These are initialization smoke values, not
  trained-model results.

## Formal run

On 2026-08-04, `nvidia_uvm` was reloaded through desktop Polkit and CUDA tensor
allocation passed on the RTX 4090. A full-data GPU smoke run completed before
the formal run was launched.

The formal scratch run uses:

```text
tmux: tacvae_5task_20260804
output: outputs/multitask_tacvae_5task_20260804/train
log: outputs/multitask_tacvae_5task_20260804/train.log
epochs: 300
balanced batches per epoch: 256
batch size: 500 (100 samples per task)
validation windows: 36,036 non-overlapping held-out windows
```

The run completed all 300 epochs normally without early stopping, NaN values,
or checkpoint corruption. The selected checkpoint is `best_macro.pt` from
epoch 296. Its held-out validation results are:

| Metric | MSE |
|---|---:|
| Equal-task macro | 0.007522 |
| Window-weighted micro | 0.009992 |
| Board | 0.008370 |
| Card | 0.004956 |
| Chip | 0.001177 |
| Socket | 0.011246 |
| Vase | 0.011863 |

The final epoch-300 checkpoint has macro MSE `0.007694`; therefore downstream
representation experiments must use `best_macro.pt`, not `latest.pt`. Both
checkpoints load successfully on CPU, all saved model tensors and all 300
epochs of recorded metrics are finite, and the best checkpoint retains the
expected canonical manifest hash
`0da3ca17844f1d8f4ff30ab7bb9e83bf03de4aadbf982aee9161033e6e8d9783`.

The next representation audit will freeze the epoch-296 encoder and reuse the
same held-out episode split. Results should be reported separately by task,
contact phase, and physical contact-quality regime so that task identity is
not mistaken for contact-state separation.
