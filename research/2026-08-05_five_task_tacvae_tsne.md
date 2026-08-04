# Five-task TacVAE representation audit

Date: 2026-08-05

## Objective

Evaluate whether the newly retrained five-task TacVAE separates tactile
contact modes, while explicitly checking whether visible t-SNE islands are
instead caused by task, collection source, or episode identity.

## Reproducibility contract

- Checkpoint: `outputs/multitask_tacvae_5task_20260804/train/best_macro.pt`
- Selected checkpoint epoch: 296
- Representation: deterministic encoder `mu_last`, flattened from `5x3x3` to 45D
- Manifest: `outputs/multitask_tacvae_5task_20260803/manifest.jsonl`
- Canonical manifest hash: `0da3ca17844f1d8f4ff30ab7bb9e83bf03de4aadbf982aee9161033e6e8d9783`
- Split: untouched `test`; it was not used for training or checkpoint selection
- Sampling seed: 42
- t-SNE: standardized 45D -> PCA-30 -> t-SNE, perplexity 50, 2,000 iterations
- Script: `TFAC_V5/visualize_five_task_tacvae_tsne.py`
- Output: `outputs/five_task_tacvae_tsne_20260805/`

All alternative colorings within one analysis reuse exactly the same sample
manifest, frozen latent vectors, PCA features, and t-SNE coordinates.

## Contact-phase analysis

The main cross-task audit uses three signal-derived phases: conservative
pre-contact, contact onset, and sustained contact. It contains 2,400 windows
from 459 test episodes, exactly balanced as 160 windows for every one of the
`5 tasks x 3 phases` cells. Adjacent 8-frame windows use stride 2 and therefore
share six frames; episode-grouped metrics are used to avoid treating them as
independent trials.

| Measurement | Contact phase | Task |
|---|---:|---:|
| Standardized 45D silhouette | -0.0183 | 0.1016 |
| t-SNE silhouette | -0.0168 | 0.0801 |
| Episode-grouped balanced accuracy | 44.23% | 91.56% |
| Chance balanced accuracy | 33.33% | 20.00% |
| 45D 10-NN purity | 54.76% | 93.96% |
| Random-label 10-NN baseline | 33.31% | 19.97% |

The representation does not form three globally separated cross-task contact
phase clusters. It contains some local phase information, as the grouped
phase probe is above chance, but task identity is substantially stronger.

## Why the plot forms many islands

The 45D 10-nearest-neighbor purity is `78.49%` for collection source and
`51.51%` for exact episode, compared with random baselines of `6.46%` and
`1.38%`. The corresponding t-SNE purities are `77.55%` and `50.18%`.
Therefore, the small separated islands mainly trace task/source-specific
sensor baselines and temporally adjacent episode trajectories. Their absolute
2D distances must not be interpreted as calibrated physical distances between
contact modes.

## Six-state quality-proxy analysis

The expanded quality plot contains 3,000 windows from 455 test episodes, with
500 windows in each category: No contact, Stable contact, Contact
dropout/insufficient pressure, Excessive pressure, Slip/Oscillation, and
Jamming/Bounce.

These are weak labels, not a six-class ground-truth benchmark:

- No contact is a conservative marker-and-force pre-contact rule.
- Stable uses successful/normal collection conditions plus a detected contact window.
- Dropout/insufficient pressure, excessive pressure, and slip/oscillation proxies exist only in Board data.
- Jamming/Bounce uses Card and Socket collision/bounce collection conditions.
- Chip has only four test episodes and no slip or crush annotation.

The quality-proxy 45D silhouette is `-0.0145`, while the episode-grouped
six-class balanced accuracy is `68.19%` versus `16.67%` chance. Thus a linear
readout can recover substantial condition information even though it is not
organized as six compact global clusters. However, state and task are
confounded (`Cramer's V = 0.4596`), so this result does not demonstrate
task-invariant physical contact-quality classes.

## Main conclusion

The retrained TacVAE learned a useful but strongly domain-structured tactile
representation. It encodes task/source identity very clearly and retains
decodable contact-quality information, but it does not spontaneously align
pre-contact, onset, and sustained-contact modes across all five tasks into a
shared global geometry. A task-invariant contact representation would require
an explicit supervised/contrastive objective, source-balanced batches, and
manually audited event labels, especially for Chip slip/crush and Socket jam.

## Artifacts

- `contact_phase_dual.{png,pdf}`: phase colors and task colors on identical coordinates
- `contact_phase_task_facets.{png,pdf}`: five task panels on the same coordinates
- `contact_quality_proxy_dual.{png,pdf}`: six proxies and task/domain comparison
- `contact_quality_proxy_task_facets.{png,pdf}`: task-specific proxy panels
- `heldout_samples_manifest.csv`: exact fixed test windows and label provenance
- `contact_phase_features.npz`, `contact_quality_proxy_features.npz`: 45D/PCA/t-SNE caches
- `summary.json`: complete counts, metrics, contingency tables, and caveats
