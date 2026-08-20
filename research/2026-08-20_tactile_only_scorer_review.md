# Tactile-only Scorer Implementation Review

## Final formal contract

The formal scorer is now task-specific and receives only a predicted TacVAE
latent sequence:

```text
candidate action -> action-conditioned Foresight -> predicted tactile latent
                 -> task-specific tactile-only scorer -> quality score
```

Action, qpos, EEF, task ID, force, and action-derived proxy features are not
scorer inputs. Force6d may still be used offline to construct supervision
labels. Task selection is outside the model; Board, Vase, Card, Chip, and
Socket use independent checkpoints.

## Compatibility boundary

New formal checkpoints require schema version 2 and record task, horizon,
latent shape, temporal stride, future offset, and TacVAE SHA256 identity.
Legacy action-aware checkpoints are rejected by the formal runtime and remain
available only through explicitly named legacy code paths. Serving arms must
also set `legacy_ablation=true` before an old runtime can be selected, preventing
an existing action-aware config from silently becoming the formal path.

## Paper consistency finding

The current paper text describing an action branch that is detached during
`latent_only` guidance no longer describes the formal implementation. The new
scorer has no action branch in either training or inference. Before the next
paper release, the method and appendix should replace the detach description
with the strict path above and remove any claim that action is an optional
formal scorer input.

## Validation boundary

Synthetic contract and gradient tests pass. No new scorer checkpoint was
trained and no real-robot rollout was run in this change, so empirical scorer
accuracy and guidance gains remain to be established after task manifests and
labels are finalized.
