# Board Chunk Energy

Blackboard wiping scorer for action-conditioned tactile consequence quality.

The first version is task-specific and chunk-level:

```text
joint action chunk + future left marker chunk -> expert-likeness energy
```

Class mapping:

```text
0 expert              260609/wipe_pos_straight_z124_125_150_20260609
1 pressure_too_small  260609/z_too_high
2 pressure_too_large  260610/z_too_low
3 pressure_unstable   260610/z_too_oscillate
```

The deployed scalar is:

```text
score_good = prototype logit for class 0 expert
energy = -score_good
```

Training defaults to `CE_4class + margin(expert > negatives)`.  Set
`--supcon_weight` above zero to add supervised contrastive learning and compare
whether it helps.

Minimal audit:

```bash
conda run -n TactileACT python -m TFAC_V5.board_chunk_energy.train --audit_only
```

Minimal smoke training:

```bash
conda run -n TactileACT python -m TFAC_V5.board_chunk_energy.train --epochs 2 --max_episodes_per_class 4 --batch_size 32
```
