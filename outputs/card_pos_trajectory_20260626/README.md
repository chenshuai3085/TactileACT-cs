# Card Positive Trajectories 2026-06-26

Source success dir: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260615_v8l_card/success`

Generated command:

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python data_collection/generate_trajectories.py --task card --type positive --batch 100 --randomize --visualize --save_dir outputs/card_pos_trajectory_20260626 --seed 260626
```

Summary:

- Count: 100 trajectories
- Length: mean 427.01 steps, range 354..569 steps
- Duration: mean 21.351 s, range 17.7..28.45 s
- XYZ min mm: [181.79200744628906, -5.73199987411499, 165.6269989013672]
- XYZ max mm: [391.6889953613281, 32.38999938964844, 229.45899963378906]
- Max speed: mean 40.054 mm/s, p95 46.299 mm/s, max 47.951 mm/s
- Dry-run checked: `card_pos_000.npy`, `card_pos_015.npy`

Notes:

- Generated from 50 success HDF5 episodes under `success/`.
- Euler angles are unwrapped for template construction and wrapped back to [-pi, pi] before saving.
- A conservative 48 mm/s generator speed cap lengthens trajectories when needed.
