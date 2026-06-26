# Card Positive Keep-Steps Trajectories

Generated from `/media/chenshuai/EXTERNAL_USB/pih_dataset/260615_v8l_card/success`.

These trajectories preserve each source success trajectory's original step count and route skeleton.
There is no time scaling or lengthening. Output `i` uses source `i % 50`.

- Count: 100
- Source count: 50
- Step range: 344..569
- Max speed: mean 53.128 mm/s, max 84.815 mm/s
- XYZ deviation from source: mean 0.134 mm, max 0.958 mm

Command:

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python data_collection/generate_card_success_keepsteps.py --save_dir outputs/card_pos_keepsteps_20260626 --batch 100 --seed 260626 --visualize
```
