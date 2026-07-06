# v8j huaping/card data analysis

Source groups:
- `huaping_main`: `/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630`
- `huaping_no_grasp_change`: `/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630/无夹取位置变化`
- `card_success`: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/success`
- `card_bounce_hengxiang`: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/bounce_hengxiang`
- `card_bounce_jiaozhun`: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/bounce_jiaozhun`

## Group summary

| group | episodes | length median | duration median(s) | eef path median(m) | ft |F| p95 median | left |F| p95 median | right |F| p95 median | marker L/R mean median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `card_bounce_hengxiang` | 20 | 418 | 20.9 | 0.350 | 15.90 | 2.45 | 41.05 | 1.525/2.664 |
| `card_bounce_jiaozhun` | 20 | 494 | 24.7 | 0.374 | 15.61 | 1.83 | 121.09 | 1.554/2.823 |
| `card_success` | 20 | 458 | 22.9 | 0.350 | 15.77 | 1.51 | 120.42 | 1.424/2.771 |
| `huaping_main` | 100 | 438 | 21.9 | 0.363 | 16.37 | 8.81 | 90.67 | 2.743/3.382 |
| `huaping_no_grasp_change` | 30 | 397 | 19.9 | 0.368 | 16.55 | 2.19 | 134.03 | 1.574/2.306 |

## Label consistency check

The most important data-quality finding is that `actions/eef_abs` is aligned with
`observations/proprio_eef` for both tasks, but huaping `actions/joint_abs` contains
large joint-solution jumps that are not present in the measured joints.

| group | episodes | eef action vs next proprio pos err median(mm) | joint action vs next proprio err median(deg) | action joint step max median(deg) | measured joint step max median(deg) |
|---|---:|---:|---:|---:|---:|
| `card_success` | 20 | 0.359 | 0.098 | 2.03 | 2.07 |
| `card_bounce_hengxiang` | 20 | 0.445 | 0.105 | 1.40 | 1.46 |
| `card_bounce_jiaozhun` | 20 | 0.371 | 0.094 | 1.69 | 1.67 |
| `huaping_main` | 100 | 0.434 | 2.184 | 227.36 | 1.11 |
| `huaping_no_grasp_change` | 30 | 0.505 | 2.043 | 213.10 | 1.21 |

Interpretation:
- Card `joint_abs` labels are consistent with measured joint states and are usable for joint-absolute DP training.
- Huaping `eef_abs` labels are usable: median position mismatch to next measured EEF is about 0.4-0.5 mm and orientation mismatch is about 0.001 rad.
- Huaping `joint_abs` labels should not be used directly for joint-absolute DP training. Every huaping-main episode has at least one 200 deg scale action jump, while measured joints move only about 1 deg at the same scale. This looks like an IK/alternate-joint-solution jump in the saved action label, not a real robot motion.

Recommended use:
- For huaping, train/deploy with `eef_abs`/`eef_rel`, or regenerate joint labels from measured `observations/proprio_joint` if a joint policy is required.
- For card, `joint_abs` training is acceptable from a label-consistency standpoint.

## Artifacts

- `episode_summary.csv`: `/home/chenshuai/Project/TactileACT-cs/outputs/v8j_huaping_card_analysis/episode_summary.csv`
- `group_summary.csv`: `/home/chenshuai/Project/TactileACT-cs/outputs/v8j_huaping_card_analysis/group_summary.csv`
- `group_summary.json`: `/home/chenshuai/Project/TactileACT-cs/outputs/v8j_huaping_card_analysis/group_summary.json`
- plots: `/home/chenshuai/Project/TactileACT-cs/outputs/v8j_huaping_card_analysis`
