# 2026-07-05 任务 TactileVAE 训练计划与夹薯片数据阻塞

## 计划

本轮为新任务分别训练 task-local marker TactileVAE，继续使用旧版 `TFAC_V5/tactile_vae.py`，保证下游 `diffusion/train_dp_tac_concat.py` 可以直接加载。

统一设置：

- `side=left`
- `temporal_window=8`
- `latent_dim=16`
- `sample_stride=2`
- `epochs=150`
- `batch_size=512`
- `kl_weight=1e-6`
- `direction_weight=0.2`

## Huaping VAE

使用：

- `/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630`

排除：

- `/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630/无夹取位置变化`

脚本：

- `scripts/pretrain/run_tactile_vae_huaping_260630_left.sh`

输出：

- `/media/chenshuai/czy_data22/pih_output/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150`

## Card VAE

使用：

- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260701_v8j_card/peg_in_hole_0701/bounce_hengxiang`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260701_v8j_card/peg_in_hole_0701/bounce_jiaozhun`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/bounce_hengxiang`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/bounce_jiaozhun`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/success`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260626_replay_card/card_pos_keepsteps_20260626`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260615_v8l_card/bounce`
- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260615_v8l_card/success`

排除：

- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/260630_v8j_huaping`

原因：这个目录是 huaping 数据，不应混进 card tactile representation。

脚本：

- `scripts/pretrain/run_tactile_vae_card_260615_260626_260629_260701_left.sh`

输出：

- `/media/chenshuai/EXTERNAL_USB/pih_output/tactile_vae_card_260615_260626_260629_260701_left_tw8_ld16_s2_e150`

## 夹薯片阻塞

用户希望先训练夹薯片 TactileVAE，并用其训练有触觉 DP。但当前 260702/260703 夹薯片 HDF5 中没有现有链路需要的 marker field：

- 缺失：`observations/tac/left/marker_offset`
- 缺失：`observations/tac/right/marker_offset`
- 存在：`observations/tac/left/img`
- 存在：`observations/tac/right/img`

因此当前不能直接使用 `TFAC_V5/pretrain_tactile_vae.py` 或 `diffusion/train_dp_tac_concat.py` 训练夹薯片的 marker TactileVAE/concat DP。

可选方案：

1. 重新采集或离线补算 `marker_offset`，之后沿用现有 marker TactileVAE + concat DP 链路。
2. 改做 raw tactile image policy，例如基于 `diffusion/train_dp_tac_img.py` 的 global+wrist+gelsight image DP；这不再是“之前一样”的 frozen TactileVAE concat 版本。
3. 新写 tactile image VAE，再接 DP；这是新模型链路，需要额外实现和部署适配。

当前先推进 huaping/card 的 marker TactileVAE 训练；夹薯片触觉 DP 等用户确认采用哪条触觉表示链路后再启动。
