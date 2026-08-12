# ForeTac Figure 2 editable PowerPoint

## Deliverable

- `paper/figures/ForeTac_Figure2_editable_20260812.pptx`
- Desktop copy: `/home/chenshuai/Desktop/ForeTac_Figure2_editable_20260812.pptx`
- Reproducible builder: `scripts/build_foretac_figure2_ppt.py`

All model blocks, text, token rails, latent grids, status icons, locks, separators, and arrows are native PowerPoint shapes. Real camera/tactile/result images remain replaceable PowerPoint pictures.

## Design decisions

- Preserve the three offline modules over one full-width inference flow.
- Remove complete equations and losses from the architecture figure.
- Use locks instead of repeatedly writing `Frozen`.
- Use real imagery only at semantic endpoints: observed modalities, verified model reconstruction/prediction, and execution.
- Keep latent grids, queries, action samples, and encoded tokens abstract.
- The scorer visibly receives both clean-action and predicted-tactile trajectories.
- The only dashed backward path terminates at the selected late diffusion state `x^k`; the updated state then continues frozen denoising.

## Evidence boundaries

- The TacVAE reconstruction crop is a real decoded output from the project CI-VAE evaluation asset, not a ground-truth observation. It comes from the older insertion dataset because no per-frame PNG had yet been exported for the latest five-task checkpoint.
- The Foresight output crops are real held-out Board multi-step predictions. The available chip visualization uses a Card checkpoint and only `t+1`, so it was not used as evidence for parallel `H=16` prediction.
- The general model diagram retains RGB input because the generic architecture supports visual tokens. Current action-conditioned Board serving checkpoints are marker-only; the main paper text/caption must distinguish the general architecture from that deployment configuration.

## Verification

- The PPT was generated with python-pptx, reopened by LibreOffice, exported to PDF, and rasterized for visual inspection.
- The final slide has a pure white background, no complete formulas, no crossed primary-flow arrows, and no duplicated panel titles.
