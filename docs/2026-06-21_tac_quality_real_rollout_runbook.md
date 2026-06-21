# TacQuality Real Rollout Runbook

## Purpose

This runbook is for collecting the missing real robot evidence for TacQuality
classifier/scorer guidance.

Current status before this runbook:

```text
offline scorer quality                 ready
Foresight-gradient guidance             ready
server-side logging schema              ready
paired rollout manifest                 ready
real paired rollout evidence            missing
```

Do not treat a dry-run, synthetic log, offline audit, or real-HDF5-window audit
as real robot performance evidence.  Final evidence requires non-synthetic
server-side `force_trace.csv` files with explicit `pair_id` metadata.

## Two Evidence Tracks

### Track A: Current Default Full Evidence

This is the default combined evidence route for the current scorecard.

```text
board:
  baseline              port 8765, arm baseline
  guided                port 8766, arm marker_joint_s12_guided
  rollout root          /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer
  manifest              /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv

insertion:
  baseline              port 8785, arm baseline
  guided                port 8786, arm good_margin_guided
  rollout root          /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer
  manifest              /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv
```

Planned trials:

```text
board      3 baseline + 3 guided
insertion  3 baseline + 3 guided
total      12 trials
```

### Track B: Force-Aware Board Research Evidence

This is the stronger board research candidate.  It should be collected in a
separate root so it does not mix with Track A.

```text
board:
  baseline              port 8765, arm baseline
  guided                port 8769, arm force_aware_guided
  rollout root          /home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer
  manifest              /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.csv
```

Planned trials:

```text
board      3 baseline + 3 guided
total      6 trials
```

## Step 1: Start Servers

Open `for_show_xiaomi/guide_forshow.sh` and copy only the needed blocks.  Do
not run the script file directly; it exits after printing the command index.

For Track A:

```text
block 1   board baseline, port 8765
block 2   board marker_joint_s12 guided, port 8766
block 3   insertion baseline, port 8785
block 4   insertion good_margin guided, port 8786
```

For Track B:

```text
block 1    board baseline, port 8765
block 2c   board force_aware guided, port 8769
```

Preflight:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/preflight_tac_quality_deploy.py
conda run --no-capture-output -n TactileACT python for_show_xiaomi/audit_tac_quality_guidance_state.py
```

Port check:

```bash
ss -ltnp | grep -E ':8765|:8766|:8769|:8785|:8786' || true
pgrep -af 'serve_dp_tac_quality_guided' || true
```

## Step 2: Run Client Commands From The Manifest

Use the per-row `client_command` in the manifest CSV/MD.  These commands pass
the required metadata to the server:

```text
rollout_pair_id
rollout_trial_order
rollout_task
rollout_group
rollout_server_arm
rollout_manifest_csv
```

Do not use the simplified port-only client commands for final evidence.  They
are only connectivity helpers; without manifest metadata, the explicit-pair
evaluator will not be final-evidence clean.

Track A manifest:

```bash
sed -n '1,240p' /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.md
```

Track B manifest:

```bash
sed -n '1,220p' /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.md
```

After each client trial, confirm that a new server-side directory exists and
contains:

```text
force_trace.csv
force_trace.npz
force_curve.png
metadata.json
```

Expected directory patterns:

```text
Track A board baseline:
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer/baseline/*_port8765_episode*

Track A board guided:
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer/guided/*_port8766_episode*

Track A insertion baseline:
/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer/baseline/*_port8785_episode*

Track A insertion guided:
/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer/guided/*_port8786_episode*

Track B force-aware board baseline:
/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer/baseline/*_port8765_episode*

Track B force-aware board guided:
/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer/guided/*_port8769_episode*
```

## Step 3: Fill Insertion Outcome Metadata

Board wiping can be evaluated from force traces directly.  Insertion needs
task outcome metadata.

For each insertion row in the Track A manifest, fill:

```text
success          true/false
stopped_early    true/false
bounce_count     integer
retry_count      integer
notes            optional text
```

Manifest:

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv
```

After editing the CSV, apply metadata back to the matched server-side trial
directories:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/apply_rollout_manifest_metadata.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv
```

For board-only Track B, no insertion metadata is needed.

## Step 4: Coverage Audit

Track A:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/audit_real_rollout_coverage.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv
```

The coverage markdown lists every missing/problem row and includes the exact
manifest `client_command` to copy for each row:

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.md
```

Expected before final evaluation:

```text
status_counts should have no missing rows
board n_complete_pairs >= 3
insertion n_complete_pairs >= 3
real_rollout_evidence_complete = true
```

Track B:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/audit_real_rollout_coverage.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.csv \
  --tag board_force_aware_coverage
```

The force-aware coverage markdown also lists the exact missing-row commands:

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/board_force_aware_coverage/tac_quality_real_rollout_coverage.md
```

Expected before final board-only evaluation:

```text
status_counts should have no missing rows
board n_complete_pairs >= 3
board_ready_for_real_eval = true
```

## Step 5: Evaluate

Track A final evaluator:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_tac_quality_real_rollouts.py \
  --board_root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  --insertion_root /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_eval \
  --tag current_s12_good_margin_tac_quality \
  --board_expected_baseline_arm baseline \
  --board_expected_guided_arm marker_joint_s12_guided \
  --insertion_expected_baseline_arm baseline \
  --insertion_expected_guided_arm good_margin_guided \
  --pairing_strategy explicit
```

Track B force-aware board-only evaluator:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_tac_quality_real_rollouts.py \
  --board_root /home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer \
  --skip_insertion \
  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_eval \
  --tag board_force_aware_tac_quality \
  --board_expected_baseline_arm baseline \
  --board_expected_guided_arm force_aware_guided \
  --min_board_pairs 3 \
  --pairing_strategy explicit
```

## Step 6: Refresh Scorecard

After real evaluation passes, refresh the evidence bundle and scorecard:

```bash
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/refresh_tac_quality_evidence_bundle.py
conda run --no-capture-output -n TactileACT python TFAC_V5/tac_quality_energy/build_current_scorecard.py
```

Inspect:

```bash
python -m json.tool /home/chenshuai/Project/output/tac_quality_current_scorecard/current_tac_quality_scorecard.json | \
  rg -n 'real_paired_rollout_complete|goal_complete|force_aware_board_real_rollout_complete' -C 2
```

## Acceptance Criteria

Track A can support the main real-evidence claim only if:

```text
board has >= 3 explicit baseline/guided pairs
insertion has >= 3 explicit baseline/guided pairs
insertion outcome metadata is complete
no synthetic logs are counted
at least one board metric improves in expected direction
at least one insertion metric improves in expected direction
```

Board expected directions:

```text
quality_force_in_band_guided_minus_baseline       positive
quality_force_smooth_guided_minus_baseline        positive
quality_force_abs_error_baseline_minus_guided     positive
```

Insertion expected directions:

```text
success_guided_minus_baseline       positive
bounce_baseline_minus_guided        positive
retry_baseline_minus_guided         positive
```

Track B can support a force-aware board claim only if:

```text
board has >= 3 explicit baseline/force_aware_guided pairs
no synthetic logs are counted
at least one board force metric improves in expected direction
```

## Failure Handling

If coverage reports `missing`:

```text
The planned manifest row has no matching force_trace.csv.
Run the missing client command again or check whether the server was started
with the expected port, arm, and server_rollout_log_dir.
```

If coverage reports `metadata_problem`:

```text
The trial exists, but task/arm/port/pair_id/synthetic metadata failed.
Check that the client command came from the manifest and that the server arm
matches the row.
```

If evaluator says `missing_explicit_pair_id`:

```text
Run apply_rollout_manifest_metadata.py, then rerun the evaluator.
If pair_id is still missing, the simplified port-only client command was used
instead of the manifest command.
```

If insertion says metadata is incomplete:

```text
Fill success, stopped_early, bounce_count, and retry_count in the manifest CSV,
then rerun apply_rollout_manifest_metadata.py.
```

If guided force/action looks worse:

```text
Do not claim improvement.  Keep the logs as negative evidence and inspect:
  guidance_report fields in force_trace.csv
  force_curve.png per trial
  score_delta / action_delta columns
  contact gate behavior
  whether baseline and guided trials were paired under comparable initial states
```
