# RA-L Draft Notes

This directory now contains an IEEE RA-L initial-submission LaTeX draft.

## Template Choice

According to the IEEE Robotics and Automation Letters author information page,
initial submissions and resubmissions use the RAS conference LaTeX template:

```tex
\documentclass[letterpaper, 10 pt, conference]{ieeeconf}
```

Final accepted manuscripts are converted to IEEE journal style later. For this
reason, `main.tex` uses `ieeeconf.cls`, not the final `IEEEtran` journal class.

The local template files were downloaded from PaperPlaza/RAS support:

- `ieeeconf.cls` from `https://ras.papercept.net/conferences/support/files/ieeeconf.zip`
- `IEEEtran.bst` and `IEEEabrv.bib` from `https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip`

## Current Paper Direction

The draft has been rewritten from the older TacScore/CoRL and PTG wording into
the current project-page route:

```text
ForeTac: Predict tactile consequences -> score contact quality -> guide action
```

The provisional method/title name in the draft is now `ForeTac`, matching the
project homepage. The name can still be changed later if the project branding
changes.

## Open Items

- Fill real robot paired baseline-versus-guided tables after controlled rollouts.
- Fill the ablation table after no-guidance, reranking, horizon, score-mode, and
  guidance-path experiments are completed.
- Keep offline scorer/foresight results clearly separated from online task claims.
- For initial RA-L submission, keep author information anonymous.
