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

The draft has been rewritten from the older TacScore/CoRL candidate-reranking
story into the current route:

```text
base policy -> multi-step tactile foresight -> TacQuality score guidance
```

The main method name in the draft is Proactive Tactile Guidance (PTG).

## Open Items

- Replace placeholder task/method figures with final diagrams.
- Fill real robot paired baseline-versus-guided tables after controlled rollouts.
- Keep offline scorer/foresight results clearly separated from online task claims.
- For initial RA-L submission, keep author information anonymous.
