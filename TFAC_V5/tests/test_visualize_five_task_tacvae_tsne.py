import unittest
from types import SimpleNamespace

from TFAC_V5.visualize_five_task_tacvae_tsne import (
    PHASES,
    QUALITY_STATES,
    TASKS,
    balance_quality,
    balance_strata,
    quality_condition,
)


class FiveTaskTsneSamplingTest(unittest.TestCase):
    def test_phase_balance_is_exact_across_task_and_phase(self):
        rows = []
        for task in TASKS:
            for phase in PHASES:
                rows.extend({"task": task, "label": phase, "sample": index} for index in range(9))
        selected, _ = balance_strata(rows, ("task", "label"), target=7, seed=42)
        counts = {(task, phase): 0 for task in TASKS for phase in PHASES}
        for row in selected:
            counts[(row["task"], row["label"])] += 1
        self.assertEqual(set(counts.values()), {7})

    def test_quality_balance_uses_weakest_class_without_replacement(self):
        rows = []
        for index, label in enumerate(QUALITY_STATES):
            rows.extend({"label": label, "task": "board", "sample": sample} for sample in range(5 + index))
        selected, candidates = balance_quality(rows, target=20, seed=42, min_per_present_task=0)
        self.assertEqual(len(selected), 5 * len(QUALITY_STATES))
        self.assertEqual(candidates[QUALITY_STATES[0]], 5)
        self.assertEqual(len({(row["label"], row["sample"]) for row in selected}), len(selected))

    def test_condition_mapping_does_not_promote_mixed_socket(self):
        self.assertIsNone(quality_condition(SimpleNamespace(task="socket", condition="mixed_success_bounce")))
        self.assertEqual(
            quality_condition(SimpleNamespace(task="socket", condition="2pin_bounce")), QUALITY_STATES[5]
        )
        self.assertEqual(
            quality_condition(SimpleNamespace(task="board", condition="too_low")), QUALITY_STATES[3]
        )


if __name__ == "__main__":
    unittest.main()
