"""Labels and default data roots for the board chunk energy scorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


BOARD_CLASS_NAMES: Tuple[str, ...] = (
    "expert",
    "pressure_too_small",
    "pressure_too_large",
    "pressure_unstable",
)
BOARD_CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(BOARD_CLASS_NAMES)}

DEFAULT_CLASS_DIRS: Dict[str, str] = {
    "expert": "/home/chenshuai/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "pressure_too_small": "/home/chenshuai/data/dataset/260609/z_too_high",
    "pressure_too_large": "/home/chenshuai/data/dataset/260610/z_too_low",
    "pressure_unstable": "/home/chenshuai/data/dataset/260610/z_too_oscillate",
}


@dataclass(frozen=True)
class ClassSpec:
    label: int
    name: str
    root: Path


def default_class_specs() -> Tuple[ClassSpec, ...]:
    return tuple(
        ClassSpec(label=BOARD_CLASS_TO_ID[name], name=name, root=Path(root))
        for name, root in DEFAULT_CLASS_DIRS.items()
    )
