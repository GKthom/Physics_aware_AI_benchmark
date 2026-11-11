"""Validate the 25-level benchmark and its executable reference solutions.

This script does not train a policy. It checks that every level uses the current
JSON/environment contract, cannot solve itself when the agent only waits, and
can be solved by at least one small sequence of legal two-point strokes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


EXPECTED_GROUPS = {
    "Trajectory": range(1, 6),
    "Target": range(6, 11),
    "Stability": range(11, 16),
    "Support": range(16, 21),
    "Release": range(21, 26),
}

# These are regression witnesses, not claimed optimal solutions. Coordinates
# are canvas pixels and each tuple is one two-point stroke (x1, y1, x2, y2).
REFERENCE_STROKES = {
    1: [(80, 260, 780, 510)],
    2: [(820, 260, 120, 510)],
    3: [(100, 210, 500, 300), (500, 300, 820, 510)],
    4: [(80, 300, 540, 200)],
    5: [(100, 230, 400, 500), (450, 230, 790, 500)],
    6: [(90, 230, 760, 475)],
    7: [(90, 250, 700, 440), (770, 420, 770, 570)],
    8: [(60, 170, 470, 345)],
    9: [(800, 200, 500, 280)],
    10: [(80, 230, 310, 470), (820, 230, 590, 470)],
    11: [(340, 490, 560, 490)],
    12: [(330, 500, 570, 500), (560, 360, 560, 510)],
    13: [(510, 280, 510, 465), (390, 280, 390, 465), (390, 465, 510, 465)],
    14: [(820, 430, 820, 555)],
    15: [(190, 490, 410, 490), (490, 490, 710, 490)],
    16: [(280, 540, 620, 540)],
    17: [(330, 470, 570, 470)],
    18: [(285, 500, 615, 500)],
    19: [(350, 245, 350, 390)],
    20: [(220, 545, 440, 545), (490, 545, 710, 545)],
    21: [(320, 100, 440, 210)],
    22: [(440, 100, 360, 220)],
    23: [(225, 100, 380, 180)],
    24: [(245, 120, 275, 120)],
    25: [(340.2, 187.1, 325.4, 89.8)],
}


def validate_json_contract(level_path: Path) -> list[dict]:
    data = json.loads(level_path.read_text(encoding="utf-8"))
    if list(data) != ["levels"]:
        raise AssertionError("Top-level JSON structure must be exactly {'levels': [...]}.")

    levels = data["levels"]
    if len(levels) != 25:
        raise AssertionError(f"Expected 25 levels, found {len(levels)}.")
    if [level.get("level_id") for level in levels] != list(range(1, 26)):
        raise AssertionError("Level IDs must be unique and sequential from 1 to 25.")

    allowed_level_keys = {
        "level_id",
        "title",
        "canvas",
        "physics",
        "objects",
        "static_segments",
        "joints",
        "goal_logic",
        "goals",
        "draw_rules",
    }
    allowed_objects = {"circle", "box", "polygon", "segment"}
    allowed_joints = {
        "pin_joint_to_world",
        "pivot_joint_to_world",
        "pin_joint",
        "pivot_joint",
    }
    allowed_goals = set(BrainItOnGeneralEnv.GOAL_TYPES)

    counts = Counter()
    for level in levels:
        level_id = int(level["level_id"])
        unexpected = set(level) - allowed_level_keys
        if unexpected:
            raise AssertionError(f"Level {level_id} has unknown keys: {unexpected}")

        category = level["title"].split()[0]
        counts[category] += 1
        expected_ids = EXPECTED_GROUPS.get(category)
        if expected_ids is None or level_id not in expected_ids:
            raise AssertionError(
                f"Level {level_id} title/category does not match its ID group."
            )

        object_ids = [obj["id"] for obj in level["objects"]]
        if len(object_ids) != len(set(object_ids)):
            raise AssertionError(f"Level {level_id} has duplicate object IDs.")
        if any(obj["type"] not in allowed_objects for obj in level["objects"]):
            raise AssertionError(f"Level {level_id} has an unsupported object type.")
        if any(
            goal["type"] not in allowed_goals or goal["object_id"] not in object_ids
            for goal in level["goals"]
        ):
            raise AssertionError(f"Level {level_id} has an invalid goal.")
        if any(
            joint["type"] not in allowed_joints
            for joint in level.get("joints", [])
        ):
            raise AssertionError(f"Level {level_id} has an unsupported joint.")

    expected_counts = {name: 5 for name in EXPECTED_GROUPS}
    if dict(counts) != expected_counts:
        raise AssertionError(f"Expected five levels per category, found {dict(counts)}.")
    return levels


def make_env(level_path: Path, level_id: int, max_steps: int, render: bool):
    return BrainItOnGeneralEnv(
        level_path=str(level_path),
        level_id=level_id,
        control_mode="agent",
        agent_decision_frames=5,
        num_stroke_points=2,
        max_steps=max_steps,
        render_mode="rgb_array" if render else None,
    )


def run_episode(env: BrainItOnGeneralEnv, strokes: Iterable[tuple]) -> tuple[bool, int]:
    env.reset(seed=1)
    terminated = truncated = False

    for stroke in strokes:
        action = np.asarray([1.0, *stroke], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        if not math.isfinite(float(reward)):
            raise AssertionError(f"Level {env.level_id} produced a non-finite reward.")
        if terminated or truncated:
            break

    wait_action = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(wait_action)
        if not np.all(np.isfinite(observation)) or not math.isfinite(float(reward)):
            raise AssertionError(f"Level {env.level_id} produced non-finite state.")

    return bool(info.get("is_success", False)), int(info["step_count"])


def validate_physics(level_path: Path, max_steps: int, render: bool) -> None:
    print("ID  GROUP          STROKES  SOLVE_FRAME  NO_DRAW")
    print("--  -------------  -------  -----------  -------")
    for level_id in range(1, 26):
        env = make_env(level_path, level_id, max_steps, render)
        env.reset(seed=1)
        if render:
            frame = env.render()
            if frame.shape != (750, 900, 3):
                raise AssertionError(
                    f"Level {level_id} rendered {frame.shape}, expected (750, 900, 3)."
                )
        solved, solve_frame = run_episode(env, REFERENCE_STROKES[level_id])
        env.close()
        if not solved:
            raise AssertionError(f"Reference strokes did not solve level {level_id}.")

        env = make_env(level_path, level_id, max_steps, False)
        auto_solved, _ = run_episode(env, [])
        group = env.level_data["title"].split()[0]
        env.close()
        if auto_solved:
            raise AssertionError(f"Level {level_id} solved itself without a stroke.")

        print(
            f"{level_id:>2}  {group:<13}  "
            f"{len(REFERENCE_STROKES[level_id]):>7}  {solve_frame:>11}  pass"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level-path",
        type=Path,
        default=ROOT / "levels" / "level.json",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render-check", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.max_steps < 500:
        raise ValueError("Reference validation requires --max-steps of at least 500.")
    validate_json_contract(args.level_path)
    validate_physics(args.level_path, args.max_steps, args.render_check)
    print("\nPASS: 25 schema-valid, non-trivial, reference-solvable levels.")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
