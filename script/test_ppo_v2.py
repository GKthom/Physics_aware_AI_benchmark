"""Evaluate PPO models trained by ``train_ppo_v2.py``.

The V2 policy uses a padded structured observation and a sequential action with
a draw/wait gate.  This evaluator rebuilds that exact interface from the
``observation_schema.json`` stored beside each trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Importing the wrapper from the trainer also applies its TensorBoard import
# compatibility setup before Stable-Baselines3 is loaded.
from script.train_ppo_v2 import FullLevelObservation
from stable_baselines3 import PPO

from env.brainiton_env_v2 import BrainItOnGeneralEnv


REQUIRED_SCHEMA_KEYS = (
    "level_id",
    "observation_size",
    "action_size",
    "max_objects",
    "max_static_segments",
    "max_goals",
    "max_agent_segments",
    "max_joints",
    "decision_frames",
)


def resolve_model_path(path: Path, checkpoint: str) -> Path:
    """Resolve a model directory, basename, or explicit zip file."""
    path = path.expanduser()

    if path.is_dir():
        candidate = path / f"{checkpoint}_model.zip"
        if not candidate.is_file():
            available = sorted(item.name for item in path.glob("*.zip"))
            raise FileNotFoundError(
                f"{candidate} was not found. Available model files: {available}"
            )
        return candidate

    if path.is_file():
        return path

    if path.suffix != ".zip":
        zip_path = path.with_suffix(".zip")
        if zip_path.is_file():
            return zip_path

    raise FileNotFoundError(f"PPO model was not found: {path}")


def load_schema(model_path: Path, requested_path: Path | None) -> Dict[str, Any]:
    schema_path = (
        requested_path.expanduser()
        if requested_path is not None
        else model_path.parent / "observation_schema.json"
    )
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Observation schema was not found: {schema_path}. "
            "V2 evaluation requires the schema saved during training."
        )

    with schema_path.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    missing = [key for key in REQUIRED_SCHEMA_KEYS if key not in schema]
    if missing:
        raise ValueError(f"Observation schema is missing required keys: {missing}")
    return schema


def parse_level_ids(value: str, trained_level: int) -> list[int]:
    if not value.strip():
        return [trained_level]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_num_stroke_points(schema: Dict[str, Any]) -> int:
    action_size = int(schema["action_size"])
    coordinate_values = action_size - 1
    if coordinate_values < 4 or coordinate_values % 2 != 0:
        raise ValueError(
            f"Unsupported V2 action size {action_size}; expected one gate followed "
            "by at least two x/y points."
        )
    return coordinate_values // 2


def load_levels(level_path: Path) -> Dict[int, Dict[str, Any]]:
    with level_path.open("r", encoding="utf-8") as file:
        levels = json.load(file).get("levels", [])
    return {int(level["level_id"]): level for level in levels}


def validate_level_capacities(
    level_path: Path,
    level_ids: Iterable[int],
    schema: Dict[str, Any],
) -> None:
    levels = load_levels(level_path)
    capacities = {
        "objects": int(schema["max_objects"]),
        "static_segments": int(schema["max_static_segments"]),
        "goals": int(schema["max_goals"]),
        "agent_segments": int(schema["max_agent_segments"]),
        "joints": int(schema["max_joints"]),
    }

    for level_id in level_ids:
        if level_id not in levels:
            raise ValueError(
                f"Level {level_id} was not found. Available levels: {sorted(levels)}"
            )
        level = levels[level_id]
        requirements = {
            "objects": len(level.get("objects", [])),
            "static_segments": len(level.get("static_segments", [])),
            "goals": len(level.get("goals", [])),
            "agent_segments": int(
                level.get("draw_rules", {}).get("max_segments", 5)
            ),
            "joints": len(level.get("joints", [])),
        }
        exceeded = {
            name: (required, capacities[name])
            for name, required in requirements.items()
            if required > capacities[name]
        }
        if exceeded:
            details = ", ".join(
                f"{name} requires {required}, model supports {capacity}"
                for name, (required, capacity) in exceeded.items()
            )
            raise ValueError(
                f"Level {level_id} does not fit this model's saved observation "
                f"schema: {details}. Retrain with larger capacities."
            )


def build_env(
    args: argparse.Namespace,
    schema: Dict[str, Any],
    level_ids: list[int],
    num_stroke_points: int,
) -> FullLevelObservation:
    stroke_body = args.stroke_body or str(
        schema.get("effective_stroke_body", "static")
    )
    env = BrainItOnGeneralEnv(
        level_path=str(args.level_path),
        level_id=level_ids[0],
        render_mode="human" if args.render else None,
        reward_mode=args.reward_mode,
        control_mode="agent",
        max_steps=args.max_steps,
        stroke_body=stroke_body,
        agent_draw_mode=args.agent_draw_mode,
        num_stroke_points=num_stroke_points,
        agent_decision_frames=int(schema["decision_frames"]),
        sequential_stroke_penalty=float(schema.get("stroke_penalty", 0.25)),
    )
    return FullLevelObservation(
        env,
        max_objects=int(schema["max_objects"]),
        max_static_segments=int(schema["max_static_segments"]),
        max_goals=int(schema["max_goals"]),
        max_agent_segments=int(schema["max_agent_segments"]),
        max_stroke_points=num_stroke_points,
        max_joints=int(schema["max_joints"]),
    )


def validate_model_spaces(
    model: PPO,
    env: FullLevelObservation,
    schema: Dict[str, Any],
) -> None:
    expected_observation = int(schema["observation_size"])
    expected_action = int(schema["action_size"])
    actual_observation = int(env.observation_space.shape[0])
    actual_action = int(env.action_space.shape[0])

    if actual_observation != expected_observation:
        raise ValueError(
            f"Reconstructed observation size is {actual_observation}, but the saved "
            f"schema expects {expected_observation}."
        )
    if actual_action != expected_action:
        raise ValueError(
            f"Reconstructed action size is {actual_action}, but the saved schema "
            f"expects {expected_action}."
        )
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Model expects observation shape {model.observation_space.shape}, but "
            f"the reconstructed environment provides {env.observation_space.shape}."
        )
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(
            f"Model expects action shape {model.action_space.shape}, but the "
            f"reconstructed environment provides {env.action_space.shape}."
        )


def describe_action(action: np.ndarray, env: FullLevelObservation) -> str:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values[0] <= 0.0:
        return f"wait gate={values[0]:+.3f}"

    draw = env.base_env.draw_region
    coordinates = values[1:].copy()
    coordinates[0::2] = draw["x_min"] + (
        (coordinates[0::2] + 1.0) * 0.5
    ) * (draw["x_max"] - draw["x_min"])
    coordinates[1::2] = draw["y_min"] + (
        (coordinates[1::2] + 1.0) * 0.5
    ) * (draw["y_max"] - draw["y_min"])
    points = [
        (round(float(coordinates[index]), 1), round(float(coordinates[index + 1]), 1))
        for index in range(0, len(coordinates), 2)
    ]
    return f"draw gate={values[0]:+.3f} points={points}"


def evaluate(args: argparse.Namespace) -> None:
    if args.episodes_per_level < 1:
        raise ValueError("--episodes-per-level must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.sleep < 0 or args.pause_after_episode < 0:
        raise ValueError("Sleep durations cannot be negative")

    model_path = resolve_model_path(args.model_path, args.checkpoint)
    schema = load_schema(model_path, args.schema_path)
    trained_level = int(schema["level_id"])
    level_ids = parse_level_ids(args.levels, trained_level)
    num_stroke_points = infer_num_stroke_points(schema)

    validate_level_capacities(args.level_path, level_ids, schema)
    env = build_env(args, schema, level_ids, num_stroke_points)
    model = PPO.load(str(model_path), device=args.device)
    validate_model_spaces(model, env, schema)

    print(f"Model: {model_path}")
    print(f"Trained level: {trained_level}")
    print(f"Evaluation levels: {level_ids}")
    print(f"Observation size: {env.observation_space.shape[0]}")
    print(f"Action size: {env.action_space.shape[0]}")
    print(f"Decision interval: {schema['decision_frames']} physics frames")

    results = []
    try:
        for level_id in level_ids:
            for episode in range(1, args.episodes_per_level + 1):
                seed = args.seed + episode - 1
                observation, info = env.reset(
                    seed=seed,
                    options={"level_id": level_id},
                )
                terminated = truncated = False
                episode_reward = 0.0
                decisions = 0
                final_info = info

                while not (terminated or truncated):
                    action, _ = model.predict(
                        observation,
                        deterministic=not args.stochastic,
                    )
                    decisions += 1
                    if args.print_actions:
                        print(
                            f"  decision={decisions:>3} frame={env.base_env.step_count:>3} "
                            f"{describe_action(action, env)}"
                        )

                    observation, reward, terminated, truncated, final_info = env.step(
                        action
                    )
                    episode_reward += float(reward)

                    # step() already renders once in human render mode. This
                    # delay approximates real time for a five-frame decision.
                    if args.render and args.sleep > 0:
                        time.sleep(args.sleep)

                success = bool(final_info.get("is_success", False))
                result = {
                    "level_id": level_id,
                    "episode": episode,
                    "success": success,
                    "reward": episode_reward,
                    "physics_frames": int(final_info.get("step_count", 0)),
                    "decisions": decisions,
                    "segments": int(final_info.get("segments_used", 0)),
                    "distance": final_info.get("goal_distance"),
                    "failure_reason": final_info.get("failure_reason"),
                }
                results.append(result)

                distance = result["distance"]
                distance_text = "n/a" if distance is None else f"{float(distance):.2f}"
                print(
                    f"Level {level_id:>2} | Episode {episode:>2} | "
                    f"reward={episode_reward:>8.2f} | success={success} | "
                    f"frames={result['physics_frames']:>3} | "
                    f"decisions={decisions:>3} | segments={result['segments']} | "
                    f"distance={distance_text} | failure={result['failure_reason']}"
                )

                if args.render and args.pause_after_episode > 0:
                    time.sleep(args.pause_after_episode)
    finally:
        env.close()

    successes = sum(int(result["success"]) for result in results)
    print("\nPPO V2 evaluation summary")
    print("-------------------------")
    print(f"Episodes: {len(results)}")
    print(f"Successes: {successes}")
    print(f"Success rate: {100.0 * successes / max(1, len(results)):.2f}%")
    print(
        "Average reward: "
        f"{sum(result['reward'] for result in results) / max(1, len(results)):.2f}"
    )
    print(
        "Average physics frames: "
        f"{sum(result['physics_frames'] for result in results) / max(1, len(results)):.2f}"
    )
    print(
        "Average decisions: "
        f"{sum(result['decisions'] for result in results) / max(1, len(results)):.2f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a sequential structured-observation PPO V2 model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Model directory, model basename, or explicit .zip file",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "final"),
        default="best",
        help="Model selected when --model-path is a directory",
    )
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument(
        "--level-path",
        type=Path,
        default=ROOT / "levels" / "level.json",
    )
    parser.add_argument(
        "--levels",
        default="",
        help="Comma-separated level IDs; defaults to the level used for training",
    )
    parser.add_argument("--episodes-per-level", type=int, default=3)
    parser.add_argument("--reward-mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--stroke-body", choices=("static", "dynamic"), default=None)
    parser.add_argument("--agent-draw-mode", choices=("stroke", "line"), default="stroke")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.067,
        help="Extra delay after each rendered PPO decision",
    )
    parser.add_argument("--pause-after-episode", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
