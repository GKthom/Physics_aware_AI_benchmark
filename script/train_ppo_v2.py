"""Train sequential PPO with a complete structured level observation.

The policy chooses whether to draw and then observes the updated world after a
fixed number of physics frames. The observation exposes padded slots for every
object, fixed obstacle, previously drawn stroke, and goal.

Example:
    python script/train_ppo_v2.py --level 10 --total-timesteps 300000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Stable-Baselines3 is PyTorch-based. Force TensorBoard's lightweight
# compatibility layer so an unrelated TensorFlow installation cannot be loaded
# (and, with some binary combinations, crash Python) during the SB3 import.
sys.modules.setdefault(
    "tensorboard.compat.notf",
    types.ModuleType("tensorboard.compat.notf"),
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


OBJECT_TYPES = ("circle", "box", "polygon", "segment")
GOAL_TYPES = tuple(BrainItOnGeneralEnv.GOAL_TYPES)


def _clip(value: float, scale: float = 1.0) -> float:
    """Scale a value and keep every observation feature in [-1, 1]."""
    return float(np.clip(float(value) / max(float(scale), 1e-8), -1.0, 1.0))


def _level_capacities(level_path: Path) -> Tuple[int, int, int, int, int]:
    """Return fixed slot counts large enough for every level in the file."""
    with level_path.open("r", encoding="utf-8") as file:
        levels = json.load(file).get("levels", [])

    if not levels:
        raise ValueError(f"No levels were found in {level_path}")

    return (
        max(len(level.get("objects", [])) for level in levels),
        max(len(level.get("static_segments", [])) for level in levels),
        max(len(level.get("goals", [])) for level in levels),
        max(int(level.get("draw_rules", {}).get("max_segments", 5)) for level in levels),
        max(len(level.get("joints", [])) for level in levels),
    )


def _level_config(level_path: Path, level_id: int) -> Dict[str, Any]:
    with level_path.open("r", encoding="utf-8") as file:
        levels = json.load(file).get("levels", [])

    for level in levels:
        if int(level.get("level_id")) == int(level_id):
            return level
    available = [level.get("level_id") for level in levels]
    raise ValueError(f"Level {level_id} was not found. Available levels: {available}")


class FullLevelObservation(gym.ObservationWrapper):
    """Expose complete structured level and physics state to an MLP policy.

    Unused object/segment/goal slots are zero padded.  The first value in each
    slot is a presence mask, so a real all-zero property vector can still be
    distinguished from padding.
    """

    GLOBAL_FEATURES = 16
    OBJECT_FEATURES = 24
    SEGMENT_FEATURES = 13
    GOAL_FEATURES = 25
    JOINT_FEATURES = 12
    AGENT_SEGMENT_FIXED_FEATURES = 12

    def __init__(
        self,
        env: gym.Env,
        *,
        max_objects: int,
        max_static_segments: int,
        max_goals: int,
        max_agent_segments: int,
        max_stroke_points: int,
        max_joints: int,
    ):
        super().__init__(env)
        self.max_objects = int(max_objects)
        self.max_static_segments = int(max_static_segments)
        self.max_goals = int(max_goals)
        self.max_agent_segments = int(max_agent_segments)
        self.max_stroke_points = max(2, int(max_stroke_points))
        self.max_joints = int(max_joints)
        self.agent_segment_features = (
            self.AGENT_SEGMENT_FIXED_FEATURES + 2 * self.max_stroke_points
        )
        self._previous_velocities = {}

        self.obs_size = (
            self.GLOBAL_FEATURES
            + self.max_objects * self.OBJECT_FEATURES
            + self.max_static_segments * self.SEGMENT_FEATURES
            + self.max_joints * self.JOINT_FEATURES
            + self.max_agent_segments * self.agent_segment_features
            + self.max_goals * self.GOAL_FEATURES
        )
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_size, dtype=np.float32),
            high=np.ones(self.obs_size, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def base_env(self) -> BrainItOnGeneralEnv:
        return self.env.unwrapped

    def observation(self, _original_observation: np.ndarray) -> np.ndarray:
        features = []
        features.extend(self._global_features())
        features.extend(
            self._padded_slots(
                self._object_features(),
                self.max_objects,
                self.OBJECT_FEATURES,
            )
        )
        features.extend(
            self._padded_slots(
                self._segment_features(),
                self.max_static_segments,
                self.SEGMENT_FEATURES,
            )
        )
        features.extend(
            self._padded_slots(
                self._joint_features(),
                self.max_joints,
                self.JOINT_FEATURES,
            )
        )
        features.extend(
            self._padded_slots(
                self._agent_segment_features(),
                self.max_agent_segments,
                self.agent_segment_features,
            )
        )
        features.extend(
            self._padded_slots(
                self._goal_features(),
                self.max_goals,
                self.GOAL_FEATURES,
            )
        )

        observation = np.asarray(features, dtype=np.float32)
        if observation.shape != (self.obs_size,):
            raise RuntimeError(
                f"Observation has shape {observation.shape}; expected {(self.obs_size,)}"
            )
        return np.clip(observation, -1.0, 1.0)

    def reset(self, **kwargs):
        self._previous_velocities = {}
        return super().reset(**kwargs)

    @staticmethod
    def _padded_slots(
        slots: Iterable[Iterable[float]],
        capacity: int,
        slot_size: int,
    ) -> list[float]:
        result = []
        slots = list(slots)
        if len(slots) > capacity:
            raise ValueError(f"Level needs {len(slots)} slots, but capacity is {capacity}")

        for slot in slots:
            slot = list(slot)
            if len(slot) != slot_size:
                raise RuntimeError(f"Slot has {len(slot)} features; expected {slot_size}")
            result.extend(slot)

        result.extend([0.0] * ((capacity - len(slots)) * slot_size))
        return result

    def _global_features(self) -> list[float]:
        env = self.base_env
        physics = env.level_data.get("physics", {})
        draw = env.draw_region
        stroke_is_static = env.stroke_body_type == "static"

        return [
            _clip(physics.get("gravity", 500.0), 1000.0),
            _clip(physics.get("damping", 0.98)),
            _clip(physics.get("iterations", 20), 100.0),
            _clip(physics.get("substeps", 4), 10.0),
            _clip(draw["x_min"], env.canvas_width),
            _clip(draw["x_max"], env.canvas_width),
            _clip(draw["y_min"], env.canvas_height),
            _clip(draw["y_max"], env.canvas_height),
            _clip(env.max_drawn_segments, 10.0),
            _clip(env.agent_segment_thickness, 50.0),
            float(stroke_is_static),
            float(not stroke_is_static),
            _clip(env.step_count, env.max_steps),
            float(env.simulation_started),
            _clip(env.collisions_since_last_decision, 20.0),
            _clip(env.collision_count, 100.0),
        ]

    def _object_features(self) -> Iterable[list[float]]:
        env = self.base_env
        for config in env.level_data.get("objects", []):
            obj = env.objects[config["id"]]
            body = obj["body"]
            dynamic = bool(config.get("dynamic", True))

            if dynamic:
                x, y = float(body.position.x), float(body.position.y)
                vx, vy = float(body.velocity.x), float(body.velocity.y)
                angle = float(body.angle)
                angular_velocity = float(body.angular_velocity)
            else:
                x = float(config.get("x", 0.0))
                y = float(config.get("y", 0.0))
                vx = vy = angular_velocity = 0.0
                angle = math.radians(float(config.get("angle_degrees", 0.0)))

            width, height, radius = self._object_dimensions(config)
            speed = math.hypot(vx, vy)
            object_type = config.get("type", "")
            previous_vx, previous_vy = self._previous_velocities.get(
                config["id"], (vx, vy)
            )
            self._previous_velocities[config["id"]] = (vx, vy)

            yield [
                1.0,
                *(float(object_type == known) for known in OBJECT_TYPES),
                float(dynamic),
                _clip(x, env.canvas_width),
                _clip(y, env.canvas_height),
                _clip(vx, env.MAX_SPEED),
                _clip(vy, env.MAX_SPEED),
                math.sin(angle),
                math.cos(angle),
                _clip(angular_velocity, 20.0),
                _clip(width, env.canvas_width),
                _clip(height, env.canvas_height),
                _clip(radius, min(env.canvas_width, env.canvas_height)),
                _clip(config.get("mass", 1.0), 20.0),
                _clip(config.get("friction", 0.8), 2.0),
                _clip(config.get("elasticity", 0.2)),
                _clip(speed, env.MAX_SPEED),
                _clip(vx - previous_vx, env.MAX_SPEED),
                _clip(vy - previous_vy, env.MAX_SPEED),
                _clip(
                    env.object_collisions_since_last_decision.get(config["id"], 0),
                    10.0,
                ),
                _clip(env.object_collision_counts.get(config["id"], 0), 100.0),
            ]

    @staticmethod
    def _object_dimensions(config: Dict[str, Any]) -> Tuple[float, float, float]:
        radius = float(config.get("radius", 0.0))
        width = float(config.get("width", radius * 2.0))
        height = float(config.get("height", radius * 2.0))

        vertices = config.get("vertices", [])
        if vertices:
            xs = [float(vertex["x"]) for vertex in vertices]
            ys = [float(vertex["y"]) for vertex in vertices]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

        return width, height, radius

    def _segment_features(self) -> Iterable[list[float]]:
        env = self.base_env
        diagonal = math.hypot(env.canvas_width, env.canvas_height)

        for segment in env.level_data.get("static_segments", []):
            ax = float(segment.get("a", {}).get("x", 0.0))
            ay = float(segment.get("a", {}).get("y", 0.0))
            bx = float(segment.get("b", {}).get("x", 0.0))
            by = float(segment.get("b", {}).get("y", 0.0))
            dx, dy = bx - ax, by - ay

            yield [
                1.0,
                _clip(ax, env.canvas_width),
                _clip(ay, env.canvas_height),
                _clip(bx, env.canvas_width),
                _clip(by, env.canvas_height),
                _clip((ax + bx) / 2.0, env.canvas_width),
                _clip((ay + by) / 2.0, env.canvas_height),
                _clip(dx, env.canvas_width),
                _clip(dy, env.canvas_height),
                _clip(math.hypot(dx, dy), diagonal),
                _clip(segment.get("thickness", 8.0), 50.0),
                _clip(segment.get("friction", 0.8), 2.0),
                _clip(segment.get("elasticity", 0.2)),
            ]

    def _agent_segment_features(self) -> Iterable[list[float]]:
        """Describe strokes already drawn, including their current geometry."""
        env = self.base_env
        diagonal = math.hypot(env.canvas_width, env.canvas_height)

        for segment in env.agent_segments:
            points = self._agent_segment_world_points(segment)
            if len(points) > self.max_stroke_points:
                indices = np.linspace(
                    0, len(points) - 1, self.max_stroke_points, dtype=int
                )
                points = [points[index] for index in indices]

            body_type = segment.get("body_type", "static")
            dynamic = body_type == "dynamic"
            body = None
            if dynamic:
                bodies = segment.get("bodies", [])
                body = bodies[0] if bodies else segment.get("body")

            vx = float(body.velocity.x) if body is not None else 0.0
            vy = float(body.velocity.y) if body is not None else 0.0
            angle = float(body.angle) if body is not None else 0.0
            length = sum(
                math.hypot(bx - ax, by - ay)
                for (ax, ay), (bx, by) in zip(points, points[1:])
            )

            point_features = []
            for x, y in points:
                point_features.extend(
                    [_clip(x, env.canvas_width), _clip(y, env.canvas_height)]
                )
            point_features.extend(
                [0.0] * (2 * (self.max_stroke_points - len(points)))
            )

            yield [
                1.0,
                float(segment.get("type") == "stroke"),
                float(segment.get("type") == "line"),
                float(not dynamic),
                float(dynamic),
                _clip(segment.get("thickness", env.agent_segment_thickness), 50.0),
                _clip(length, diagonal),
                _clip(vx, env.MAX_SPEED),
                _clip(vy, env.MAX_SPEED),
                math.sin(angle),
                math.cos(angle),
                _clip(len(points), self.max_stroke_points),
                *point_features,
            ]

    def _joint_features(self) -> Iterable[list[float]]:
        env = self.base_env
        diagonal = math.hypot(env.canvas_width, env.canvas_height)
        known_types = (
            "pin_joint_to_world",
            "pivot_joint_to_world",
            "pin_joint",
            "pivot_joint",
        )

        for joint_data in env.joints:
            config = joint_data["config"]
            constraint = joint_data["constraint"]
            anchor_a, anchor_b = env._joint_world_anchors(joint_data)
            distance = math.hypot(anchor_b.x - anchor_a.x, anchor_b.y - anchor_a.y)
            max_force = float(constraint.max_force)
            max_force_feature = 1.0 if not math.isfinite(max_force) else _clip(
                max_force, 100_000.0
            )

            yield [
                1.0,
                *(float(config.get("type") == known) for known in known_types),
                _clip(anchor_a.x, env.canvas_width),
                _clip(anchor_a.y, env.canvas_height),
                _clip(anchor_b.x, env.canvas_width),
                _clip(anchor_b.y, env.canvas_height),
                _clip(distance, diagonal),
                _clip(constraint.impulse, 10_000.0),
                max_force_feature,
            ]

    @staticmethod
    def _agent_segment_world_points(segment) -> list[Tuple[float, float]]:
        if segment.get("body_type") == "dynamic":
            shapes = segment.get("shapes", [])
            if not shapes and segment.get("shape") is not None:
                shapes = [segment["shape"]]
            if not shapes:
                return []

            first_shape = shapes[0]
            points = [first_shape.body.local_to_world(first_shape.a)]
            points.extend(shape.body.local_to_world(shape.b) for shape in shapes)
            return [(float(point.x), float(point.y)) for point in points]

        if segment.get("type") == "stroke":
            return [
                (float(point["x"]), float(point["y"]))
                for point in segment.get("points", [])
            ]

        a = segment.get("a", {})
        b = segment.get("b", {})
        return [
            (float(a.get("x", 0.0)), float(a.get("y", 0.0))),
            (float(b.get("x", 0.0)), float(b.get("y", 0.0))),
        ]

    def _goal_features(self) -> Iterable[list[float]]:
        env = self.base_env
        goals = env.level_data.get("goals", [])
        object_ids = [config["id"] for config in env.level_data.get("objects", [])]
        any_logic = env.level_data.get("goal_logic", "all") == "any"

        for goal in goals:
            region = goal.get("region", {})
            region_shape = region.get("shape", "")
            try:
                object_index = object_ids.index(goal.get("object_id"))
                object_slot = (object_index + 1) / max(1, self.max_objects)
            except ValueError:
                object_slot = 0.0

            duration = int(goal.get("duration_steps", 0))
            hold_progress = 0.0
            if duration > 0:
                hold_progress = env.goal_evaluator.goal_memory.get(
                    goal.get("id"), 0
                ) / duration

            yield [
                1.0,
                *(float(goal.get("type") == known) for known in GOAL_TYPES),
                _clip(object_slot),
                float(region_shape == "circle"),
                float(region_shape == "rectangle"),
                _clip(region.get("x", 0.0), env.canvas_width),
                _clip(region.get("y", 0.0), env.canvas_height),
                _clip(region.get("width", 0.0), env.canvas_width),
                _clip(region.get("height", 0.0), env.canvas_height),
                _clip(region.get("r", 0.0), min(env.canvas_width, env.canvas_height)),
                _clip(goal.get("y", 0.0), env.canvas_height),
                _clip(goal.get("min_y", 0.0), env.canvas_height),
                _clip(goal.get("max_speed", 0.0), env.MAX_SPEED),
                _clip(goal.get("duration_steps", 0), env.max_steps),
                _clip(goal.get("min_angle", 0.0), math.pi),
                _clip(goal.get("max_angle", 0.0), math.pi),
                float(any_logic),
                _clip(env._single_goal_score(goal)),
                _clip(hold_progress),
            ]


def make_env(
    *,
    level_path: Path,
    level_id: int,
    reward_mode: str,
    stroke_body: str,
    max_steps: int,
    agent_draw_mode: str,
    num_stroke_points: int,
    decision_frames: int,
    stroke_penalty: float,
    capacities: Tuple[int, int, int, int, int],
):
    def _init():
        env = BrainItOnGeneralEnv(
            level_path=str(level_path),
            level_id=level_id,
            train_level_ids=[level_id],
            render_mode=None,
            reward_mode=reward_mode,
            control_mode="agent",
            max_steps=max_steps,
            stroke_body=stroke_body,
            agent_draw_mode=agent_draw_mode,
            num_stroke_points=num_stroke_points,
            agent_decision_frames=decision_frames,
            sequential_stroke_penalty=stroke_penalty,
        )
        env = FullLevelObservation(
            env,
            max_objects=capacities[0],
            max_static_segments=capacities[1],
            max_goals=capacities[2],
            max_agent_segments=capacities[3],
            max_stroke_points=num_stroke_points,
            max_joints=capacities[4],
        )
        return Monitor(env, info_keywords=("is_success", "is_failure", "level_id"))

    return _init


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PPO on one level with complete structured observations."
    )
    parser.add_argument("--level-path", type=Path, default=ROOT / "levels" / "level.json")
    parser.add_argument("--level", type=int, required=True, help="Exact level ID to train on")
    parser.add_argument("--reward-mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--stroke-body", choices=("static", "dynamic"), default="static")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--agent-draw-mode", choices=("stroke", "line"), default="stroke")
    parser.add_argument("--num-stroke-points", type=int, default=2)
    parser.add_argument(
        "--decision-frames",
        type=int,
        default=5,
        help="Physics frames advanced between PPO decisions",
    )
    parser.add_argument(
        "--stroke-penalty",
        type=float,
        default=0.25,
        help="Penalty for every successfully added stroke",
    )
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--model-name", default=None)
    return parser


def train(args: argparse.Namespace) -> None:
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.decision_frames < 1:
        raise ValueError("--decision-frames must be at least 1")

    capacities = _level_capacities(args.level_path)
    level_config = _level_config(args.level_path, args.level)
    effective_stroke_body = level_config.get("draw_rules", {}).get(
        "stroke_body", args.stroke_body
    )
    model_name = args.model_name or (
        f"ppo_v2_level{args.level}_{args.reward_mode}_{effective_stroke_body}"
        f"_{args.agent_draw_mode}_seq{args.decision_frames}"
        f"_{args.num_stroke_points}pt"
    )
    model_dir = ROOT / "models" / model_name
    log_dir = ROOT / "logs" / model_name
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_factory_args = dict(
        level_path=args.level_path,
        level_id=args.level,
        reward_mode=args.reward_mode,
        stroke_body=args.stroke_body,
        max_steps=args.max_steps,
        agent_draw_mode=args.agent_draw_mode,
        num_stroke_points=args.num_stroke_points,
        decision_frames=args.decision_frames,
        stroke_penalty=args.stroke_penalty,
        capacities=capacities,
    )

    if not args.skip_env_check:
        checked_env = make_env(**env_factory_args)()
        check_env(checked_env, warn=True)
        checked_env.close()

    train_env = DummyVecEnv(
        [make_env(**env_factory_args) for _ in range(args.n_envs)]
    )
    eval_env = DummyVecEnv([make_env(**env_factory_args)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device="auto",
    )

    schema = {
        "level_id": args.level,
        "observation_size": train_env.observation_space.shape[0],
        "action_size": train_env.action_space.shape[0],
        "action_layout": {
            "index_0": "draw gate: value > 0 draws; value <= 0 waits",
            "remaining_values": "normalized stroke coordinates in x/y order",
        },
        "max_objects": capacities[0],
        "max_static_segments": capacities[1],
        "max_goals": capacities[2],
        "max_agent_segments": capacities[3],
        "max_joints": capacities[4],
        "effective_stroke_body": effective_stroke_body,
        "decision_frames": args.decision_frames,
        "stroke_penalty": args.stroke_penalty,
        "global_features": FullLevelObservation.GLOBAL_FEATURES,
        "features_per_object": FullLevelObservation.OBJECT_FEATURES,
        "features_per_static_segment": FullLevelObservation.SEGMENT_FEATURES,
        "features_per_agent_segment": (
            FullLevelObservation.AGENT_SEGMENT_FIXED_FEATURES
            + 2 * max(2, args.num_stroke_points)
        ),
        "features_per_joint": FullLevelObservation.JOINT_FEATURES,
        "features_per_goal": FullLevelObservation.GOAL_FEATURES,
        "note": "Padded slots include a presence mask as their first feature.",
    }
    with (model_dir / "observation_schema.json").open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)

    print(f"Training only on level: {args.level}")
    if effective_stroke_body != args.stroke_body:
        print(
            f"Level JSON overrides --stroke-body: using {effective_stroke_body!r} "
            f"instead of {args.stroke_body!r}"
        )
    print(f"Observation size: {schema['observation_size']}")
    print(
        "Slots: "
        f"objects={capacities[0]}, obstacles={capacities[1]}, "
        f"drawn_strokes={capacities[3]}, goals={capacities[2]}, "
        f"joints={capacities[4]}"
    )
    print(f"PPO decision interval: every {args.decision_frames} physics frames")
    print(f"Models: {model_dir}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
            tb_log_name="tb",
        )
        model.save(str(model_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    print(f"Best model: {model_dir / 'best_model.zip'}")
    print(f"Final model: {model_dir / 'final_model.zip'}")


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
