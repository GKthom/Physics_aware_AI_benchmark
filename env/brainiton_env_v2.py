import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

from env.object_factory import create_object,create_segment
from env.goal_evaluator import GoalEvaluator


class BrainItOnGeneralEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    FPS = 60
    DT = 1.0 / FPS
    MAX_SPEED = 1200.0
    OBS_SIZE = 32
    GOAL_TYPES = [
        "object_reaches_region",
        "object_stays_in_region",
        "object_angle_range",
        "object_above_height",
        "object_below_height",
        "object_velocity_below",
        "object_stays_above_y",
    ]

    def __init__(
        self,
        level_path: str,
        render_mode: Optional[str] = None,
        reward_mode: str = "dense",
        control_mode: str = "agent",
        max_steps: int = 500,
        level_id: int = 1,
        train_level_ids: Optional[list] = None,
        agent_draw_mode: str = "stroke",  
        stroke_body: str = "static",
        simulation_started: bool = False,
        num_stroke_points: int = 3,
        agent_decision_frames: Optional[int] = None,
        sequential_stroke_penalty: float = 0.25,
    ):
        super().__init__()
        self.simulation_started = simulation_started
        self.default_stroke_body_type = stroke_body
        self.stroke_body_type = stroke_body

        self.level_path = Path(level_path)
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.control_mode = control_mode
        self.agent_draw_mode = agent_draw_mode
        self.agent_decision_frames = agent_decision_frames
        if self.agent_decision_frames is not None and self.agent_decision_frames < 1:
            raise ValueError("agent_decision_frames must be at least 1 or None.")
        self.sequential_stroke_penalty = float(sequential_stroke_penalty)
        self.max_steps = max_steps
        self.level_id = int(level_id)
        self.train_level_ids = (
            [int(level_id) for level_id in train_level_ids]
            if train_level_ids is not None
            else None
        )
        self.num_stroke_points = num_stroke_points
        
        self.all_levels = self._load_levels_file(self.level_path)
        self.level_data = self._get_level_by_id(self.level_id)

        canvas = self.level_data.get("canvas", {})
        self.canvas_width = int(canvas.get("width", 900))
        self.canvas_height = int(canvas.get("height", 620))
        self.ui_height = 130
        self.window_width = self.canvas_width
        self.window_height = self.canvas_height + self.ui_height

        self._apply_level_draw_rules()

        self.space = None
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.static_segments = []
        self.agent_segments = []
        self.joints = []

        self.goal_evaluator = None
        self.goal_reached = False
        self.step_count = 0
        self.prev_goal_distance = 0.0
        self.prev_goal_score = 0.0
        self.collision_count = 0
        self.collisions_since_last_decision = 0
        self.object_collision_counts = {}
        self.object_collisions_since_last_decision = {}
        self._shape_to_object_id = {}

        self.screen = None
        self.clock = None
        self.font = None
        self.current_stroke_start = None
        self.current_stroke_points = []
        self.preview_stroke_points = []

        self.last_action = None
        self.last_line_length = 0.0

        self.action_space = self._make_action_space()

        self.observation_space = spaces.Box(
            low=-np.ones(self.OBS_SIZE, dtype=np.float32),
            high=np.ones(self.OBS_SIZE, dtype=np.float32),
            dtype=np.float32,
        )

    def _load_level(self, level_path: Path) -> Dict[str, Any]:
        if not level_path.exists():
            raise FileNotFoundError(f"Level file not found: {level_path}")

        with open(level_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_levels_file(self, level_path: Path) -> Dict[str, Any]:
        if not level_path.exists():
            raise FileNotFoundError(f"Levels file not found: {level_path}")

        with open(level_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "levels" not in data:
            raise ValueError("levels.json must contain a top-level 'levels' list.")

        return data
    
    def _get_level_by_id(self, level_id: int) -> Dict[str, Any]:
        for level in self.all_levels["levels"]:
            if int(level["level_id"]) == int(level_id):
                return level

        available = [level["level_id"] for level in self.all_levels["levels"]]
        raise ValueError(f"Level {level_id} not found. Available levels: {available}")
    
    def get_available_level_ids(self):
        return sorted([int(level["level_id"]) for level in self.all_levels["levels"]])

    def _make_action_space(self):
        if self.agent_draw_mode == "stroke":
            action_dim = self.num_stroke_points * 2
        else:
            action_dim = 4

        # Sequential control adds a leading draw/no-draw gate. A positive gate
        # asks the environment to draw; a non-positive gate advances physics
        # without adding another stroke.
        if self.control_mode == "agent" and self.agent_decision_frames is not None:
            action_dim += 1

        return spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _action_to_draw_coordinates(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()

        # PPO works best with normalized continuous actions. Pixel coordinates
        # are still accepted so scripted/LLM callers can pass direct draw points.
        if np.all(action >= -1.05) and np.all(action <= 1.05):
            scaled = action.copy()
            x_min = self.draw_region["x_min"]
            x_max = self.draw_region["x_max"]
            y_min = self.draw_region["y_min"]
            y_max = self.draw_region["y_max"]

            scaled[0::2] = x_min + ((scaled[0::2] + 1.0) * 0.5) * (x_max - x_min)
            scaled[1::2] = y_min + ((scaled[1::2] + 1.0) * 0.5) * (y_max - y_min)
            return scaled

        return action

    def _apply_level_settings(self):
        canvas = self.level_data.get("canvas", {})
        self.canvas_width = int(canvas.get("width", 900))
        self.canvas_height = int(canvas.get("height", 620))
        self.window_width = self.canvas_width
        self.window_height = self.canvas_height + self.ui_height
        self._apply_level_draw_rules()

    def _apply_level_draw_rules(self):
        draw_rules = self.level_data.get("draw_rules", {})
        self.stroke_body_type = draw_rules.get("stroke_body", self.default_stroke_body_type)

        if self.stroke_body_type not in ("static", "dynamic"):
            raise ValueError(
                f"Invalid stroke_body '{self.stroke_body_type}'. "
                "Use 'static' or 'dynamic'."
            )

        self.max_drawn_segments = int(draw_rules.get("max_segments", 5))
        self.agent_segment_thickness = float(draw_rules.get("segment_thickness", 6.0))
        self.draw_region = draw_rules.get(
            "draw_region",
            {
                "x_min": 100.0,
                "x_max": 700.0,
                "y_min": 80.0,
                "y_max": 500.0,
            },
        )
        self.action_space = self._make_action_space()
    
    def load_level_by_id(self, level_id: int):
        self.level_id = int(level_id)
        self.level_data = self._get_level_by_id(self.level_id)

        self._apply_level_settings()

        self.reset()


    def next_level(self):
        level_ids = self.get_available_level_ids()
        current_index = level_ids.index(self.level_id)
        next_index = (current_index + 1) % len(level_ids)
        self.load_level_by_id(level_ids[next_index])


    def previous_level(self):
        level_ids = self.get_available_level_ids()
        current_index = level_ids.index(self.level_id)
        previous_index = (current_index - 1) % len(level_ids)
        self.load_level_by_id(level_ids[previous_index])

    def _setup_world(self):
        self.space = pymunk.Space()


        physics = self.level_data.get("physics", {})

        gravity = float(physics.get("gravity", 500.0))
        damping = float(physics.get("damping", 0.98))
        iterations = int(physics.get("iterations", 20))

        self.space.gravity = (0.0, gravity)
        self.space.damping = damping
        self.space.iterations = iterations

        self.objects = {}
        self.static_segments = []
        self.agent_segments = []
        self.joints = []
        self._shape_to_object_id = {}

        for obj_config in self.level_data.get("objects", []):
            obj = create_object(self.space, obj_config)
            self.objects[obj["id"]] = obj
            self._shape_to_object_id[obj["shape"]] = obj["id"]

        for seg_config in self.level_data.get("static_segments", []):
            fixed_seg_config = dict(seg_config)
            fixed_seg_config["type"] = "segment"
            fixed_seg_config["dynamic"] = False

            seg = create_segment(self.space, fixed_seg_config)
            self.static_segments.append(seg)

        for joint_config in self.level_data.get("joints", []):
            joint = self._create_joint(joint_config)
            self.space.add(joint)
            self.joints.append(
                {
                    "id": joint_config.get("id", "joint"),
                    "config": joint_config,
                    "constraint": joint,
                }
            )

        self.goal_evaluator = GoalEvaluator(self.level_data, self.objects)
        self.goal_evaluator.reset()
        self.space.on_collision(None, None, begin=self._on_collision_begin)

    def _create_joint(self, config):
        """Create a supported constraint from a level joint definition."""
        joint_type = config.get("type")

        if joint_type in ("pin_joint_to_world", "pivot_joint_to_world"):
            object_id = config["object_id"]
            if object_id not in self.objects:
                raise KeyError(f"Joint object '{object_id}' was not found.")
            body = self.objects[object_id]["body"]
            world_anchor = config.get("world_anchor", {})
            object_anchor = config.get("object_anchor", {})
            anchor_world = (
                float(world_anchor.get("x", 0.0)),
                float(world_anchor.get("y", 0.0)),
            )
            anchor_object = (
                float(object_anchor.get("x", 0.0)),
                float(object_anchor.get("y", 0.0)),
            )

            if joint_type == "pin_joint_to_world":
                constraint = pymunk.PinJoint(
                    self.space.static_body,
                    body,
                    anchor_world,
                    anchor_object,
                )
            else:
                constraint = pymunk.PivotJoint(
                    self.space.static_body,
                    body,
                    anchor_world,
                    anchor_object,
                )

        elif joint_type in ("pin_joint", "pivot_joint"):
            object_a_id = config["object_a_id"]
            object_b_id = config["object_b_id"]
            if object_a_id not in self.objects or object_b_id not in self.objects:
                raise KeyError(
                    f"Joint objects '{object_a_id}' and '{object_b_id}' must exist."
                )
            body_a = self.objects[object_a_id]["body"]
            body_b = self.objects[object_b_id]["body"]
            anchor_a_config = config.get("anchor_a", {})
            anchor_b_config = config.get("anchor_b", {})
            anchor_a = (
                float(anchor_a_config.get("x", 0.0)),
                float(anchor_a_config.get("y", 0.0)),
            )
            anchor_b = (
                float(anchor_b_config.get("x", 0.0)),
                float(anchor_b_config.get("y", 0.0)),
            )

            if joint_type == "pin_joint":
                constraint = pymunk.PinJoint(body_a, body_b, anchor_a, anchor_b)
            else:
                constraint = pymunk.PivotJoint(body_a, body_b, anchor_a, anchor_b)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")

        constraint.collide_bodies = bool(config.get("collide_bodies", False))
        if "max_force" in config:
            constraint.max_force = float(config["max_force"])
        if "error_bias" in config:
            constraint.error_bias = float(config["error_bias"])
        return constraint

    @staticmethod
    def _joint_world_anchors(joint_data):
        constraint = joint_data["constraint"]
        anchor_a = constraint.a.local_to_world(constraint.anchor_a)
        anchor_b = constraint.b.local_to_world(constraint.anchor_b)
        return anchor_a, anchor_b

    def _on_collision_begin(self, arbiter, _space, _data):
        """Record contacts occurring between two agent decisions."""
        self.collision_count += 1
        self.collisions_since_last_decision += 1

        for shape in arbiter.shapes:
            object_id = self._shape_to_object_id.get(shape)
            if object_id is None:
                continue
            self.object_collision_counts[object_id] = (
                self.object_collision_counts.get(object_id, 0) + 1
            )
            self.object_collisions_since_last_decision[object_id] = (
                self.object_collisions_since_last_decision.get(object_id, 0) + 1
            )

    def _reset_attempt_state(self):
        self.step_count = 0
        self.goal_reached = False
        self.last_action = None
        self.last_line_length = 0.0
        self.current_stroke_points = []
        self.preview_stroke_points = []
        self.simulation_started = False
        self.collision_count = 0
        self.collisions_since_last_decision = 0
        self.object_collision_counts = {object_id: 0 for object_id in self.objects}
        self.object_collisions_since_last_decision = {
            object_id: 0 for object_id in self.objects
        }

        if self.goal_evaluator is not None:
            self.goal_evaluator.reset()

        if self.space is not None:
            self.prev_goal_distance = self._distance_to_first_goal()
            self.prev_goal_score = self._goal_progress_score()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        requested_level_id = None
        if options is not None and "level_id" in options:
            requested_level_id = int(options["level_id"])
        elif self.train_level_ids:
            requested_level_id = int(self.np_random.choice(self.train_level_ids))

        if requested_level_id is not None and requested_level_id != self.level_id:
            self.level_id = requested_level_id
            self.level_data = self._get_level_by_id(self.level_id)

        self._apply_level_settings()

        self._setup_world()
        self._reset_attempt_state()

        obs = self._get_obs()
        info = {
            "level_id": self.level_id,
            "level_title": self.level_data.get("title", "Untitled Level"),
            "is_success": False,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        if self.space is None:
            raise RuntimeError("Physics space is not initialized. Call reset() before step().")

        substeps = int(self.level_data.get("physics", {}).get("substeps", 4))

        if self.control_mode == "agent":
            if self.agent_decision_frames is not None:
                return self._step_agent_sequential(action, substeps)

            if not self.simulation_started:
                segment_count_before = len(self.agent_segments)

                if self.agent_draw_mode == "stroke":
                    draw_success = self._apply_agent_stroke_action(action)
                else:
                    draw_success = self._apply_draw_action(action)
                    draw_success = draw_success and len(self.agent_segments) > segment_count_before

                if not draw_success:
                    obs = self._get_obs()
                    info = {
                        "is_success": False,
                        "is_failure": True,
                        "failure_reason": "invalid_draw",
                        "step_count": self.step_count,
                        "segments_used": len(self.agent_segments),
                        "goal_distance": self.prev_goal_distance,
                        "level_id": self.level_id,
                        "level_title": self.level_data.get("title", "Untitled Level"),
                    }
                    return obs, -100.0, False, True, info

            episode_reward = 0.0

            while self.step_count < self.max_steps and not self.goal_reached:
                self.step_count += 1

                if self._advance_physics_frame(substeps):
                    self.goal_reached = True

                current_goal_distance = self._distance_to_first_goal()
                current_goal_score = self._goal_progress_score()
                episode_reward += self._compute_reward(
                    self.prev_goal_score,
                    current_goal_score,
                )
                self.prev_goal_distance = current_goal_distance
                self.prev_goal_score = current_goal_score

                if self.render_mode == "human":
                    self.render()

            terminated = self.goal_reached
            truncated = (self.step_count >= self.max_steps) and not self.goal_reached

            if truncated:
                episode_reward -= 50.0

            obs = self._get_obs()
            info = {
                "is_success": self.goal_reached,
                "is_failure": truncated,
                "failure_reason": "timeout" if truncated else None,
                "step_count": self.step_count,
                "segments_used": len(self.agent_segments),
                "goal_distance": self.prev_goal_distance,
                "level_id": self.level_id,
                "level_title": self.level_data.get("title", "Untitled Level"),
            }

            return obs, float(episode_reward), terminated, truncated, info

        if self.control_mode == "human" and not self.simulation_started:
            # In human mode, the level waits after reset so the player can draw.
            # Once a stroke is drawn, _apply_stroke_points() starts simulation.
            obs = self._get_obs()
            info = {
                "is_success": self.goal_reached,
                "step_count": self.step_count,
                "segments_used": len(self.agent_segments),
                "goal_distance": self.prev_goal_distance,
                "level_id": self.level_id,
                "level_title": self.level_data.get("title", "Untitled Level"),
            }
            return obs, 0.0, False, False, info

        self.step_count += 1

        if self._advance_physics_frame(substeps):
            self.goal_reached = True

        current_goal_distance = self._distance_to_first_goal()
        current_goal_score = self._goal_progress_score()
        reward = self._compute_reward(self.prev_goal_score, current_goal_score)
        self.prev_goal_distance = current_goal_distance
        self.prev_goal_score = current_goal_score

        terminated = self.goal_reached
        truncated = (self.step_count >= self.max_steps) and not self.goal_reached

        if truncated:
            reward -= 50.0

        obs = self._get_obs()

        info = {
            "is_success": self.goal_reached,
            "is_failure": truncated,
            "failure_reason": "timeout" if truncated else None,
            "step_count": self.step_count,
            "segments_used": len(self.agent_segments),
            "goal_distance": current_goal_distance,
            "level_id": self.level_id,
            "level_title": self.level_data.get("title", "Untitled Level"),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _advance_physics_frame(self, substeps):
        """Advance exactly one 1/FPS frame in every control mode."""
        physics_substeps = max(1, int(substeps))
        substep_dt = self.DT / physics_substeps

        for substep_index in range(physics_substeps):
            self.space.step(substep_dt)
            update_memory = substep_index == physics_substeps - 1
            if self.goal_evaluator.check_success(update_memory=update_memory):
                return True
        return False

    def _step_agent_sequential(self, action, substeps):
        """Apply one optional stroke, then advance a fixed number of frames."""
        action = np.asarray(action, dtype=np.float32).flatten()
        expected_size = self._make_action_space().shape[0]
        if action.shape[0] != expected_size:
            raise ValueError(
                f"Sequential action has {action.shape[0]} values; "
                f"expected {expected_size}."
            )

        draw_requested = bool(action[0] > 0.0)
        draw_success = False
        invalid_draw = False

        if draw_requested:
            draw_action = action[1:]
            segment_count_before = len(self.agent_segments)
            if self.agent_draw_mode == "stroke":
                draw_success = self._apply_agent_stroke_action(draw_action)
            else:
                draw_success = self._apply_draw_action(draw_action)
                draw_success = (
                    draw_success and len(self.agent_segments) > segment_count_before
                )
            invalid_draw = not draw_success

        # A no-draw action is valid: the policy can wait and observe motion.
        self.simulation_started = True
        self.collisions_since_last_decision = 0
        self.object_collisions_since_last_decision = {
            object_id: 0 for object_id in self.objects
        }

        frames_advanced = 0
        frames_to_advance = min(
            self.agent_decision_frames,
            max(0, self.max_steps - self.step_count),
        )
        for _ in range(frames_to_advance):
            self.step_count += 1
            frames_advanced += 1

            if self._advance_physics_frame(substeps):
                self.goal_reached = True

            if self.goal_reached:
                break

        current_goal_distance = self._distance_to_first_goal()
        current_goal_score = self._goal_progress_score()
        reward = self._compute_sequential_reward(
            previous_score=self.prev_goal_score,
            current_score=current_goal_score,
            frames_advanced=frames_advanced,
            drew_stroke=draw_success,
            invalid_draw=invalid_draw,
        )
        self.prev_goal_distance = current_goal_distance
        self.prev_goal_score = current_goal_score

        terminated = self.goal_reached
        truncated = self.step_count >= self.max_steps and not terminated
        if truncated:
            reward -= 50.0

        obs = self._get_obs()
        info = {
            "is_success": terminated,
            "is_failure": truncated,
            "failure_reason": "timeout" if truncated else None,
            "step_count": self.step_count,
            "frames_advanced": frames_advanced,
            "decision_frames": self.agent_decision_frames,
            "draw_requested": draw_requested,
            "draw_success": draw_success,
            "invalid_draw": invalid_draw,
            "segments_used": len(self.agent_segments),
            "goal_distance": current_goal_distance,
            "collisions_since_decision": self.collisions_since_last_decision,
            "total_collisions": self.collision_count,
            "level_id": self.level_id,
            "level_title": self.level_data.get("title", "Untitled Level"),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _compute_sequential_reward(
        self,
        *,
        previous_score,
        current_score,
        frames_advanced,
        drew_stroke,
        invalid_draw,
    ):
        """Reward progress while making unnecessary extra strokes costly."""
        if self.reward_mode == "sparse":
            reward = 100.0 if self.goal_reached else 0.0
        else:
            progress = current_score - previous_score
            reward = -0.01 * frames_advanced
            reward += 10.0 * progress
            reward += 0.05 * current_score
            if self.goal_reached:
                reward += 100.0

        if drew_stroke:
            reward -= self.sequential_stroke_penalty
        if invalid_draw:
            reward -= 2.0
        return float(reward)
    
    def _apply_agent_stroke_action(self, action):
        if len(self.agent_segments) >= self.max_drawn_segments:
            return False

        action = self._action_to_draw_coordinates(action)

        if action.shape[0] != self.num_stroke_points * 2:
            return False

        points = []

        for i in range(self.num_stroke_points):
            x = self._clamp(action[2 * i], self.draw_region["x_min"], self.draw_region["x_max"])
            y = self._clamp(action[2 * i + 1], self.draw_region["y_min"], self.draw_region["y_max"])
            points.append((x, y))

        total_length = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if total_length < 25:
            self.last_line_length = total_length
            self.last_action = action
            return False

        if not self._apply_stroke_points(points):
            return False

        if self.render_mode == "human":
            print("PPO drew stroke:", points)

        return True
    
    def _apply_stroke_points(self, points):
        if self.space is None:
            return False

        if len(self.agent_segments) >= self.max_drawn_segments:
            return False

        if len(points) < 2:
            return False

        draw_rules = self.level_data.get("draw_rules", {})
        friction = float(draw_rules.get("friction", 0.9))
        elasticity = float(draw_rules.get("elasticity", 0.3))
        stroke_mass = float(draw_rules.get("stroke_mass", 2.0))

        cleaned_points = []
        min_dist = 5.0

        # Clean and clamp points
        for p in points:
            x = self._clamp(p[0], self.draw_region["x_min"], self.draw_region["x_max"])
            y = self._clamp(p[1], self.draw_region["y_min"], self.draw_region["y_max"])

            if not cleaned_points:
                cleaned_points.append((x, y))
            else:
                px, py = cleaned_points[-1]
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

                if dist >= min_dist:
                    cleaned_points.append((x, y))

        if len(cleaned_points) < 2:
            return False

        stroke_shapes = []
        stroke_bodies = []

        # -------------------------------------------------
        # CASE 1: Dynamic stroke
        # One body + many segment shapes
        # -------------------------------------------------
        if self.stroke_body_type == "dynamic":
            center_x = sum(p[0] for p in cleaned_points) / len(cleaned_points)
            center_y = sum(p[1] for p in cleaned_points) / len(cleaned_points)

            local_points = [
                (x - center_x, y - center_y)
                for x, y in cleaned_points
            ]

            min_x = min(p[0] for p in local_points)
            max_x = max(p[0] for p in local_points)
            min_y = min(p[1] for p in local_points)
            max_y = max(p[1] for p in local_points)

            width = max(10.0, max_x - min_x)
            height = max(10.0, max_y - min_y)

            moment = pymunk.moment_for_box(stroke_mass, (width, height))

            body = pymunk.Body(stroke_mass, moment)
            body.position = (center_x, center_y)

            for i in range(len(local_points) - 1):
                local_a = local_points[i]
                local_b = local_points[i + 1]

                shape = pymunk.Segment(
                    body,
                    local_a,
                    local_b,
                    self.agent_segment_thickness / 2.0,
                )

                shape.friction = friction
                shape.elasticity = elasticity

                stroke_shapes.append(shape)

            self.space.add(body, *stroke_shapes)
            stroke_bodies.append(body)

        # -------------------------------------------------
        # CASE 2: Static stroke
        # Static body + many fixed segment shapes
        # -------------------------------------------------
        else:
            body = self.space.static_body

            for i in range(len(cleaned_points) - 1):
                x1, y1 = cleaned_points[i]
                x2, y2 = cleaned_points[i + 1]

                shape = pymunk.Segment(
                    body,
                    (x1, y1),
                    (x2, y2),
                    self.agent_segment_thickness / 2.0,
                )

                shape.friction = friction
                shape.elasticity = elasticity

                self.space.add(shape)
                stroke_shapes.append(shape)

            stroke_bodies.append(body)

        stroke_data_points = [
            {"x": x, "y": y}
            for x, y in cleaned_points
        ]

        stroke_data = {
            "type": "stroke",
            "body_type": self.stroke_body_type,
            "points": stroke_data_points,
            "bodies": stroke_bodies,
            "shapes": stroke_shapes,
            "thickness": self.agent_segment_thickness,
        }

        self.agent_segments.append(stroke_data)

        # Calculate total stroke length
        total_length = 0.0
        for i in range(len(cleaned_points) - 1):
            x1, y1 = cleaned_points[i]
            x2, y2 = cleaned_points[i + 1]
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        self.last_line_length = total_length

        self.last_action = np.array(
            [
                cleaned_points[0][0],
                cleaned_points[0][1],
                cleaned_points[-1][0],
                cleaned_points[-1][1],
            ],
            dtype=np.float32,
        )

        self.simulation_started = True
        return True
       

    def _apply_draw_action(self, action):
        if len(self.agent_segments) >= self.max_drawn_segments:
            return False

        action = self._action_to_draw_coordinates(action)
        if action.shape[0] != 4:
            return False

        x1, y1, x2, y2 = action.tolist()

        x1 = self._clamp(x1, self.draw_region["x_min"], self.draw_region["x_max"])
        y1 = self._clamp(y1, self.draw_region["y_min"], self.draw_region["y_max"])
        x2 = self._clamp(x2, self.draw_region["x_min"], self.draw_region["x_max"])
        y2 = self._clamp(y2, self.draw_region["y_min"], self.draw_region["y_max"])

        if abs(x1 - x2) < 1e-3 and abs(y1 - y2) < 1e-3:
            x2 = min(self.draw_region["x_max"], x1 + 40.0)

        draw_rules = self.level_data.get("draw_rules", {})
        friction = float(draw_rules.get("friction", 0.9))
        elasticity = float(draw_rules.get("elasticity", 0.3))
        stroke_mass = float(draw_rules.get("stroke_mass", 1.0))

        if self.stroke_body_type == "dynamic":
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0

            local_a = (x1 - mid_x, y1 - mid_y)
            local_b = (x2 - mid_x, y2 - mid_y)

            moment = pymunk.moment_for_segment(
                stroke_mass,
                local_a,
                local_b,
                self.agent_segment_thickness / 2.0,
            )

            body = pymunk.Body(stroke_mass, moment)
            body.position = (mid_x, mid_y)

            shape = pymunk.Segment(
                body,
                local_a,
                local_b,
                self.agent_segment_thickness / 2.0,
            )

            shape.friction = friction
            shape.elasticity = elasticity

            self.space.add(body, shape)

        else:
            body = self.space.static_body

            shape = pymunk.Segment(
                body,
                (x1, y1),
                (x2, y2),
                self.agent_segment_thickness / 2.0,
            )

            shape.friction = friction
            shape.elasticity = elasticity

            self.space.add(shape)
       

        segment_data = {
            "type": "line",
            "body_type": self.stroke_body_type,
            "body": body,
            "a": {"x": x1, "y": y1},
            "b": {"x": x2, "y": y2},
            "thickness": self.agent_segment_thickness,
            "shape": shape,
        }

        self.agent_segments.append(segment_data)
        self.last_action = action
        self.last_line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Start simulation after first valid stroke
        if not self.simulation_started:
            self.simulation_started = True
        if self.render_mode == "human":
            print(f"PPO drew line: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
        return True

    def _compute_reward(self, prev_score, current_score):
        if self.reward_mode == "sparse":
            return 100.0 if self.goal_reached else 0.0

        reward = -0.02

        progress = current_score - prev_score
        reward += 10.0 * progress

        reward -= 0.03 * len(self.agent_segments)

        if self.last_action is not None and self.last_line_length < 25:
            reward -= 1.0

        reward += 0.05 * current_score

        if self.goal_reached:
            reward += 100.0

        return float(reward)

    def _goal_progress_score(self):
        goals = self.level_data.get("goals", [])
        if not goals:
            return 0.0

        scores = [self._single_goal_score(goal) for goal in goals]

        if self.level_data.get("goal_logic", "all") == "any":
            return float(max(scores))

        return float(sum(scores) / len(scores))

    def _single_goal_score(self, goal):
        object_id = goal.get("object_id")
        if object_id not in self.objects:
            return 0.0

        obj = self.objects[object_id]
        body = obj["body"]
        goal_type = goal["type"]
        diag = math.sqrt(self.canvas_width ** 2 + self.canvas_height ** 2)

        if goal_type in ("object_reaches_region", "object_stays_in_region"):
            dist = self._distance_object_to_goal_region(
                obj,
                goal,
                require_containment=goal_type == "object_stays_in_region",
            )
            return np.clip(1.0 - (dist / max(1.0, diag)), -1.0, 1.0)

        if goal_type == "object_below_height":
            target_y = float(goal["y"])
            return np.clip(
                (float(obj["shape"].bb.top) - target_y) / self.canvas_height,
                -1.0,
                1.0,
            )

        if goal_type == "object_above_height":
            target_y = float(goal["y"])
            return np.clip(
                (target_y - float(obj["shape"].bb.bottom)) / self.canvas_height,
                -1.0,
                1.0,
            )

        if goal_type == "object_stays_above_y":
            min_y = float(goal["min_y"])
            return np.clip(
                (min_y - float(obj["shape"].bb.top)) / self.canvas_height,
                -1.0,
                1.0,
            )

        if goal_type == "object_velocity_below":
            max_speed = float(goal.get("max_speed", 10.0))
            speed = float(body.velocity.length)
            return np.clip((max_speed - speed) / self.MAX_SPEED, -1.0, 1.0)

        if goal_type == "object_angle_range":
            min_angle = float(goal["min_angle"])
            max_angle = float(goal["max_angle"])
            center = (min_angle + max_angle) / 2.0
            half_width = max((max_angle - min_angle) / 2.0, 1e-6)
            angle_error = abs(float(body.angle) - center)
            return np.clip(1.0 - (angle_error / half_width), -1.0, 1.0)

        return 0.0

    def _distance_object_to_goal_region(self, obj, goal, require_containment=False):
        shape = obj["shape"]
        region = goal.get("region", {})

        if region.get("shape") == "circle":
            cx = float(region.get("x", 0.0))
            cy = float(region.get("y", 0.0))
            radius = float(region.get("r", 0.0))
            if require_containment:
                if obj["type"] == "circle":
                    centre = shape.body.local_to_world(shape.offset)
                    farthest_distance = math.hypot(centre.x - cx, centre.y - cy) + float(
                        shape.radius
                    )
                else:
                    vertices = self.goal_evaluator._shape_world_vertices(shape)
                    farthest_distance = max(
                        (math.hypot(x - cx, y - cy) for x, y in vertices),
                        default=0.0,
                    )
                return max(0.0, farthest_distance - radius)
            return max(0.0, float(shape.point_query((cx, cy)).distance) - radius)

        if region.get("shape") == "rectangle":
            rx = float(region.get("x", 0.0))
            ry = float(region.get("y", 0.0))
            right = rx + float(region.get("width", 0.0))
            bottom = ry + float(region.get("height", 0.0))
            bb = shape.bb
            if require_containment:
                return max(
                    0.0,
                    rx - float(bb.left),
                    float(bb.right) - right,
                    ry - float(bb.bottom),
                    float(bb.top) - bottom,
                )
            dx = max(rx - float(bb.right), 0.0, float(bb.left) - right)
            dy = max(ry - float(bb.top), 0.0, float(bb.bottom) - bottom)
            return math.hypot(dx, dy)

        return 0.0

    def _get_obs(self):
        dynamic_objects = [
            obj for obj in self.objects.values()
            if obj["config"].get("dynamic", True)
        ]

        if not dynamic_objects:
            return np.zeros(self.OBS_SIZE, dtype=np.float32)

        obj = dynamic_objects[0]
        body = obj["body"]

        x = body.position.x / self.canvas_width
        y = body.position.y / self.canvas_height
        vx = np.clip(body.velocity.x / self.MAX_SPEED, -1.0, 1.0)
        vy = np.clip(body.velocity.y / self.MAX_SPEED, -1.0, 1.0)

        gx, gy = self._first_goal_center()

        gx_norm = gx / self.canvas_width
        gy_norm = gy / self.canvas_height

        rel_x = (gx - body.position.x) / self.canvas_width
        rel_y = (gy - body.position.y) / self.canvas_height

        step_norm = min(1.0, self.step_count / max(1, self.max_steps))
        seg_norm = min(1.0, len(self.agent_segments) / max(1, self.max_drawn_segments))

        draw_x_min = self.draw_region["x_min"] / self.canvas_width
        draw_x_max = self.draw_region["x_max"] / self.canvas_width
        draw_y_min = self.draw_region["y_min"] / self.canvas_height
        draw_y_max = self.draw_region["y_max"] / self.canvas_height

        obj_width = float(obj["config"].get("width", obj["config"].get("radius", 0.0) * 2.0))
        obj_height = float(obj["config"].get("height", obj["config"].get("radius", 0.0) * 2.0))
        obj_w_norm = np.clip(obj_width / self.canvas_width, 0.0, 1.0)
        obj_h_norm = np.clip(obj_height / self.canvas_height, 0.0, 1.0)

        gravity_norm = np.clip(float(self.space.gravity.y) / 1000.0, -1.0, 1.0)

        goal_features = self._goal_obs_features()
        static_features = self._static_segment_obs_features()

        obs = np.array(
            [
                x,
                y,
                vx,
                vy,
                gx_norm,
                gy_norm,
                rel_x,
                rel_y,
                step_norm,
                seg_norm,
                draw_x_min,
                draw_x_max,
                draw_y_min,
                draw_y_max,
                obj_w_norm,
                obj_h_norm,
                gravity_norm,
                *goal_features,
                *static_features,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def _goal_obs_features(self):
        features = []
        goals = self.level_data.get("goals", [])
        goal = goals[0] if goals else {}
        goal_type = goal.get("type", "")

        for known_type in self.GOAL_TYPES:
            features.append(1.0 if goal_type == known_type else 0.0)

        region = goal.get("region", {})
        goal_param_1 = 0.0
        goal_param_2 = 0.0
        goal_param_3 = 0.0
        goal_param_4 = 0.0

        if region.get("shape") == "circle":
            goal_param_1 = float(region.get("r", 0.0)) / self.canvas_width
        elif region.get("shape") == "rectangle":
            goal_param_1 = float(region.get("width", 0.0)) / self.canvas_width
            goal_param_2 = float(region.get("height", 0.0)) / self.canvas_height
        elif goal_type in ("object_above_height", "object_below_height"):
            goal_param_1 = float(goal.get("y", 0.0)) / self.canvas_height
        elif goal_type == "object_stays_above_y":
            goal_param_1 = float(goal.get("min_y", 0.0)) / self.canvas_height
            goal_param_2 = float(goal.get("duration_steps", 0.0)) / max(1.0, self.max_steps)
        elif goal_type == "object_velocity_below":
            goal_param_1 = float(goal.get("max_speed", 0.0)) / self.MAX_SPEED
        elif goal_type == "object_angle_range":
            goal_param_1 = float(goal.get("min_angle", 0.0)) / math.pi
            goal_param_2 = float(goal.get("max_angle", 0.0)) / math.pi

        goal_param_3 = np.clip(self._goal_progress_score(), -1.0, 1.0)
        goal_param_4 = 1.0 if self.level_data.get("goal_logic", "all") == "any" else 0.0

        features.extend([goal_param_1, goal_param_2, goal_param_3, goal_param_4])
        return features

    def _static_segment_obs_features(self):
        if not self.level_data.get("static_segments"):
            return [0.0, 0.0, 0.0, 0.0]

        seg = self.level_data["static_segments"][0]
        a = seg.get("a", {})
        b = seg.get("b", {})

        return [
            float(a.get("x", 0.0)) / self.canvas_width,
            float(a.get("y", 0.0)) / self.canvas_height,
            float(b.get("x", 0.0)) / self.canvas_width,
            float(b.get("y", 0.0)) / self.canvas_height,
        ]

    def _first_goal_center(self):
        goals = self.level_data.get("goals", [])
        if not goals:
            return 0.0, 0.0

        region = goals[0].get("region", {})
        x = float(region.get("x", 0.0))
        y = float(region.get("y", 0.0))
        if region.get("shape") == "rectangle":
            x += float(region.get("width", 0.0)) / 2.0
            y += float(region.get("height", 0.0)) / 2.0
        return x, y

    def _distance_to_first_goal(self):
        goals = self.level_data.get("goals", [])
        if not goals:
            return 0.0

        object_id = goals[0].get("object_id")
        if object_id not in self.objects:
            return 0.0

        goal = goals[0]
        obj = self.objects[object_id]
        body = obj["body"]
        goal_type = goal.get("type")

        if goal_type in ("object_reaches_region", "object_stays_in_region"):
            return self._distance_object_to_goal_region(
                obj,
                goal,
                require_containment=goal_type == "object_stays_in_region",
            )
        if goal_type == "object_above_height":
            return max(0.0, float(obj["shape"].bb.bottom) - float(goal["y"]))
        if goal_type == "object_below_height":
            return max(0.0, float(goal["y"]) - float(obj["shape"].bb.top))
        if goal_type == "object_stays_above_y":
            return max(0.0, float(obj["shape"].bb.top) - float(goal["min_y"]))
        if goal_type == "object_velocity_below":
            return max(0.0, float(body.velocity.length) - float(goal["max_speed"]))
        if goal_type == "object_angle_range":
            angle = float(body.angle)
            return max(
                float(goal["min_angle"]) - angle,
                0.0,
                angle - float(goal["max_angle"]),
            )
        return 0.0

    def render(self):
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.font.init()

            if self.render_mode == "human":
                pygame.display.set_caption("BrainItOn General Env")
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            else:
                self.screen = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18)

        self.screen.fill((245, 245, 248))


        # ==========================================
        # DRAW ACTION SPACE OVERLAY
        # ==========================================

        draw_region = self.draw_region

        x_min = int(draw_region["x_min"])
        x_max = int(draw_region["x_max"])
        y_min = int(draw_region["y_min"]) + self.ui_height
        y_max = int(draw_region["y_max"]) + self.ui_height

        # Transparent overlay
        overlay = pygame.Surface(
            (self.window_width, self.window_height),
            pygame.SRCALPHA
        )

        # Fill full gameplay area with light red
        pygame.draw.rect(
            overlay,
            (255, 180, 180, 70),
            pygame.Rect(
                0,
                self.ui_height,
                self.window_width,
                self.canvas_height
            )
        )

        # Draw action space as white
        pygame.draw.rect(
            overlay,
            (255, 255, 255, 140),
            pygame.Rect(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            )
        )

        # Optional blue border
        pygame.draw.rect(
            overlay,
            (50, 120, 255, 255),
            pygame.Rect(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            ),
            3
        )

        self.screen.blit(overlay, (0, 0))

# ==========================================

        self._draw_goals()
        self._draw_joints()
        self._draw_static_segments()
        self._draw_agent_segments()
        self._draw_objects()
        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def _draw_goals(self):
        for goal in self.level_data.get("goals", []):
            region = goal.get("region", {})

            if region.get("shape") == "circle":
                pygame.draw.circle(
                    self.screen,
                    (100, 220, 120),
                    (int(region["x"]), int(region["y"] + self.ui_height)),
                    int(region["r"]),
                )

            elif region.get("shape") == "rectangle":
                pygame.draw.rect(
                    self.screen,
                    (100, 220, 120),
                    pygame.Rect(
                        int(region["x"]),
                        int(region["y"] + self.ui_height),
                        int(region["width"]),
                        int(region["height"]),
                    ),
                    width=2,
                )

    def _draw_static_segments(self):
        for seg in self.level_data.get("static_segments", []):
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (int(seg["a"]["x"]), int(seg["a"]["y"])+self.ui_height),
                (int(seg["b"]["x"]), int(seg["b"]["y"])+self.ui_height),
                int(seg.get("thickness", 8.0)),
            )

    def _draw_joints(self):
        for joint_data in self.joints:
            anchor_a, anchor_b = self._joint_world_anchors(joint_data)
            a = (int(anchor_a.x), int(anchor_a.y + self.ui_height))
            b = (int(anchor_b.x), int(anchor_b.y + self.ui_height))
            pygame.draw.line(self.screen, (90, 90, 120), a, b, 3)
            pygame.draw.circle(self.screen, (60, 60, 90), a, 6)
            pygame.draw.circle(self.screen, (60, 60, 90), b, 5)

    def _draw_agent_segments(self):
        for seg in self.agent_segments:
            thickness = int(seg.get("thickness", self.agent_segment_thickness))

            if seg.get("type") == "stroke":
                if seg.get("body_type") == "dynamic":
                    for shape in seg.get("shapes", []):
                        body = shape.body

                        a = body.local_to_world(shape.a)
                        b = body.local_to_world(shape.b)

                        pygame.draw.line(
                            self.screen,
                            (40, 120, 220),
                            (int(a.x), int(a.y + self.ui_height)),
                            (int(b.x), int(b.y + self.ui_height)),
                            thickness,
                        )
                else:
                    points = seg.get("points", [])

                    if len(points) >= 2:
                        pygame_points = [
                            (int(p["x"]), int(p["y"] + self.ui_height))
                            for p in points
                        ]

                        pygame.draw.lines(
                            self.screen,
                            (40, 120, 220),
                            False,
                            pygame_points,
                            thickness,
                        )

            else:
                if seg.get("body_type") == "dynamic":
                    shape = seg["shape"]
                    body = seg["body"]

                    a = body.local_to_world(shape.a)
                    b = body.local_to_world(shape.b)

                    pygame.draw.line(
                        self.screen,
                        (40, 120, 220),
                        (int(a.x), int(a.y + self.ui_height)),
                        (int(b.x), int(b.y + self.ui_height)),
                        thickness,
                    )
                else:
                    pygame.draw.line(
                        self.screen,
                        (40, 120, 220),
                        (int(seg["a"]["x"]), int(seg["a"]["y"] + self.ui_height)),
                        (int(seg["b"]["x"]), int(seg["b"]["y"] + self.ui_height)),
                        thickness,
                    )
    def _draw_objects(self):
        for obj in self.objects.values():
            obj_type = obj["type"]
            body = obj["body"]
            config = obj["config"]
            color = tuple(config.get("color", [220, 70, 70]))

            if obj_type == "circle":
                radius = int(config["radius"])
                centre = obj["shape"].body.local_to_world(obj["shape"].offset)
                x = int(centre.x)
                y = int(centre.y + self.ui_height)

                pygame.draw.circle(self.screen, color, (x, y), radius)

                # rotation marker
                angle = body.angle
                marker_x = x + int(math.cos(angle) * radius)
                marker_y = y + int(math.sin(angle) * radius)

                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (x, y),
                    (marker_x, marker_y),
                    4,
                )

            elif obj_type == "box":
                vertices = obj["shape"].get_vertices()
                points = []

                for v in vertices:
                    world_v = body.local_to_world(v)
                    points.append((int(world_v.x), int(world_v.y)+self.ui_height))

                pygame.draw.polygon(self.screen, color, points)
            elif obj_type == "polygon":
                vertices = obj["shape"].get_vertices()
                points = []

                for v in vertices:
                    world_v = body.local_to_world(v)
                    points.append(
                        (int(world_v.x), int(world_v.y + self.ui_height))
                    )

                pygame.draw.polygon(self.screen, color, points)

    def _draw_hud(self):
        title = f"Level {self.level_id}: {self.level_data.get('title', 'Untitled')}"

        # Background top bar
        pygame.draw.rect(
            self.screen,
            (245, 245, 248),
            pygame.Rect(0, 0, self.window_width, self.ui_height),
        )

        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (0, self.ui_height - 1),
            (self.window_width, self.ui_height - 1),
            2,
        )

        title_font = pygame.font.SysFont("Arial", 22, bold=True)
        small_font = pygame.font.SysFont("Arial", 16)

        # Truncate title if too long
        max_title_chars = 55
        if len(title) > max_title_chars:
            title = title[:max_title_chars] + "..."

        title_surface = title_font.render(title, True, (25, 25, 25))
        self.screen.blit(title_surface, (16, 12))

        left_lines = [
            f"Mode: {self.control_mode} | Reward: {self.reward_mode}",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Segments: {len(self.agent_segments)}/{self.max_drawn_segments}",
        ]

        right_lines = [
            f"Success: {self.goal_reached}",
            # f"Speed: {speed:.2f}",
            # f"Angular velocity: {angular_velocity:.2f}",
            # f"Gravity: {gravity:.1f} | Damping: {damping:.3f} | Substeps: {substeps}",
        ]

        y = 45
        for line in left_lines:
            surface = small_font.render(line, True, (35, 35, 35))
            self.screen.blit(surface, (18, y))
            y += 22

        y = 45
        for line in right_lines:
            color = (20, 20, 20)

            if "Success: True" in line:
                color = (0, 150, 0)
            elif "Success: False" in line:
                color = (180, 0, 0)
            surface = small_font.render(line, True, color)
            self.screen.blit(surface, (360, y))
            y += 22

        
        hint = "R: reset | C: clear | N/P: level | Space: pause"

        hint_surface = small_font.render(hint, True, (80, 80, 80))
        self.screen.blit(hint_surface, (18, self.ui_height - 24))

    def clear_drawn_segments(self):
        if self.space is None:
            return

        for seg in self.agent_segments:
            if seg.get("type") == "stroke":
                for shape in seg.get("shapes", []):
                    if shape is not None and shape in self.space.shapes:
                        body = shape.body
                        self.space.remove(shape)

                        if body is not self.space.static_body and body in self.space.bodies:
                            self.space.remove(body)
            else:
                shape = seg.get("shape")
                body = seg.get("body")

                if shape is not None and shape in self.space.shapes:
                    self.space.remove(shape)

                if body is not None and body is not self.space.static_body and body in self.space.bodies:
                    self.space.remove(body)

        self.agent_segments.clear()
        self._reset_attempt_state()
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

        self.space = None
        self.objects = {}
        self.static_segments = []
        self.agent_segments = []
        self.joints = []

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))
