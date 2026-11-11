"""Benchmark OpenRouter LLMs and VLMs on sequential physics puzzles.

LLM mode receives the structured level design and live physics state. VLM mode
receives a gameplay screenshot plus the goal and action rules. Both modes make
one draw/wait decision every N physics frames and use the same environment.

No API calls are made unless this script is executed explicitly.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULT_FIELDS = [
    "timestamp_utc",
    "mode",
    "requested_model",
    "resolved_model",
    "level_id",
    "effective_stroke_body",
    "attempt",
    "success",
    "episode_reward",
    "physics_frames",
    "decisions",
    "draw_requests",
    "wait_requests",
    "strokes_used",
    "invalid_responses",
    "invalid_draws",
    "collisions",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reported_cost_usd",
    "failure_reason",
    "error",
]


class OpenRouterRequestError(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, maximum_usd: float):
        self.maximum_usd = max(0.0, float(maximum_usd))
        self.reported_spend_usd = 0.0

    @property
    def exhausted(self) -> bool:
        return (
            self.maximum_usd > 0.0
            and self.reported_spend_usd >= self.maximum_usd
        )

    def add_usage(self, usage: Dict[str, Any]) -> None:
        self.reported_spend_usd += usage_number(usage, "cost")


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_level_ids(value: str) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def load_levels(level_path: Path) -> Dict[str, Any]:
    with level_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_level(levels_data: Dict[str, Any], level_id: int) -> Dict[str, Any]:
    for level in levels_data.get("levels", []):
        if int(level.get("level_id")) == int(level_id):
            return level
    available = [level.get("level_id") for level in levels_data.get("levels", [])]
    raise ValueError(f"Level {level_id} was not found. Available levels: {available}")


def compact_level_design(level: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "level_id",
        "title",
        "canvas",
        "physics",
        "objects",
        "static_segments",
        "joints",
        "goals",
        "goal_logic",
        "draw_rules",
    )
    return {key: level[key] for key in keys if key in level}


def round_number(value: float) -> float:
    return round(float(value), 3)


def agent_segment_world_points(segment: Dict[str, Any]) -> list[list[float]]:
    if segment.get("body_type") == "dynamic":
        shapes = segment.get("shapes", [])
        if not shapes and segment.get("shape") is not None:
            shapes = [segment["shape"]]
        if not shapes:
            return []
        points = [shapes[0].body.local_to_world(shapes[0].a)]
        points.extend(shape.body.local_to_world(shape.b) for shape in shapes)
        return [[round_number(point.x), round_number(point.y)] for point in points]

    if segment.get("type") == "stroke":
        return [
            [round_number(point["x"]), round_number(point["y"])]
            for point in segment.get("points", [])
        ]

    return [
        [round_number(segment["a"]["x"]), round_number(segment["a"]["y"])],
        [round_number(segment["b"]["x"]), round_number(segment["b"]["y"])],
    ]


def current_scene_state(env: BrainItOnGeneralEnv) -> Dict[str, Any]:
    objects = []
    for config in env.level_data.get("objects", []):
        obj = env.objects[config["id"]]
        body = obj["body"]
        dynamic = bool(config.get("dynamic", True))
        if dynamic:
            position = [round_number(body.position.x), round_number(body.position.y)]
            velocity = [round_number(body.velocity.x), round_number(body.velocity.y)]
            angle = round_number(body.angle)
            angular_velocity = round_number(body.angular_velocity)
        else:
            position = [
                round_number(config.get("x", 0.0)),
                round_number(config.get("y", 0.0)),
            ]
            velocity = [0.0, 0.0]
            angle = round_number(math.radians(config.get("angle_degrees", 0.0)))
            angular_velocity = 0.0

        objects.append(
            {
                "id": config["id"],
                "type": config.get("type"),
                "dynamic": dynamic,
                "position": position,
                "velocity": velocity,
                "speed": round_number(math.hypot(*velocity)),
                "angle_radians": angle,
                "angular_velocity": angular_velocity,
                "recent_collisions": env.object_collisions_since_last_decision.get(
                    config["id"], 0
                ),
                "total_collisions": env.object_collision_counts.get(config["id"], 0),
            }
        )

    strokes = []
    for index, segment in enumerate(env.agent_segments):
        strokes.append(
            {
                "index": index,
                "body_type": segment.get("body_type"),
                "points": agent_segment_world_points(segment),
            }
        )

    goals = []
    for goal in env.level_data.get("goals", []):
        duration = int(goal.get("duration_steps", 0))
        hold_frames = int(env.goal_evaluator.goal_memory.get(goal.get("id"), 0))
        goals.append(
            {
                "id": goal.get("id"),
                "type": goal.get("type"),
                "object_id": goal.get("object_id"),
                "progress_score": round_number(env._single_goal_score(goal)),
                "hold_frames": hold_frames,
                "required_hold_frames": duration,
                "hold_progress": round_number(
                    hold_frames / duration if duration > 0 else 0.0
                ),
            }
        )

    joints = []
    for joint_data in env.joints:
        anchor_a, anchor_b = env._joint_world_anchors(joint_data)
        joints.append(
            {
                "id": joint_data["id"],
                "type": joint_data["config"].get("type"),
                "anchor_a": [round_number(anchor_a.x), round_number(anchor_a.y)],
                "anchor_b": [round_number(anchor_b.x), round_number(anchor_b.y)],
                "impulse": round_number(joint_data["constraint"].impulse),
            }
        )

    return {
        "physics_frame": env.step_count,
        "maximum_physics_frames": env.max_steps,
        "decision_interval_frames": env.agent_decision_frames,
        "strokes_used": len(env.agent_segments),
        "strokes_remaining": max(0, env.max_drawn_segments - len(env.agent_segments)),
        "recent_collisions": env.collisions_since_last_decision,
        "total_collisions": env.collision_count,
        "objects": objects,
        "joints": joints,
        "drawn_strokes": strokes,
        "goal_progress": goals,
    }


def compact_history(history: Sequence[Dict[str, Any]], limit: int) -> list[Dict[str, Any]]:
    if limit <= 0:
        return []
    return list(history[-limit:])


def action_instructions(env: BrainItOnGeneralEnv, num_points: int) -> str:
    draw = env.draw_region
    return f"""
Return ONLY a JSON object with this shape:
{{
  "action": "draw" or "wait",
  "stroke_points": [[x1, y1], ...],
  "reason": "brief explanation"
}}

Decision rules:
- If action is "draw", stroke_points must contain exactly {num_points} points.
- If action is "wait", stroke_points must be an empty list.
- Coordinates are game-canvas pixels, with (0, 0) at the top-left.
- Positive y points downward; gravity therefore usually increases y.
- Every draw point must satisfy x=[{draw['x_min']}, {draw['x_max']}] and
  y=[{draw['y_min']}, {draw['y_max']}].
- The stroke must have total length of at least 25 pixels.
- You may wait to observe motion before deciding whether another stroke is needed.
- Avoid unnecessary strokes because each stroke has a reward penalty.
- Do not include Markdown or text outside the JSON object.
""".strip()


def build_llm_prompt(
    env: BrainItOnGeneralEnv,
    level: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    num_points: int,
    history_limit: int,
) -> str:
    return f"""
You control a sequential Brain-It-On style 2D physics puzzle.
Study the complete level geometry and current physics state, then choose whether
to draw one new stroke or wait for the next observation.

LEVEL DESIGN:
{json.dumps(compact_level_design(level), separators=(',', ':'))}

CURRENT STATE:
{json.dumps(current_scene_state(env), separators=(',', ':'))}

RECENT DECISIONS:
{json.dumps(compact_history(history, history_limit), separators=(',', ':'))}

{action_instructions(env, num_points)}
""".strip()


def build_vlm_prompt(
    env: BrainItOnGeneralEnv,
    level: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    num_points: int,
    history_limit: int,
) -> str:
    visual_context = {
        "title": level.get("title"),
        "goals": level.get("goals", []),
        "goal_logic": level.get("goal_logic", "all"),
        "draw_rules": level.get("draw_rules", {}),
        "current_state": {
            "physics_frame": env.step_count,
            "strokes_used": len(env.agent_segments),
            "strokes_remaining": max(
                0, env.max_drawn_segments - len(env.agent_segments)
            ),
            "recent_collisions": env.collisions_since_last_decision,
            "goal_hold_progress": [
                {
                    "id": goal.get("id"),
                    "hold_frames": env.goal_evaluator.goal_memory.get(
                        goal.get("id"), 0
                    ),
                    "required_hold_frames": goal.get("duration_steps", 0),
                }
                for goal in level.get("goals", [])
                if goal.get("duration_steps")
            ],
        },
        "recent_decisions": compact_history(history, history_limit),
    }
    return f"""
You control a sequential Brain-It-On style 2D physics puzzle.
Inspect the attached current gameplay image and choose whether to draw one new
stroke or wait. The image is cropped to the game canvas, so its image pixel
coordinates exactly match the required game coordinates. Red shapes are usually
controlled objects, green areas are goals, dark lines are fixed obstacles, and
blue lines are strokes already drawn.

TASK CONTEXT:
{json.dumps(visual_context, separators=(',', ':'))}

{action_instructions(env, num_points)}
""".strip()


def frame_to_png_data_url(frame: np.ndarray, ui_height: int) -> str:
    """Crop the HUD so VLM image coordinates equal game coordinates."""
    gameplay = np.ascontiguousarray(frame[ui_height:, :, :3], dtype=np.uint8)
    height, width = gameplay.shape[:2]
    surface = pygame.image.frombuffer(gameplay.tobytes(), (width, height), "RGB")
    buffer = io.BytesIO()
    pygame.image.save(surface, buffer, "scene.png")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def response_schema(num_points: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "physics_puzzle_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["draw", "wait"]},
                    "stroke_points": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "minItems": 0,
                        "maxItems": num_points,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["action", "stroke_points", "reason"],
                "additionalProperties": False,
            },
        },
    }


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        retries: int,
        structured_output: bool,
        referer: Optional[str],
        title: str,
    ):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.structured_output = structured_output
        self.referer = referer
        self.title = title

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        num_points: int,
        image_data_url: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], str, float]:
        user_content: Any = prompt
        if image_data_url is not None:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a physics-puzzle control policy. Follow the "
                        "requested JSON action format exactly."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.structured_output:
            payload["response_format"] = response_schema(num_points)
            payload["provider"] = {"require_parameters": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.title,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer

        encoded_payload = json.dumps(payload).encode("utf-8")
        started = time.monotonic()
        last_error = None

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                OPENROUTER_CHAT_URL,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise OpenRouterRequestError(
                        "OpenRouter returned a non-object JSON response"
                    )
                if result.get("error"):
                    error = result["error"]
                    if isinstance(error, dict):
                        error = error.get("message") or json.dumps(error)
                    raise OpenRouterRequestError(f"OpenRouter error: {error}")
                choices = result.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise OpenRouterRequestError(
                        "OpenRouter response did not contain any choices"
                    )
                if not isinstance(choices[0], dict):
                    raise OpenRouterRequestError(
                        "OpenRouter response contained an invalid choice"
                    )
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise OpenRouterRequestError(
                        "OpenRouter response choice did not contain a message"
                    )
                usage = result.get("usage", {}) or {}
                if not isinstance(usage, dict):
                    raise OpenRouterRequestError(
                        "OpenRouter response contained invalid usage data"
                    )
                latency = time.monotonic() - started
                return (
                    message_text(message.get("content", "")),
                    usage,
                    result.get("model", model),
                    latency,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = OpenRouterRequestError(
                    f"OpenRouter HTTP {exc.code}: {body[:1000]}"
                )
                retryable = exc.code in (408, 409, 429) or exc.code >= 500
                if not retryable or attempt >= self.retries:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = OpenRouterRequestError(f"OpenRouter request failed: {exc}")
                if attempt >= self.retries:
                    raise last_error from exc
            except OpenRouterRequestError as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise

            time.sleep(min(2**attempt, 8))

        raise last_error or OpenRouterRequestError("OpenRouter request failed")


class DryRunClient:
    """Exercise the benchmark loop without network use or API charges."""

    def chat(self, *, model, prompt, num_points, image_data_url=None):
        if image_data_url is not None and not image_data_url.startswith("data:image/png"):
            raise ValueError("Dry-run VLM did not receive a PNG data URL")
        response = json.dumps(
            {"action": "wait", "stroke_points": [], "reason": "dry run"}
        )
        return response, {}, model, 0.0


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "".join(texts)
    return str(content)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("The model response did not contain a JSON object")
        result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("The model response JSON must be an object")
    return result


def model_decision_to_action(
    decision: Dict[str, Any],
    env: BrainItOnGeneralEnv,
    num_points: int,
) -> Tuple[np.ndarray, str, list[list[float]]]:
    action_name = str(decision.get("action", "")).strip().lower()
    points = decision.get("stroke_points")
    if action_name == "wait":
        if points not in ([], None):
            raise ValueError("A wait action must have an empty stroke_points list")
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0] = -1.0
        return action, "wait", []

    if action_name != "draw":
        raise ValueError("action must be either 'draw' or 'wait'")
    if not isinstance(points, list) or len(points) != num_points:
        raise ValueError(f"A draw action must contain exactly {num_points} points")

    draw = env.draw_region
    clean_points = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid stroke point: {point!r}")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Stroke coordinates must be finite numbers")
        if not (draw["x_min"] <= x <= draw["x_max"]):
            raise ValueError(f"Stroke x={x} is outside the drawing region")
        if not (draw["y_min"] <= y <= draw["y_max"]):
            raise ValueError(f"Stroke y={y} is outside the drawing region")
        clean_points.append([x, y])

    length = sum(
        math.hypot(bx - ax, by - ay)
        for (ax, ay), (bx, by) in zip(clean_points, clean_points[1:])
    )
    if length < 25.0:
        raise ValueError(f"Stroke length {length:.2f} is below 25 pixels")

    values = [1.0]
    for x, y in clean_points:
        x_norm = 2.0 * (x - draw["x_min"]) / (draw["x_max"] - draw["x_min"]) - 1.0
        y_norm = 2.0 * (y - draw["y_min"]) / (draw["y_max"] - draw["y_min"]) - 1.0
        values.extend([x_norm, y_norm])
    return np.asarray(values, dtype=np.float32), "draw", clean_points


def invalid_response_action(env: BrainItOnGeneralEnv) -> np.ndarray:
    """Request a zero-length draw so the environment applies its invalid penalty."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    return action


def usage_number(usage: Dict[str, Any], key: str) -> float:
    value = usage.get(key, 0) or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_episode(
    *,
    client,
    mode: str,
    model: str,
    level: Dict[str, Any],
    level_path: Path,
    attempt: int,
    args: argparse.Namespace,
    trace_file,
    budget: BudgetTracker,
) -> Dict[str, Any]:
    render_mode = "rgb_array" if mode == "vlm" else None
    env = BrainItOnGeneralEnv(
        level_path=str(level_path),
        level_id=int(level["level_id"]),
        render_mode=render_mode,
        reward_mode=args.reward_mode,
        control_mode="agent",
        max_steps=args.max_steps,
        stroke_body=args.stroke_body,
        agent_draw_mode="stroke",
        num_stroke_points=args.num_stroke_points,
        agent_decision_frames=args.decision_frames,
        sequential_stroke_penalty=args.stroke_penalty,
    )
    env.reset(seed=args.seed + attempt - 1)

    history = []
    total_reward = 0.0
    invalid_responses = 0
    invalid_draws = 0
    draw_requests = 0
    wait_requests = 0
    total_latency = 0.0
    prompt_tokens = completion_tokens = total_tokens = reported_cost = 0.0
    resolved_model = model
    final_info: Dict[str, Any] = {}
    episode_error = ""

    try:
        for decision_index in range(1, args.max_decisions + 1):
            if budget.exhausted:
                final_info = {
                    **final_info,
                    "is_success": False,
                    "failure_reason": "cost_limit",
                    "step_count": env.step_count,
                    "segments_used": len(env.agent_segments),
                    "total_collisions": env.collision_count,
                }
                break

            if mode == "llm":
                prompt = build_llm_prompt(
                    env,
                    level,
                    history,
                    args.num_stroke_points,
                    args.history_limit,
                )
                image_data_url = None
            else:
                prompt = build_vlm_prompt(
                    env,
                    level,
                    history,
                    args.num_stroke_points,
                    args.history_limit,
                )
                frame = env.render()
                image_data_url = frame_to_png_data_url(frame, env.ui_height)

            raw_response, usage, resolved_model, latency = client.chat(
                model=model,
                prompt=prompt,
                num_points=args.num_stroke_points,
                image_data_url=image_data_url,
            )
            total_latency += latency
            prompt_tokens += usage_number(usage, "prompt_tokens")
            completion_tokens += usage_number(usage, "completion_tokens")
            total_tokens += usage_number(usage, "total_tokens")
            reported_cost += usage_number(usage, "cost")
            budget.add_usage(usage)

            validation_error = ""
            points = []
            try:
                decision = extract_json_object(raw_response)
                action, action_name, points = model_decision_to_action(
                    decision, env, args.num_stroke_points
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                invalid_responses += 1
                validation_error = str(exc)
                decision = {
                    "action": "invalid",
                    "stroke_points": [],
                    "reason": validation_error,
                }
                action_name = "invalid"
                action = invalid_response_action(env)

            if action_name == "draw":
                draw_requests += 1
            elif action_name == "wait":
                wait_requests += 1

            _, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            invalid_draws += int(final_info.get("invalid_draw", False))

            history_entry = {
                "decision": decision_index,
                "frame_after_action": final_info.get("step_count"),
                "action": action_name,
                "stroke_points": points,
                "reason": str(decision.get("reason", ""))[:500],
                "reward": round_number(reward),
                "success": bool(final_info.get("is_success", False)),
                "collisions": final_info.get("collisions_since_decision", 0),
            }
            history.append(history_entry)

            trace_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "requested_model": model,
                "resolved_model": resolved_model,
                "level_id": int(level["level_id"]),
                "attempt": attempt,
                **history_entry,
                "validation_error": validation_error,
                "raw_response": raw_response,
                "latency_seconds": latency,
                "usage": usage,
            }
            trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            trace_file.flush()

            print(
                f"[{mode.upper()}] {model} | level={level['level_id']} "
                f"attempt={attempt} decision={decision_index} "
                f"frame={final_info.get('step_count')} action={action_name} "
                f"reward={float(reward):.3f} "
                f"success={final_info.get('is_success', False)}"
            )

            if terminated or truncated:
                break
        else:
            final_info = {
                **final_info,
                "is_success": False,
                "failure_reason": "decision_limit",
                "step_count": env.step_count,
                "segments_used": len(env.agent_segments),
                "total_collisions": env.collision_count,
            }
    except OpenRouterRequestError as exc:
        episode_error = str(exc)
        final_info = {
            **final_info,
            "is_success": False,
            "failure_reason": "api_error",
            "step_count": env.step_count,
            "segments_used": len(env.agent_segments),
            "total_collisions": env.collision_count,
        }
    finally:
        env.close()

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "requested_model": model,
        "resolved_model": resolved_model,
        "level_id": int(level["level_id"]),
        "effective_stroke_body": env.stroke_body_type,
        "attempt": attempt,
        "success": bool(final_info.get("is_success", False)),
        "episode_reward": total_reward,
        "physics_frames": int(final_info.get("step_count", 0)),
        "decisions": len(history),
        "draw_requests": draw_requests,
        "wait_requests": wait_requests,
        "strokes_used": int(final_info.get("segments_used", 0)),
        "invalid_responses": invalid_responses,
        "invalid_draws": invalid_draws,
        "collisions": int(final_info.get("total_collisions", 0)),
        "latency_seconds": total_latency,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "reported_cost_usd": reported_cost,
        "failure_reason": final_info.get("failure_reason") or "",
        "error": episode_error,
    }


def print_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups = defaultdict(list)
    level_groups = defaultdict(list)
    for result in results:
        groups[(result["mode"], result["requested_model"])].append(result)
        level_groups[
            (result["mode"], result["requested_model"], result["level_id"])
        ].append(result)

    summary = {"overall": {}, "by_level": {}}
    print("\nBenchmark summary")
    print("-" * 100)
    for (mode, model), rows in groups.items():
        episodes = len(rows)
        valid_rows = [
            row
            for row in rows
            if row["failure_reason"] not in ("api_error", "cost_limit")
        ]
        successes = sum(int(row["success"]) for row in valid_rows)
        key = f"{mode}:{model}"
        data = {
            "mode": mode,
            "model": model,
            "episodes": episodes,
            "evaluated_episodes": len(valid_rows),
            "api_errors": sum(row["failure_reason"] == "api_error" for row in rows),
            "cost_limited_episodes": sum(
                row["failure_reason"] == "cost_limit" for row in rows
            ),
            "successes": successes,
            "success_rate": successes / max(1, len(valid_rows)),
            "average_reward": sum(row["episode_reward"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "average_strokes": sum(row["strokes_used"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "average_decisions": sum(row["decisions"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "total_invalid_responses": sum(row["invalid_responses"] for row in rows),
            "total_latency_seconds": sum(row["latency_seconds"] for row in rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "reported_cost_usd": sum(row["reported_cost_usd"] for row in rows),
        }
        summary["overall"][key] = data
        print(
            f"{mode.upper():3} | {model:45} | "
            f"success={100 * data['success_rate']:6.2f}% | "
            f"reward={data['average_reward']:9.2f} | "
            f"strokes={data['average_strokes']:5.2f} | "
            f"decisions={data['average_decisions']:6.2f} | "
            f"cost=${data['reported_cost_usd']:.6f}"
        )

    for (mode, model, level_id), rows in level_groups.items():
        episodes = len(rows)
        valid_rows = [
            row
            for row in rows
            if row["failure_reason"] not in ("api_error", "cost_limit")
        ]
        successes = sum(int(row["success"]) for row in valid_rows)
        key = f"{mode}:{model}:level_{level_id}"
        summary["by_level"][key] = {
            "mode": mode,
            "model": model,
            "level_id": level_id,
            "episodes": episodes,
            "evaluated_episodes": len(valid_rows),
            "api_errors": sum(row["failure_reason"] == "api_error" for row in rows),
            "successes": successes,
            "success_rate": successes / max(1, len(valid_rows)),
            "average_reward": sum(row["episode_reward"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "average_strokes": sum(row["strokes_used"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "average_decisions": sum(row["decisions"] for row in valid_rows)
            / max(1, len(valid_rows)),
            "reported_cost_usd": sum(row["reported_cost_usd"] for row in rows),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenRouter LLMs and VLMs on sequential puzzles."
    )
    parser.add_argument("--level-path", type=Path, default=ROOT / "levels" / "level.json")
    parser.add_argument("--levels", default="1")
    parser.add_argument("--mode", choices=("llm", "vlm", "both"), default="both")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated models used for both modes unless mode-specific lists are set",
    )
    parser.add_argument("--llm-models", default="")
    parser.add_argument("--vlm-models", default="")
    parser.add_argument("--attempts-per-level", type=int, default=1)
    parser.add_argument("--decision-frames", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--num-stroke-points", type=int, default=2)
    parser.add_argument("--stroke-body", choices=("static", "dynamic"), default="static")
    parser.add_argument("--stroke-penalty", type=float, default=0.25)
    parser.add_argument("--reward-mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--history-limit", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=5.0,
        help="Stop before the next call once completed responses report this spend; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request strict JSON Schema; only use with models that support it",
    )
    parser.add_argument("--referer", default=os.getenv("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--app-title", default="Puzzler V2 Thesis Benchmark")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "openrouter")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the complete local pipeline with synthetic wait actions and no API calls",
    )
    return parser


def resolve_benchmarks(args: argparse.Namespace) -> list[Tuple[str, str]]:
    common = parse_csv_strings(args.models)
    llm_models = parse_csv_strings(args.llm_models) or common
    vlm_models = parse_csv_strings(args.vlm_models) or common
    benchmarks = []
    if args.mode in ("llm", "both"):
        if not llm_models:
            raise ValueError("Provide --models or --llm-models for LLM evaluation")
        benchmarks.extend(("llm", model) for model in llm_models)
    if args.mode in ("vlm", "both"):
        if not vlm_models:
            raise ValueError("Provide --models or --vlm-models for VLM evaluation")
        benchmarks.extend(("vlm", model) for model in vlm_models)
    return benchmarks


def evaluate(args: argparse.Namespace) -> None:
    if args.decision_frames < 1 or args.max_steps < 1 or args.max_decisions < 1:
        raise ValueError("Decision frames, max steps, and max decisions must be positive")
    if args.num_stroke_points < 2:
        raise ValueError("--num-stroke-points must be at least 2")
    if args.max_cost_usd < 0:
        raise ValueError("--max-cost-usd cannot be negative")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    benchmarks = resolve_benchmarks(args)
    levels_data = load_levels(args.level_path)
    levels = [get_level(levels_data, level_id) for level_id in parse_level_ids(args.levels)]
    if not levels:
        raise ValueError("At least one level must be selected")
    if args.attempts_per_level < 1:
        raise ValueError("--attempts-per-level must be at least 1")

    for level in levels:
        effective_stroke_body = level.get("draw_rules", {}).get(
            "stroke_body", args.stroke_body
        )
        if effective_stroke_body != args.stroke_body:
            print(
                f"Level {level['level_id']} overrides --stroke-body: using "
                f"{effective_stroke_body!r} instead of {args.stroke_body!r}"
            )

    decisions_per_episode = min(
        args.max_decisions,
        math.ceil(args.max_steps / args.decision_frames),
    )
    maximum_calls = (
        len(benchmarks)
        * len(levels)
        * args.attempts_per_level
        * decisions_per_episode
    )
    print(
        f"Planned benchmark: {len(benchmarks)} model/mode pairs, "
        f"{len(levels)} levels, up to {maximum_calls} API calls"
    )
    if args.max_cost_usd > 0:
        print(f"Reported-cost stop limit: ${args.max_cost_usd:.2f}")

    if args.dry_run:
        client = DryRunClient()
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        client = OpenRouterClient(
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            structured_output=args.structured_output,
            referer=args.referer,
            title=args.app_title,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episodes_path = args.output_dir / f"episodes_{run_id}.csv"
    traces_path = args.output_dir / f"decisions_{run_id}.jsonl"
    summary_path = args.output_dir / f"summary_{run_id}.json"
    config_path = args.output_dir / f"config_{run_id}.json"

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["benchmarks"] = [
        {"mode": mode, "model": model} for mode, model in benchmarks
    ]
    config["run_id"] = run_id
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    results = []
    budget = BudgetTracker(args.max_cost_usd)
    with traces_path.open("w", encoding="utf-8") as trace_file:
        for mode, model in benchmarks:
            for level in levels:
                for attempt in range(1, args.attempts_per_level + 1):
                    if budget.exhausted:
                        break
                    result = evaluate_episode(
                        client=client,
                        mode=mode,
                        model=model,
                        level=level,
                        level_path=args.level_path,
                        attempt=attempt,
                        args=args,
                        trace_file=trace_file,
                        budget=budget,
                    )
                    results.append(result)
                if budget.exhausted:
                    break
            if budget.exhausted:
                break

    with episodes_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    summary = print_summary(results)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nEpisode results: {episodes_path}")
    print(f"Decision traces: {traces_path}")
    print(f"Summary: {summary_path}")
    print(f"Run configuration: {config_path}")
    print(f"Completed-response spend reported by OpenRouter: ${budget.reported_spend_usd:.6f}")
    if budget.exhausted:
        print("Benchmark stopped because --max-cost-usd was reached.")


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
