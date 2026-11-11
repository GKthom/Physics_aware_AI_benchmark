import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


def parse_level_ids(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_levels(level_path):
    with open(level_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_level_by_id(levels_data, level_id):
    for level in levels_data.get("levels", []):
        if int(level.get("level_id")) == int(level_id):
            return level
    raise ValueError(f"Level {level_id} not found.")


def compact_level_for_prompt(level_data):
    keys = [
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
    ]
    return {key: level_data[key] for key in keys if key in level_data}


def build_prompt(level_data, num_stroke_points):
    compact_level = compact_level_for_prompt(level_data)

    return f"""
You are solving a Brain-It-On style 2D physics puzzle.

Return one drawn stroke that may solve the level.

Return ONLY valid JSON in this exact format:
{{
  "stroke_points": [[x1, y1], [x2, y2]],
  "reason": "short reason"
}}

Rules:
- Use exactly {num_stroke_points} stroke points.
- Points are pixel coordinates in the game canvas.
- Every point must be inside draw_rules.draw_region.
- The stroke is drawn once, then physics simulation runs.
- Think about gravity, static platforms/walls, dynamic objects, and the goal.
- Do not return markdown or extra text.

Level JSON:
{json.dumps(compact_level, indent=2)}
"""


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON:\n{text}")

    return json.loads(match.group(0))


def ask_openai_for_stroke(level_data, model, num_stroke_points, temperature):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed in this environment. "
            "Install it or use the environment where your LLM script already works."
        ) from exc

    if not os.getenv("OPEN_AI_KEY"):
        raise RuntimeError("OPEN_AI_KEY is not set.")

    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
    prompt = build_prompt(level_data, num_stroke_points)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a physics puzzle solver. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    raw_text = response.choices[0].message.content
    return extract_json(raw_text)


def validate_points(level_data, points):
    draw_region = level_data.get("draw_rules", {}).get("draw_region", {})
    x_min = float(draw_region.get("x_min", 0.0))
    x_max = float(draw_region.get("x_max", level_data.get("canvas", {}).get("width", 900)))
    y_min = float(draw_region.get("y_min", 0.0))
    y_max = float(draw_region.get("y_max", level_data.get("canvas", {}).get("height", 620)))

    if len(points) < 2:
        raise ValueError("Stroke must contain at least two points.")

    clean_points = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid point: {point}")

        x = float(point[0])
        y = float(point[1])

        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            raise ValueError(
                f"Point {(x, y)} is outside draw region "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )

        clean_points.append([x, y])

    return clean_points


def resample_points(points, target_count):
    if len(points) == target_count:
        return points

    if len(points) < 2:
        raise ValueError("Cannot resample a stroke with fewer than two points.")

    segment_lengths = []
    total_length = 0.0

    for index in range(len(points) - 1):
        ax, ay = points[index]
        bx, by = points[index + 1]
        length = float(np.hypot(bx - ax, by - ay))
        segment_lengths.append(length)
        total_length += length

    if total_length <= 1e-6:
        return [points[0] for _ in range(target_count)]

    sampled = []
    for sample_index in range(target_count):
        target_dist = (sample_index / max(1, target_count - 1)) * total_length
        walked = 0.0

        for index, segment_length in enumerate(segment_lengths):
            if walked + segment_length >= target_dist or index == len(segment_lengths) - 1:
                ax, ay = points[index]
                bx, by = points[index + 1]
                alpha = 0.0 if segment_length <= 1e-6 else (target_dist - walked) / segment_length
                sampled.append([ax + alpha * (bx - ax), ay + alpha * (by - ay)])
                break

            walked += segment_length

    return sampled


def points_to_action(points, num_stroke_points):
    points = resample_points(points, num_stroke_points)
    return np.array([value for point in points for value in point], dtype=np.float32)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Ask an LLM to solve levels and visualize strokes.")
    parser.add_argument("--level-path", default=str(ROOT / "levels" / "level.json"))
    parser.add_argument("--levels", default="1")
    parser.add_argument("--attempts-per-level", type=int, default=1)
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--stroke-body", choices=["static", "dynamic"], default="static")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-stroke-points", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--pause-after-episode", type=float, default=3.0)
    return parser


def evaluate(args):
    levels_data = load_levels(args.level_path)
    level_ids = parse_level_ids(args.levels)

    env = BrainItOnGeneralEnv(
        level_path=args.level_path,
        level_id=level_ids[0],
        render_mode="human" if args.render else None,
        reward_mode=args.reward_mode,
        control_mode="agent",
        max_steps=args.max_steps,
        stroke_body=args.stroke_body,
        agent_draw_mode="stroke",
        num_stroke_points=args.num_stroke_points,
    )

    successes = 0
    total_attempts = 0

    try:
        for level_id in level_ids:
            level_data = get_level_by_id(levels_data, level_id)

            for attempt in range(args.attempts_per_level):
                print(f"\nLevel {level_id} | Attempt {attempt + 1}")
                print(level_data.get("title", "Untitled Level"))

                if args.provider == "openai":
                    llm_result = ask_openai_for_stroke(
                        level_data=level_data,
                        model=args.model,
                        num_stroke_points=args.num_stroke_points,
                        temperature=args.temperature,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {args.provider}")

                points = validate_points(level_data, llm_result.get("stroke_points", []))
                action = points_to_action(points, args.num_stroke_points)

                print("Reason:", llm_result.get("reason", ""))
                print("Stroke points:", points)
                print("Action:", action.tolist())

                obs, info = env.reset(options={"level_id": level_id})
                terminated = False
                truncated = False
                episode_reward = 0.0
                final_info = info

                while not (terminated or truncated):
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    episode_reward += float(reward)

                    if args.render:
                        env.render()
                        time.sleep(args.sleep)

                success = bool(final_info.get("is_success", False))
                successes += int(success)
                total_attempts += 1

                distance = final_info.get("goal_distance")
                distance_text = "n/a" if distance is None else f"{distance:.2f}"

                print(
                    f"Result: reward={episode_reward:.2f}, "
                    f"success={success}, "
                    f"steps={final_info.get('step_count')}, "
                    f"segments={final_info.get('segments_used')}, "
                    f"distance={distance_text}, "
                    f"failure={final_info.get('failure_reason')}"
                )

                if args.render and args.pause_after_episode > 0:
                    env.render()
                    time.sleep(args.pause_after_episode)

    finally:
        env.close()

    print("\nLLM Evaluation Summary")
    print("----------------------")
    print(f"Levels: {level_ids}")
    print(f"Attempts: {total_attempts}")
    print(f"Success Rate: {(successes / max(1, total_attempts)) * 100:.2f}%")


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
