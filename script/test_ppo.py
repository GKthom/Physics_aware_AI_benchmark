import argparse
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


DEFAULT_EVAL_LEVELS = [1,2,3, 4,5, 6,7,8,9, 10, 11, 12, 13, 14, 15]


def parse_level_ids(value):
    if value is None or value.strip() == "":
        return DEFAULT_EVAL_LEVELS
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--level-path", default=str(ROOT / "levels" / "level.json"))
    parser.add_argument("--levels", default=",".join(map(str, DEFAULT_EVAL_LEVELS)))
    parser.add_argument("--episodes-per-level", type=int, default=3)
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--stroke-body", choices=["static", "dynamic"], default="static")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--agent-draw-mode", choices=["stroke", "line"], default="stroke")
    parser.add_argument("--num-stroke-points", type=int, default=2)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.03)
    return parser


def evaluate(args):
    level_ids = parse_level_ids(args.levels)
    model_path = args.model_path

    if model_path is None:
        model_name = (
            f"ppo_multilevel_{args.reward_mode}_{args.stroke_body}"
            f"_{args.agent_draw_mode}_{args.num_stroke_points}pt"
        )
        model_path = ROOT / "models" / model_name / "best_model.zip"

    model = PPO.load(str(model_path))

    env = BrainItOnGeneralEnv(
        level_path=args.level_path,
        level_id=level_ids[0],
        render_mode="human" if args.render else None,
        reward_mode=args.reward_mode,
        control_mode="agent",
        max_steps=args.max_steps,
        stroke_body=args.stroke_body,
        agent_draw_mode=args.agent_draw_mode,
        num_stroke_points=args.num_stroke_points,
    )

    results = []

    for level_id in level_ids:
        for episode in range(args.episodes_per_level):
            obs, info = env.reset(options={"level_id": level_id})

            terminated = False
            truncated = False
            episode_reward = 0.0
            final_info = info

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, final_info = env.step(action)
                episode_reward += float(reward)

                if args.render:
                    env.render()
                    time.sleep(args.sleep)

            success = bool(final_info.get("is_success", False))
            steps = int(final_info.get("step_count", args.max_steps))
            distance = final_info.get("goal_distance")
            segments = final_info.get("segments_used")
            failure_reason = final_info.get("failure_reason")

            results.append(
                {
                    "level_id": level_id,
                    "success": success,
                    "reward": episode_reward,
                    "steps": steps,
                    "distance": distance,
                    "segments": segments,
                    "failure_reason": failure_reason,
                }
            )

            distance_text = "n/a" if distance is None else f"{distance:.2f}"
            print(
                f"Level {level_id:>2} | Episode {episode + 1:>2} | "
                f"reward={episode_reward:>8.2f} | success={success} | "
                f"steps={steps:>3} | segments={segments} | "
                f"distance={distance_text} | failure={failure_reason}"
            )

    env.close()

    successes = sum(item["success"] for item in results)
    rewards = [item["reward"] for item in results]
    steps = [item["steps"] for item in results]
    distances = [item["distance"] for item in results if item["distance"] is not None]

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Model: {model_path}")
    print(f"Levels: {level_ids}")
    print(f"Episodes: {len(results)}")
    print(f"Success Rate: {(successes / max(1, len(results))) * 100:.2f}%")
    print(f"Average Reward: {sum(rewards) / max(1, len(rewards)):.2f}")
    print(f"Average Steps: {sum(steps) / max(1, len(steps)):.2f}")

    if distances:
        print(f"Average Final Distance: {sum(distances) / len(distances):.2f}")


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
