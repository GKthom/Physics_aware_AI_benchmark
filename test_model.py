import os
import csv
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from brainiton_v2 import BrainItOnEnvV2


# ------------------------------------------------------------
# Helper function to evaluate one model
# ------------------------------------------------------------
def evaluate_model(
    model_path: str,
    level_id: int,
    reward_mode: str,
    n_episodes: int = 20,
    render: bool = False,
) -> Dict:
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=" * 70)
    print(f"Evaluating model: {model_path}")
    print(f"Level ID: {level_id} | Reward mode: {reward_mode}")
    print("=" * 70)

    env = BrainItOnEnvV2(
        level_id=level_id,
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        control_mode="agent",
        max_steps=80,
        line_simulation_steps=10,
        max_drawn_segments=5,
        gravity=900.0,
        ball_friction=0.95,
        allow_accumulated_drawing=True,
    )

    model = PPO.load(model_path)

    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_success: List[int] = []
    final_goal_distances: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_info = info

        episode_rewards.append(total_reward)
        episode_steps.append(final_info["step_count"])
        episode_success.append(1 if final_info["is_success"] else 0)
        final_goal_distances.append(final_info["goal_distance"])

        print(
            f"Episode {ep+1:02d} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {final_info['step_count']} | "
            f"Success: {final_info['is_success']} | "
            f"Final Goal Distance: {final_info['goal_distance']:.2f}"
        )

    env.close()

    results = {
        "model_path": model_path,
        "level_id": level_id,
        "reward_mode": reward_mode,
        "episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_steps": float(np.mean(episode_steps)),
        "success_rate": float(np.mean(episode_success)),
        "mean_final_goal_distance": float(np.mean(final_goal_distances)),
    }

    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results


# ------------------------------------------------------------
# Save results to CSV
# ------------------------------------------------------------
def save_results_to_csv(results: List[Dict], csv_path: str = "evaluation_results_v2.csv"):
    if not results:
        print("No results to save.")
        return

    fieldnames = list(results[0].keys())

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")


# ------------------------------------------------------------
# Main comparison runner
# ------------------------------------------------------------
def run_comparison():
    results = []

    # Update these paths if needed
    dense_model_path = "models_v2/ppo_brainiton_v2_dense_level1_final"
    sparse_model_path = "models_v2/ppo_brainiton_v2_sparse_level1_final"

    # Evaluate dense model on level 1
    results.append(
        evaluate_model(
            model_path=dense_model_path,
            level_id=1,
            reward_mode="dense",
            n_episodes=20,
            render=False,
        )
    )

    # Evaluate sparse model on level 1
    results.append(
        evaluate_model(
            model_path=sparse_model_path,
            level_id=1,
            reward_mode="sparse",
            n_episodes=20,
            render=False,
        )
    )

    # Optional: cross-test both models on a harder level
    # This shows generalization
    results.append(
        evaluate_model(
            model_path=dense_model_path,
            level_id=2,
            reward_mode="dense",
            n_episodes=20,
            render=False,
        )
    )

    results.append(
        evaluate_model(
            model_path=sparse_model_path,
            level_id=2,
            reward_mode="sparse",
            n_episodes=20,
            render=False,
        )
    )

    save_results_to_csv(results)


if __name__ == "__main__":
    run_comparison()