import os
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from brainiton_v2 import BrainItOnEnvV2


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODEL_DIR = "models_v2"
LOG_DIR = "logs_v2"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------
def make_env(
    level_id: int = 1,
    reward_mode: str = "dense",
    control_mode: str = "agent",
    max_steps: int = 80,
    line_simulation_steps: int = 10,
    max_drawn_segments: int = 5,
    gravity: float = 900.0,
    ball_friction: float = 0.95,
    allow_accumulated_drawing: bool = True,
    render_mode=None,
) -> Callable[[], gym.Env]:
    def _init():
        env = BrainItOnEnvV2(
            level_id=level_id,
            render_mode=render_mode,
            reward_mode=reward_mode,
            control_mode=control_mode,
            max_steps=max_steps,
            line_simulation_steps=line_simulation_steps,
            max_drawn_segments=max_drawn_segments,
            gravity=gravity,
            ball_friction=ball_friction,
            allow_accumulated_drawing=allow_accumulated_drawing,
        )
        env = Monitor(env)
        return env

    return _init


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------
def train_ppo(
    level_id: int = 1,
    reward_mode: str = "dense",
    total_timesteps: int = 100_000,
    model_name: str = "ppo_brainiton_v2",
):
    print("=" * 60)
    print("Starting PPO training")
    print(f"Level ID: {level_id}")
    print(f"Reward mode: {reward_mode}")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 60)

    # Training environment
    train_env = DummyVecEnv(
        [
            make_env(
                level_id=level_id,
                reward_mode=reward_mode,
                control_mode="agent",
                max_steps=80,
                line_simulation_steps=10,
                max_drawn_segments=5,
                gravity=900.0,
                ball_friction=0.95,
                allow_accumulated_drawing=True,
                render_mode=None,
            )
        ]
    )
    train_env = VecMonitor(train_env)

    # Evaluation environment
    eval_env = DummyVecEnv(
        [
            make_env(
                level_id=level_id,
                reward_mode=reward_mode,
                control_mode="agent",
                max_steps=80,
                line_simulation_steps=10,
                max_drawn_segments=5,
                gravity=900.0,
                ball_friction=0.95,
                allow_accumulated_drawing=True,
                render_mode=None,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix=f"{model_name}_{reward_mode}_level{level_id}",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_model_path = os.path.join(
        MODEL_DIR, f"{model_name}_{reward_mode}_level{level_id}_final"
    )
    model.save(final_model_path)

    print(f"\nTraining complete.")
    print(f"Final model saved at: {final_model_path}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Train dense reward model
    train_ppo(
        level_id=1,
        reward_mode="dense",
        total_timesteps=100_000,
        model_name="ppo_brainiton_v2",
    )

    # Train sparse reward model
    train_ppo(
        level_id=1,
        reward_mode="sparse",
        total_timesteps=100_000,
        model_name="ppo_brainiton_v2",
    )