import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


DEFAULT_TRAIN_LEVELS = [1, 2, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17]
DEFAULT_EVAL_LEVELS = [3, 4, 6, 13, 15, 18, 19, 20]


def parse_level_ids(value):
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_env(
    *,
    level_path,
    start_level_id,
    train_level_ids,
    reward_mode,

    max_steps,
    agent_draw_mode,
    num_stroke_points,
):
    def _init():
        env = BrainItOnGeneralEnv(
            level_path=level_path,
            level_id=start_level_id,
            train_level_ids=train_level_ids,
            render_mode=None,
            reward_mode=reward_mode,
            control_mode="agent",
            max_steps=max_steps,
            agent_draw_mode=agent_draw_mode,
            num_stroke_points=num_stroke_points,
        )
        return Monitor(env, info_keywords=("is_success", "is_failure", "level_id"))

    return _init


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PPO on BrainItOnGeneralEnv.")
    parser.add_argument("--level-path", default=str(ROOT / "levels" / "level.json"))
    parser.add_argument("--train-levels", default=",".join(map(str, DEFAULT_TRAIN_LEVELS)))
    parser.add_argument("--eval-levels", default=",".join(map(str, DEFAULT_EVAL_LEVELS)))
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--agent-draw-mode", choices=["stroke", "line"], default="stroke")
    parser.add_argument("--num-stroke-points", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--model-name", default=None)
    return parser


def train(args):
    train_level_ids = parse_level_ids(args.train_levels) or DEFAULT_TRAIN_LEVELS
    eval_level_ids = parse_level_ids(args.eval_levels) or train_level_ids

    model_name = args.model_name or (
        f"ppo_multilevel_{args.reward_mode}"
        f"_{args.agent_draw_mode}_{args.num_stroke_points}pt"
    )

    model_dir = ROOT / "models" / model_name
    log_dir = ROOT / "logs" / model_name
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_env_check:
        check_env(
            BrainItOnGeneralEnv(
                level_path=args.level_path,
                level_id=train_level_ids[0],
                train_level_ids=train_level_ids,
                render_mode=None,
                reward_mode=args.reward_mode,
                control_mode="agent",
                max_steps=args.max_steps,
                agent_draw_mode=args.agent_draw_mode,
                num_stroke_points=args.num_stroke_points,
            ),
            warn=True,
        )

    train_env = DummyVecEnv(
        [
            make_env(
                level_path=args.level_path,
                start_level_id=train_level_ids[0],
                train_level_ids=train_level_ids,
                reward_mode=args.reward_mode,
                max_steps=args.max_steps,
                agent_draw_mode=args.agent_draw_mode,
                num_stroke_points=args.num_stroke_points,
            )
            for _ in range(args.n_envs)
        ]
    )
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv(
        [
            make_env(
                level_path=args.level_path,
                start_level_id=eval_level_ids[0],
                train_level_ids=eval_level_ids,
                reward_mode=args.reward_mode,
                max_steps=args.max_steps,
                agent_draw_mode=args.agent_draw_mode,
                num_stroke_points=args.num_stroke_points,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

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
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device="auto",
    )

    print(f"Training levels: {train_level_ids}")
    print(f"Evaluation levels: {eval_level_ids}")
    print(f"Saving only best and final models in: {model_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name="tb",
    )

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))

    train_env.close()
    eval_env.close()

    print(f"Best model: {model_dir / 'best_model.zip'}")
    print(f"Final model: {final_model_path}.zip")


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
