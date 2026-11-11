import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run BrainItOn environment")

    parser.add_argument(
        "--env",
        type=str,
        default="brainiton",
        choices=["brainiton"],
        help="Environment name",
    )

    parser.add_argument(
        "--level_path",
        type=str,
        default="levels/level.json",
        help="Path to levels JSON file",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Level ID to load",
    )

    parser.add_argument(
        "--render",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Turn rendering on or off",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "agent", "llm"],
        help="Control mode",
    )

    parser.add_argument(
        "--reward",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Reward mode",
    )

#     parser.add_argument(
#     "--stroke_body",
#     type=str,
#     default="static",
#     choices=["static", "dynamic"],
#     help="Whether drawn strokes are fixed or affected by gravity",
# )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum episode steps",
    )


    return parser.parse_args()


def run_human_mode(env):
    obs, info = env.reset()

    running = True
    simulate = True
    done = False

    print("Human mode started.")
    print("Drag mouse to draw.")
    print("R = reset | C = clear | N = next | P = previous | SPACE = pause | ESC = quit")

    while running:
        rendered_by_step = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

                elif event.key == pygame.K_c:
                    env.clear_drawn_segments()
                    done = False

                elif event.key == pygame.K_p:
                    env.previous_level()
                    done = False

                elif event.key == pygame.K_n:
                    env.next_level()
                    done = False

                elif event.key == pygame.K_SPACE:
                    simulate = not simulate

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos

                    if my > env.ui_height:
                        env.current_stroke_points = [(mx, my - env.ui_height)]
                        env.preview_stroke_points = [(mx, my - env.ui_height)]

            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0] and env.current_stroke_points:
                    mx, my = event.pos

                    if my > env.ui_height:
                        env.current_stroke_points.append((mx, my - env.ui_height))
                        env.preview_stroke_points = env.current_stroke_points[:]

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and env.current_stroke_points:
                    env._apply_stroke_points(env.current_stroke_points)
                    done = False
                    env.current_stroke_points = []
                    env.preview_stroke_points = []

        if simulate  and not done  and not env.goal_reached:
            # In human mode, step() renders after advancing physics. Remember
            # whether simulation had started so we do not rate-limit the same
            # physics frame a second time below. Before the first stroke,
            # step() intentionally returns without rendering.
            simulation_was_started = env.simulation_started
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            rendered_by_step = (
                simulation_was_started and env.render_mode == "human"
            )

            if terminated or truncated:
                print("Episode ended:", info)

                done = True

        # Keep drawing previews and paused/finished scenes responsive. During
        # active simulation, step() has already rendered exactly once.
        if not rendered_by_step:
            env.render()

    env.close()


def run_agent_mode(env):
    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if env.render_mode == "human":
            env.render()

    print("Episode ended:", info)
    env.close()


def run_llm_mode(env):
    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # placeholder random action for now
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if env.render_mode == "human":
            env.render()

    print("Episode ended:", info)
    env.close()


def main():
    args = parse_args()

    render_mode = "human" if args.render == "on" else None

    if args.env == "brainiton":
        env = BrainItOnGeneralEnv(
            level_path=args.level_path,
            level_id=args.level,
            render_mode=render_mode,
            reward_mode=args.reward,
            control_mode=args.mode,
            max_steps=args.max_steps,
            
        )
    else:
        raise ValueError(f"Unsupported env: {args.env}")

    if args.mode == "human":
        run_human_mode(env)

    elif args.mode == "agent":
        run_agent_mode(env)

    elif args.mode == "llm":
        run_llm_mode(env)


if __name__ == "__main__":
    main()
