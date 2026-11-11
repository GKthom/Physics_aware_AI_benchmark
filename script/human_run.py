import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.brainiton_env_v2 import BrainItOnGeneralEnv


def main():
    env = BrainItOnGeneralEnv(
        level_path="levels/level_boal_goal.json",
        level_id=1,
        render_mode="human",
        reward_mode="dense",
        control_mode="human",
        max_steps=500,
    )

    obs, info = env.reset()

    running = True
    simulate = True

    print("Human mode started.")
    print("Drag mouse to draw.")
    print("R = reset | C = clear drawn lines | SPACE = pause | ESC = quit")

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

                elif event.key == pygame.K_c:
                    env.clear_drawn_segments()

                elif event.key == pygame.K_p:
                    env.previous_level()

                elif event.key == pygame.K_n:
                    env.next_level()

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

                    env.current_stroke_points = []
                    env.preview_stroke_points = []

        if simulate and not env.goal_reached:
            # step() renders active human-mode physics. Avoid rendering the
            # same frame again at the bottom of the loop.
            simulation_was_started = env.simulation_started
            obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            rendered_by_step = (
                simulation_was_started and env.render_mode == "human"
            )

            if terminated or truncated:
                print("Episode ended:", info)

        if not rendered_by_step:
            env.render()

    env.close()


if __name__ == "__main__":
    main()
