from brainiton_v2 import BrainItOnEnvV2
import pygame

def main():
    llm_env = BrainItOnEnvV2(
        level_id=2,
        render_mode="human",
        reward_mode="dense",
        control_mode="llm",
        max_steps=40,
        line_simulation_steps=10,
        max_drawn_segments=4,
        levels_path="levels/levels.json",
    )
    obs, info = llm_env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = llm_env.step(np.zeros(4, dtype=np.float32))
        done = terminated or truncated
        print(info, reward)
    pygame.time.wait(1500)
    llm_env.close()


if __name__ == "__main__":
    main()