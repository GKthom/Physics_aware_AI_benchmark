from brainiton_v2 import BrainItOnEnvV2


def main():
    env = BrainItOnEnvV2(
        level_id=1,
        render_mode="human",
        reward_mode="dense",
        control_mode="human",
        max_steps=300,
        line_simulation_steps=1,   # important for smooth human simulation
        max_drawn_segments=8,
        gravity=900,
        ball_friction=0.01,
    
        allow_accumulated_drawing=True,
        levels_path="levels/level.json",
    )

    env.run_human_mode()


if __name__ == "__main__":
    main()