# RL Double Inverted Pendulum

This project implements a custom 2D double inverted pendulum control task using pymunk physics, pygame rendering, and PPO from Stable-Baselines3.

## Project Structure

- configs/config.yaml: Centralized defaults for training, PPO hyperparameters, and evaluation.
- src/env/environment.py: Custom DoublePendulumEnv implementation.
- src/agents/ppo_agent.py: PPO wrapper.
- src/training/train_pipeline.py: End-to-end training pipeline.
- src/evaluation/evaluate_pipeline.py: End-to-end evaluation pipeline.
- src/utils/logger.py: CSV logging utility.
- src/utils/plotting.py: Reward curve plotting utility.
- src/utils/gif_generator.py: GIF generation utility.
- train.py: CLI training entrypoint.
- evaluate.py: CLI evaluation entrypoint.

### Environment Design

The environment is a cart on a horizontal track with two linked poles.

- Physics engine: pymunk.Space with gravity and damping.
- Bodies: one cart body and two pole bodies.
- Constraints:
  - GrooveJoint constrains cart horizontal motion.
  - PivotJoint connects cart to pole 1.
  - PivotJoint connects pole 1 to pole 2.
- Observation vector shape: (6,)
  - cart_x
  - cart_vx
  - pole1_angle
  - pole1_angular_velocity
  - pole2_angle
  - pole2_angular_velocity
- Action vector shape: (1,), continuous in [-1, 1], scaled to force.
- Episode ends when:
  - cart exits track bounds
  - either pole angle crosses failure threshold
  - max step limit is reached

### Reward Function Design

Two reward modes are implemented and selectable via reward_type.

1. Baseline reward

- Formula: cos(theta1) + cos(theta2)
- Purpose: reward upright poles only.

2. Shaped reward

- Upright bonus: cos(theta1) + cos(theta2)
- Center penalty: -0.1 \* abs(cart_x)
- Velocity penalty: -0.01 \* (abs(omega1) + abs(omega2))
- Action penalty: -0.001 \* action^2

Rationale:

- Baseline captures the main objective but can learn slowly.
- Shaped adds dense feedback for stabilization and smoother control.
- This typically improves sample efficiency and policy quality.

### How to Run

1. Local venv setup

- Create and activate virtual environment.
- Install dependencies:
  - pip install -r requirements.txt

2. Train

- Example shaped run:
  - python train.py --timesteps 500000 --reward_type shaped --save_path models/ppo_shaped
- Example baseline run:
  - python train.py --timesteps 500000 --reward_type baseline --save_path models/ppo_baseline

3. Evaluate

- python evaluate.py --model_path models/ppo_shaped.zip --reward_type shaped --episodes 1 --max_steps 300 --gif_path media/agent_final.gif

4. Generate reward comparison plot

- python src/utils/plotting.py --baseline_csv logs/training_metrics_baseline.csv --shaped_csv logs/training_metrics_shaped.csv --output_path reward_comparison.png

5. Run tests

- python tests/test_env.py

6. Docker

- Build:
  - docker compose build
- Train service:
  - docker compose run --rm train python train.py --timesteps 1000 --reward_type shaped --save_path models/docker_train_smoke --log_dir logs
- Evaluate service:
  - docker compose run --rm evaluate python evaluate.py --model_path models/ppo_shaped_300k.zip --reward_type shaped --episodes 1 --max_steps 64 --gif_path media/agent_final.gif

## Outputs

- Logs: logs/training_metrics_baseline.csv and logs/training_metrics_shaped.csv
- Plot: reward_comparison.png
- GIFs: media/agent_initial.gif and media/agent_final.gif
