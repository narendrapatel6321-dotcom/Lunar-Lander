LunarLander-v3 — Deep Q-Learning from Scratch

This repository contains a from-scratch implementation of Deep Q-Learning (DQN) using PyTorch to solve the LunarLander-v3 environment from Gymnasium.

The purpose of this project is not to showcase a fast solve using modern RL libraries, but to understand, debug, and stabilize DQN training dynamics on a physics-based control task.

Highlights

DQN implemented from scratch (no RL frameworks)

Experience Replay Buffer

Target Network

Double DQN target computation (decoupled action selection and evaluation)

ε-greedy exploration with controlled decay

Explicit separation of training and evaluation

Modular and readable codebase

Reproducible training artifacts (model and plots)

Why LunarLander?

LunarLander is often treated as a “hello-world” environment in reinforcement learning.
However, training DQN reliably on LunarLander is slow and unstable, which makes it useful for:

understanding value overestimation

debugging replay buffer and target updates

studying exploration vs exploitation trade-offs

observing late-stage convergence behavior

This project intentionally avoids PPO/SAC to expose DQN’s limitations and learning dynamics.

Repository Structure

lunar_lander_dqn/
├── agent.py    # DQN / Double DQN logic
├── network.py   # Q-network definition
├── replay_buffer.py # Experience replay buffer
├── helpers.py   # train(), evaluate(), plot_rewards()
├── main.py    # Entry point (train / eval switch)
└── README.md

Algorithm Details

Algorithm: DQN with target network

Target computation: Double DQN

Optimizer: Adam

Learning rate: 5e-4

Batch size: 32

Replay buffer size: 100,000

Discount factor (γ): 0.99

Training frequency: every 4 environment steps

Target network update: every 2000 training steps

Exploration strategy: ε-greedy with per-episode decay

Final ε: 0.05

Training Results

The environment was successfully solved after approximately 9,600 episodes, which is consistent with known DQN behavior on LunarLander (slow but stable convergence).

Training Curve

PLACEHOLDER: training_curve.png
(Insert reward vs episode plot here)

Evaluation

After training, the agent was evaluated using a fully greedy policy (ε = 0) with environment rendering enabled.

Evaluation Video

PLACEHOLDER: evaluation_video.mp4 or GIF
(Insert rendered evaluation run here)

During evaluation, the agent consistently achieved rewards in the 250–320 range with stable landings.

How to Run
Install dependencies

pip install torch gymnasium box2d-py matplotlib

Training mode

In main.py, set:

TRAIN = True

Then run:

python main.py

This will:

train the agent

save the trained model

save the training reward plot

Evaluation mode (no training)

In main.py, set:

TRAIN = False

Then run again:

python main.py

This will:

load the saved model

run evaluation episodes with rendering

no learning occurs

Key Observations

Correct training logic matters more than network architecture

DQN may show little progress for thousands of episodes before converging

Exploration scheduling strongly affects late-stage performance

Double DQN reduces overestimation but does not eliminate instability

Policy-gradient methods (e.g. PPO) solve this task much faster

Limitations

DQN is not sample-efficient for this environment

Training is slow on CPU-only hardware

Results can vary across random seeds

This implementation is intended as a learning and debugging exercise, not a production-ready RL solution.

Possible Extensions

PPO or SAC comparison on the same environment

Robustness tests (gravity or wind variations)

Reward shaping ablation study

Custom control environments

Final Note

This project focuses on understanding reinforcement learning internals, not achieving leaderboard results.
The primary value lies in the debugging process, training stability analysis, and algorithmic insight gained from implementing DQN end-to-end.
