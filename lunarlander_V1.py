import gymnasium as gym
from agent import Agent
from helpers import *

TRAIN = False
MODEL_PATH = "dqn_lunarlander.pth"

if __name__ == "__main__":
    train_env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3", render_mode="human")

    agent = Agent(train_env, lr=5e-4, epsilon_decay=0.0001)

    if TRAIN:
        print("Training mode")

        rewards = train(agent, train_env, episodes=10000, save_path=MODEL_PATH)
        plot_rewards(rewards, save_path="training_curve.png")
        agent.save(MODEL_PATH)

    else:
        print("Evaluation mode")

        agent.load(MODEL_PATH)
        evaluate(agent, eval_env, episodes=5)

    train_env.close()
    eval_env.close()
