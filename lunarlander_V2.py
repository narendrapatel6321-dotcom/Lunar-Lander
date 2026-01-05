import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import Agent
from helpers import *

TRAIN = False
MODEL_PATH = "dqn_lunarlander_v2.pth"

if __name__ == "__main__":
    train_env = gym.make("LunarLander-v3",enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    eval_env = gym.make("LunarLander-v3", render_mode='human', enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    
    agent = Agent(train_env, lr=5e-4, epsilon_decay=0.0001)

    if TRAIN:
        print("Training mode")

        rewards = train(agent, train_env, episodes=10000,save_path=MODEL_PATH)
        plot_rewards(rewards, save_path="training_curve_model2.png")
        agent.save(path = MODEL_PATH)

    else:
        print("Evaluation mode")

        agent.load(path = MODEL_PATH)
        evaluate(agent, eval_env, episodes=5)

    train_env.close()
    eval_env.close()
