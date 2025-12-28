import matplotlib.pyplot as plt
import numpy as np
def train(agent, env, episodes=3000):
    rewards = []

    for ep in range(episodes):
        
        state, _ = env.reset()
        env_steps = 0
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            env_steps += 1
            if env_steps % 4 == 0:
                agent.train_step()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

        if ep % 50 == 0:
            print(
                f"Episode {ep:4d} | "
                f"Reward {total_reward:7.1f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

        if ep % 500 == 0:
            agent.save()

    return rewards


def evaluate(agent, env, episodes=5):
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    agent.online_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"[EVAL] Episode {ep+1} | Reward {total_reward:.1f}")

    agent.epsilon = old_eps
    agent.online_net.train()

def plot_rewards(rewards, window=100, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    smoothed = np.convolve(
        rewards, np.ones(window)/window, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(range(window-1, len(rewards)), smoothed, label="Smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()