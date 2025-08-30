import os
import random
import numpy as np
import tensorflow as tf
import wandb
import gym
import procgen
#seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self, _=None):
        return np.random.choice(self.action_dim)

    def save(self, *args, **kwargs):
        print("Nothing to save")

    def load(self, *args, **kwargs):
        print("Nothing to load")
def collect_rollout(env, agent, horizon=256):
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    obs = env.reset()
    env.action_space.seed(SEED)
    episode_reward, episode_length = 0, 0

    finished_rewards = []
    finished_lengths = []

    for _ in range(horizon):
        action = agent.get_action()
        next_obs, reward, done, info = env.step(action)

        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(done)

        obs = next_obs
        episode_reward += reward
        episode_length += 1

        if done:
            finished_rewards.append(episode_reward)
            finished_lengths.append(episode_length)
            wandb.log({
                "episode_reward": episode_reward,
                "episode_length": episode_length
            })
            obs = env.reset()
            episode_reward, episode_length = 0, 0
    if finished_rewards:
        wandb.log({
            "episode_reward_mean": np.mean(finished_rewards),
            "episode_length_mean": np.mean(finished_lengths),
            "episode_reward_max": np.max(finished_rewards),
            "episode_reward_min": np.min(finished_rewards),
            "num_episodes": len(finished_rewards)
        })

    return np.array(obs_buf), np.array(act_buf), np.array(rew_buf), np.array(done_buf)

def evaluate(agent, env_name, num_episodes=10, render=False):
    eval_env = gym.make(
        f"procgen:procgen-{env_name}-v0",
        num_levels=500, #change in 200
        start_level=500,
        distribution_mode="hard", #change in easy
        render_mode='rgb_array' if render else None
    )
    eval_env.action_space.seed(SEED+100)
    rewards = []

    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.get_action()
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward
            if render:
                _ = eval_env.render()
        rewards.append(ep_reward)

    eval_env.close()
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    wandb_project_name = "random-procgen-500" #change in  "random-procgen-200"
    wandb_entity = "alessandrablasioli"
    exp_name = "random_baseline"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    games = ['starpilot','caveflyer', 'coinrun', 'bigfish']
    for instance_id, env_name in enumerate(games):
        run_name = f"{env_name}_instance_{instance_id}__{exp_name}"

        env = gym.make(
            f"procgen:procgen-{env_name}-v0",
            num_levels=500, #change in 200
            start_level=0,
            distribution_mode="hard" #change in easy
        )
        env.action_space.seed(SEED)

        num_actions = env.action_space.n
        agent = RandomAgent(num_actions)

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=run_name,
            config={"env_name": env_name, "instance_id": instance_id, "num_iterations":1000, "exp_name":exp_name},
            reinit=True
        )

        for update in range(1000):
            collect_rollout(env, agent, horizon=256)
            wandb.log({"update": update})

            if update % 10 == 0:
                mean_r, std_r = evaluate(agent, env_name, num_episodes=10, render=False)
                wandb.log({
                    "eval_reward_mean": mean_r,
                    "eval_reward_std": std_r,
                    "update": update
                })
                print(f"[EVAL] {env_name} - Update {update} - Reward: {mean_r:.2f} ± {std_r:.2f}")

        wandb.finish()