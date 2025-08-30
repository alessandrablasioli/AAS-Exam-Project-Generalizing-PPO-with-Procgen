import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import wandb
import gym
import procgen

#seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

def actor_network(num_actions):
    return keras.Sequential([
        keras.layers.Conv2D(32, 8, strides=4, activation="relu"),
        keras.layers.Conv2D(64, 4, strides=2, activation="relu"),
        keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(num_actions, activation="softmax")
    ])

def critic_network():
    return keras.Sequential([
        keras.layers.Conv2D(32, 8, strides=4, activation="relu"),
        keras.layers.Conv2D(64, 4, strides=2, activation="relu"),
        keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1)
    ])

class PPOAgent:
    def __init__(self, num_actions, clip_ratio=0.2, gamma=0.99, lam=0.95, actor_lr=2.5e-4, critic_lr=2.5e-4):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.actor = actor_network(num_actions)
        self.critic = critic_network()
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)

        #GAE
    def compute_advantage(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        adv = np.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def update(self, states, actions, returns, advantages, old_logp, ppo_epochs=4, minibatch_size=2048, entropy_coef=0.02, value_coef=1.0):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)
        old_logp = old_logp.astype(np.float32)
        actions = actions.astype(np.int32)

        n_samples = len(states)

        for _ in range(ppo_epochs):
            idxs = np.arange(n_samples)
            np.random.shuffle(idxs)
            for start in range(0, n_samples, minibatch_size):
                mb_idx = idxs[start:start+minibatch_size]
                s_mb = tf.gather(states, mb_idx)
                a_mb = tf.gather(actions, mb_idx)
                adv_mb = tf.gather(advantages, mb_idx)
                ret_mb = tf.gather(returns, mb_idx)
                old_logp_mb = tf.gather(old_logp, mb_idx)

                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    probs = self.actor(s_mb, training=True)
                    dist = tfp.distributions.Categorical(probs=probs)
                    new_logp = dist.log_prob(a_mb)
                    ratio = tf.exp(new_logp - old_logp_mb)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_mb, clipped_ratio * adv_mb))

                    values = tf.squeeze(self.critic(s_mb, training=True), axis=-1)
                    value_loss = value_coef * tf.reduce_mean((ret_mb - values) ** 2)
                    entropy = tf.reduce_mean(dist.entropy())
                    total_loss = policy_loss + value_loss - entropy_coef * entropy

                actor_grads = tape1.gradient(total_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(value_loss, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        wandb.log({
            "policy_loss": policy_loss.numpy(),
            "value_loss": value_loss.numpy(),
            "entropy": entropy.numpy(),
        })

    def load(self, actor_path, critic_path):
        dummy_input = tf.random.normal((1, 64, 64, 3))
        _ = self.actor(dummy_input)
        _ = self.critic(dummy_input)
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

def collect_rollout(env, agent, horizon=256):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    env.action_space.seed(SEED)
    episode_reward, episode_length = 0, 0

    finished_rewards = []
    finished_lengths = []

    for _ in range(horizon):
        obs_norm = obs.astype(np.float32)/255.0
        obs_input = np.expand_dims(obs_norm, axis=0)
        probs = agent.actor(obs_input)
        dist = tfp.distributions.Categorical(probs=probs)
        action = dist.sample().numpy()[0]
        logp = dist.log_prob(action).numpy()
        value = agent.critic(obs_input).numpy()[0,0]

        next_obs, reward, done, info = env.step(action)
        obs_buf.append(obs_norm)
        act_buf.append(action)
        rew_buf.append(reward)
        val_buf.append(value)
        logp_buf.append(logp)
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

    obs_buf = np.array(obs_buf)
    act_buf = np.array(act_buf)
    rew_buf = np.array(rew_buf)
    val_buf = np.array(val_buf)
    logp_buf = np.array(logp_buf)
    done_buf = np.array(done_buf)

    next_val = agent.critic(np.expand_dims(obs.astype(np.float32)/255.0, axis=0)).numpy()[0,0]
    next_values = np.roll(val_buf, -1)
    next_values[-1] = next_val

    adv = agent.compute_advantage(rew_buf, val_buf, next_values, done_buf)
    returns = adv + val_buf
    return obs_buf, act_buf, returns, adv, logp_buf

def evaluate(agent, env_name, num_episodes=10, render=False):
    eval_env = gym.make(
        f"procgen:procgen-{env_name}-v0",
        num_levels=200, #change to 500
        start_level=500,
        distribution_mode="easy", #change in hard
        render_mode='rgb_array' if render else None
    )
    eval_env.action_space.seed(SEED+100)
    rewards = []

    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            obs_norm = obs.astype(np.float32)/255.0
            probs = agent.actor(np.expand_dims(obs_norm, axis=0))
            action = tf.argmax(probs, axis=-1).numpy()[0]
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward
            if render:
                img = eval_env.render()
        rewards.append(ep_reward)
    eval_env.close()
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    wandb_project_name = "ppo-procgen-final" #change in ppo-procgen-final-500
    wandb_entity = "alessandrablasioli"
    exp_name = "baseline"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    games = ['starpilot','caveflyer', 'coinrun', 'bigfish']
    for instance_id, env_name in enumerate(games):
        run_name = f"{env_name}_instance_{instance_id}__{exp_name}"

        env = gym.make(
            f"procgen:procgen-{env_name}-v0",
            num_levels=200, #change in 500
            start_level=0,
            distribution_mode="easy" #change in hard
        )
        env.action_space.seed(SEED)

        num_actions = env.action_space.n
        agent = PPOAgent(num_actions)

        best_actor_path = os.path.join(save_dir, f"{env_name}_best_actor.h5")
        best_critic_path = os.path.join(save_dir, f"{env_name}_best_critic.h5")
        if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
            agent.load(best_actor_path, best_critic_path)


        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=run_name,
            config={"env_name": env_name, "instance_id": instance_id, "num_iterations":1000, "exp_name":exp_name},
            reinit=True
        )

        for update in range(1000):
            states, actions, returns, adv, old_logp = collect_rollout(env, agent, horizon=256)
            agent.update(states, actions, returns, adv, old_logp)
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