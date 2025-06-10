#tensorflow
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from torch.utils.tensorboard import SummaryWriter

#env
import gym 
import procgen

#others
import numpy as np
import pandas as pd
import os
import wandb
from multiprocessing import Process, Value, Lock


print("Available GPU:", tf.config.list_physical_devices('GPU'))

save_dir = "C:/Users/Acer/Desktop/Autonomous and Adaptive Systems/proj/results"


'''
cose da fare:
- metti if per cambiare nome progetto su wandb
- inserire altri agenrti: random e DQN
- sistemare il codice di DQN e intanto provare il random
- capire come parallelizzare in modo efficente


- run con env paralleli su 4 giochi (portare su colab?)
- salvataggio checkpoint ogni 10 epoche sennò mi sparo
- aggiungere altri tipi di calcolo di reward (almeno 2)

- file a parte per i cambi di reward e i vantaggi

'''
#env_names = ["coinrun", "starpilot", "caveflyer", "jumper", "leaper", "bigfish", "bossfight", "chaser", "climber", "dodgeball", "fruitbot", "heist", "maze", "miner", "ninja", "plunder"]
#envs = [gym.make(f"procgen:procgen-{env_name}-v0") for env_name in env_names]

#Actor Network
class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim):
        super(ActorNetwork, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'),  # (64x64x3) -> (15x15x32)
            tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'),  # -> (6x6x64)
            tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'),  # -> (4x4x64)
            tf.keras.layers.GlobalAveragePooling2D(),  # -> (64,) dimensione gestibile
        ])

        # LSTM: input shape (batch, time, features)
        self.lstm = tf.keras.layers.LSTM(units=512, return_sequences=False, return_state=True)

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(128)
        self.dense3 = tf.keras.layers.Dense(64)
        self.dense4 = tf.keras.layers.Dense(32)

        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, x, states=None, training=False):
        # Input: x shape (batch, H, W, C)
        batch_size = tf.shape(x)[0]

        # CNN encoder
        x = self.encoder(x)  # -> (batch, features)

        # Add time dimension for LSTM (sequence_len = 1)
        x = tf.expand_dims(x, axis=1)  # -> (batch, 1, features)

        # LSTM
        if states is None:
            h0 = tf.zeros((batch_size, 512))
            c0 = tf.zeros((batch_size, 512))
            states = [h0, c0]

        x, h, c = self.lstm(x, initial_state=states)  # -> (batch, 512)

        # Feedforward layers
        x = self.relu(self.dense1(x))
        x = self.dropout(x, training=training)
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))

        # Output
        policy = self.output_layer(x)

        return policy

#Critic Network
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)


#Agent
class PPOAgent:
    def __init__(self, action_dim, actor_lr=1e-4, critic_lr=1e-4, gamma=0.95, clip_ratio=0.2):
        self.actor = ActorNetwork(action_dim)
        self.critic = CriticNetwork()
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
    def get_action(self, state):
        probs = self.actor((tf.convert_to_tensor([state], dtype=tf.float32) / 255.0)).numpy()[0] 
        return np.random.choice(len(probs), p=probs)

    def compute_advantage(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantage = []
        gae = 0
        for delta in reversed(deltas):
            gae = delta + self.gamma * 0.01 * gae
            
            advantage.insert(0, gae)
        return tf.convert_to_tensor(advantage)


    def update(self, states, actions, rewards, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32) / 255.0 #to normalize
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32) / 255.0 #to normalize
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        old_probs = self.actor(states)
        old_probs = tf.gather(old_probs, actions, axis=1, batch_dims=1)
        old_values = tf.squeeze(self.critic(states))
        next_values = tf.squeeze(self.critic(next_states))

        advantages = self.compute_advantage(rewards, old_values, next_values, dones)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        for _ in range(50):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                new_probs = self.actor(states)
                new_probs = tf.gather(new_probs, actions, axis=1, batch_dims=1)
                ratio = new_probs / (old_probs + 1e-10)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = - tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                
                values = tf.squeeze(self.critic(states))
                value_loss = tf.reduce_mean((rewards + self.gamma * next_values * (1 - dones) - values) ** 2)

                #Entropy 
                entropy = tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-10))

                # Total loss
                total_loss = policy_loss + 1 * value_loss - 0.01 * entropy 
            # Apply gradients
            actor_grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            critic_grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        wandb.log({"Total Loss": total_loss, "Value Loss": value_loss, "Policy Loss": policy_loss, "Ratio": ratio, "Clipped Ratio": clipped_ratio, "Advantages": advantages})
            

    def train(self, env, episodes, max_steps=200):
            all_rewards = []
            
            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                states, actions, rewards, next_states, dones = [], [], [], [], []

                for step in range(max_steps):
                    action = self.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                    dones_int = [0 if done == False else 1 for done in dones]
                    state = next_state
                    episode_reward += reward

                    if done:
                        self.update(states, actions, rewards, next_states, dones_int)
                        states, actions, rewards, next_states, dones = [], [], [], [], []


                all_rewards.append(episode_reward)
                print(f"Episode {episode+1}, Reward: {episode_reward}")
                wandb.log({"Episode": episode + 1, "Episode Reward": episode_reward})
            return all_rewards

    def save(self, actor_filepath, critic_filepath):

        print(f"Saving actor model to {actor_filepath}")
        self.actor.save_weights(actor_filepath)
        
        print(f"Saving critic model to {critic_filepath}")
        self.critic.save_weights(critic_filepath)


    def load(self, actor_filepath, critic_filepath):

        print(f"Loading actor model from {actor_filepath}")
        self.actor.load_weights(actor_filepath)
        
        print(f"Loading critic model from {critic_filepath}")
        self.critic.load_weights(critic_filepath)



class Args:
    def __init__(self, num_envs, num_steps, num_minibatches, total_timesteps, env_id, exp_name, seed, cuda, wandb_project_name, wandb_entity, torch_deterministic=True):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.total_timesteps = total_timesteps
        self.env_id = env_id
        self.exp_name = exp_name
        self.seed = seed
        self.cuda = cuda
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.torch_deterministic = torch_deterministic
        
        # Derived attributes
        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // self.batch_size

def evaluate_agent(agent, env_name, num_episodes=200):
    """Evaluate the agent and render its behavior."""
    env = gym.make(f"procgen:procgen-{env_name}-v0", render_mode="human")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = np.expand_dims(state, axis=0) / 255.0
            action = agent.get_action(state[0])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        wandb.log({"Episode": episode + 1, "Total Reward": total_reward})
    env.close()


def setup_logging(args, run_name):
    """Setup logging with wandb and TensorBoard."""
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,  # Nome personalizzato per la run
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )
    return writer


def train_env(env_name, num_iterations, save_dir, best_score, lock):
    args = Args(
            num_envs=1,
            num_steps=2048,
            num_minibatches=64,
            total_timesteps=10000,
            env_id="procgen",
            exp_name="ppo",
            seed=42,
            cuda=True,
            wandb_project_name="ppo",
            wandb_entity="alessandrablasioli"
        )
    run_name = f"{env_name}__{args.exp_name}"
    writer = setup_logging(args, run_name)


    agent = PPOAgent(action_dim=15)

    envs = [
        gym.make(f"procgen:procgen-{env_name}-v0", distribution_mode="easy")
        for _ in range(32)
    ]
    print(f"Training on {env_name}")

    for idx, env in enumerate(envs):
        avg_reward = 0
        for _ in range(num_iterations):
            rewards = agent.train(env, 200)
            avg_reward += sum(rewards) / len(rewards)
        
        avg_reward /= num_iterations
        print(f"Average reward for {env_name}, instance {idx + 1}: {avg_reward}")
        wandb.log({"env_name": env_name, "instance": idx + 1, "avg_reward": avg_reward})

        with lock:
            if avg_reward > best_score.value:
                best_score.value = avg_reward
                best_model_path = os.path.join(save_dir, f"{env_name}_best_model.pth")
                agent.save(actor_filepath=best_model_path, critic_filepath=best_model_path)
                print(f"New best model saved for {env_name} with avg_reward={avg_reward}")
    evaluate_agent(agent, env_name, num_episodes=200)
    wandb.finish()

def main():
    with tf.device('/GPU:0'):
        seed=42
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        actor_path = os.path.join(save_dir, "actor_model.weights.h5")
        critic_path = os.path.join(save_dir, "critic_model.weights.h5")

        print(f"Actor Path: {actor_path}")
        print(f"Critic Path: {critic_path}")

        # env_names = ["caveflyer", "starpilot", "coinrun", "bigfish"]
        env_names = ["starpilot"]
        num_iterations = 10
        best_score = Value('d', float('-inf'))
        lock = Lock()

        processes = []
        for env_name in env_names:
            p = Process(target=train_env, args=(env_name, num_iterations, save_dir, best_score, lock))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    
if __name__ == '__main__':
    main()