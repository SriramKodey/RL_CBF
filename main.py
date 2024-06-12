import numpy as np
import gym
import torch
import torch.nn as nn

import model
from ddpg import DDPG
from model import Critic
from model import Actor
from replay_buffer import Memory

env = gym.make('InvertedPendulum-v4', render_mode='human')

# Initialize the environment
observation = env.reset()

# Begin the rendering loop
done = False
while not done:
    # Render the current state of the environment
    env.render()

    # Take a random action (just an example, replace with your agent's action)
    action = env.action_space.sample()

    # Execute the action and observe the next state
    observation, reward, terminated, truncated, info = env.step(action)

    done = truncated

# Close the rendering
env.close()

def train():
    agent = DDPG(4, 1, 20)

def main():
    env = gym.make('InvertedPendulum-v4')

    n = 4 # State Pace dimensions
    m = 1 # Action space dimensions 

    num_iters = 10000
    actor_batch_size = 100
    critic_batch_size = 100

    # Initialize networks
    actor = Actor(n, m)
    actor_target = Actor(n, m)
    critic = Critic(n+m, m)
    critic_target = Critic(n+m, m)



if __name__ == "__main__":
    print(f"Using device : {model.device}")
    main()

