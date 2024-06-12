import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BUFFER_SIZE = 1000000
BATCH_SIZE = 64
MAX_EPISODES = 700
MAX_STEPS = 200

class Dynamics:
    def __call__(self, state, control):
        return self.drift_dynamics(state) + self.control_matrix(state) @ control
    
    def drift_dynamics(self, state: ArrayLike) -> tuple:
        g = -9.8 # acceleration due to gravity
        m_c = 1.0 # mass of cart
        m = 0.1 # mass of pole
        l = 0.5 # half-pole length
        mu_c = 0.0005 # coefficient of friction of cart on track
        mu_p = 0.000002 # coefficient of friction of pole on cart

        x, theta, x_dot, theta_dot = state

        theta_ddot = (g*jnp.sin(theta) + jnp.cos(theta)*((-m*l*(theta_dot**2)*jnp.sin(theta) + mu_c*(x_dot/jnp.abs(x_dot)))/(m_c+m)) - ((mu_p*theta_dot)/(m*l)))/(l*(4/3 - (m*jnp.cos(theta)*jnp.cos(theta))/(m_c+m)))

        x_ddot = (m*l*((theta_dot**2) * jnp.sin(theta) - theta_ddot*jnp.cos(theta)) - mu_c*(x_dot/jnp.abs(x_dot)))/(m_c + m)

        return jnp.array([x_dot,
                          theta_dot,
                          x_ddot,
                          theta_ddot])

    def control_matrix(self, state: ArrayLike):
        g = -9.8 # acceleration due to gravity
        m_c = 1.0 # mass of cart
        m = 0.1 # mass of pole
        l = 0.5 # half-pole length
        mu_c = 0.0005 # coefficient of friction of cart on track
        mu_p = 0.000002 # coefficient of friction of pole on cart

        x, theta, x_dot, theta_dot = state
        return jnp.array([[0.],
                          [0.],
                          [1/(m_c+m)],
                          [(-1*(jnp.cos(theta))/(m_c+m))/(l*(4/3 - (m*jnp.cos(theta)*jnp.cos(theta))/(m_c+m)))]])

# Global utils for Barrier Function
dynamics = Dynamics()
alpha1 = lambda x: 2 * x
alpha2 = lambda x: 2 * x
ETA = 0.01

def barrier_function(state, r=0.7):
    x, theta, x_dot, theta_dot = state
    b =  r**2 - x**2
    return b

def lie_derivative(func, vector_field_func, state):
    '''
    func: a function that takes in a state and returns a scalar value.
          i.e., func(state) = scalar
    vector_field_func: a function that takes in a state and returns a
                      vector/matrix. i.e., func(state) = vector/matrix
    state: an array describing the state which is the input to func and
          vector_field_func
    '''

    ## put your code here##
    grad_at_state = jax.grad(func)(state)
    vector_at_state = vector_field_func(state) 
    lie_d = grad_at_state@vector_at_state
    return lie_d

def control_constraint_degree_2(h, dynamics, state, class_K_funcs):
    '''
    h: a function that takes in a state and returns a scalar value.
          i.e., h(state) = scalar
    dynamics: the DynamicallyExtendedUnicycle class defined above
    state: an array describing the state which is the input to func and
          vector_field_func
    class_K_funcs: a 2-list of class K function [alpha_func_1, alpha_func_2]

    Compute the coefficients for the CBF/CLF inequality terms, assuming all the terms are moved to the LHS

    Lf2h(z) + LgLfh(z)u + Lfa1h(z) + a2_term

    Returns:
    Lf2h
    LgLfh
    Lfa1h
    a2_term
    '''
    ## put your code here##
    alpha_func_1 = lambda x: class_K_funcs[0](x)
    alpha_func_2 = lambda x: class_K_funcs[1](x)
    Lf2h = lie_derivative(lambda x: lie_derivative(h, dynamics.drift_dynamics, x), dynamics.drift_dynamics, state)
    LgLfh = lie_derivative(lambda x: lie_derivative(h, dynamics.drift_dynamics, x), dynamics.control_matrix, state)
    Lfa1h = lie_derivative(lambda x: alpha_func_1(h(x)), dynamics.drift_dynamics, state)
    a2_term = alpha_func_2(alpha_func_1(h(state)) + lie_derivative(h, dynamics.drift_dynamics, state))
    return Lf2h, LgLfh, Lfa1h, a2_term
    #######################

def get_cbf_val(state, control):
    L2fb, LgLfb, Lfa1b, a2_term = control_constraint_degree_2(barrier_function, dynamics, state, [alpha1, alpha2])
    db = LgLfb
    b = L2fb + Lfa1b + a2_term
    cbf_val = b+db@control
    return cbf_val, b, db


def cbf_controller(state, u_p):
    u_cbf = 0.
    cbf_val, b, db = get_cbf_val(state, u_p)
    if cbf_val <= 0:
        u_cbf = ((-1)*(b+db*u_p)/db) + (ETA/db)
    return u_cbf

# Neural Network Architectures
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).reshape(-1, 1).cuda()

        # Critic loss
        target_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, target_actions)
        target_q = rewards + (1 - dones) * GAMMA * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_actor(self, filename):
        torch.save(self.actor_target.state_dict(), filename)

# Main function to train the DDPG agent
def train_ddpg():
    env = gym.make('InvertedPendulum-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            u_cbf = cbf_controller(state, action)
            next_state, reward, terminated, truncated, info = env.step(action+u_cbf)
            done = terminated or truncated
            agent.add_to_replay_buffer(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Save the target actor model after training
    agent.save_actor("ddpgcbf_actor_model2.pt")

if __name__ == "__main__":
    train_ddpg()
