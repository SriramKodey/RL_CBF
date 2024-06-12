import gym
import torch
import numpy as np
from ddpg import Actor
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

device = ("cuda" if torch.cuda.is_available() else "cpu")

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



def load_policy(path, state_dim, action_dim, max_action):
    actor = Actor(state_dim, action_dim, max_action)
    state_dict = torch.load(path)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor

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

def get_cbf_vals(states, controls):
    dynamics = Dynamics()
    cbf_vals = []
    alpha1 = lambda x: 2 * x
    alpha2 = lambda x: 2 * x
    for i in range(len(controls)):
        L2fb, LgLfb, Lfa1b, a2_term = control_constraint_degree_2(barrier_function, dynamics, states[i], [alpha1, alpha2])
        db = LgLfb
        b = L2fb + Lfa1b + a2_term
        cbf_vals.append(b+db@controls[i])
   
    return cbf_vals


def main(env, policy):
    obs, _ = env.reset()
    env.render()
    done = False
    observations = [obs]
    controls = []
    while not done:
        state = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = policy.forward(state)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        controls.append(action.detach().numpy())
        done = truncated

    env.close()

if __name__== "__main__":
    #env = gym.make('InvertedPendulum-v4')
    env = gym.make('InvertedPendulum-v4', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    path_1 = "ddpg_actor_model.pt"
    actor_1 = load_policy(path_1, state_dim, action_dim, max_action)
    main(env, actor_1)

    path_2 = "ddpgcbf_actor_model.pt"
    actor_2 = load_policy(path_2, state_dim, action_dim, max_action)
    main(env, actor_2)
