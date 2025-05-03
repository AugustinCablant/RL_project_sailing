import sys
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
from joblib import dump, load
import xgboost as xgb
import matplotlib.animation as animation
from typing import Dict, Any
from copy import deepcopy
import random 
from sklearn.base import BaseEstimator
from torch import nn
import torch
from torch.distributions.categorical import Categorical
from sklearn.utils import gen_batches

# Add the src directory to the path
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('..'))

# Import the BaseAgent class
from src.agents.base_agent import BaseAgent
from initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS
from src.env_sailing import SailingEnv
from src.test_agent_validity import validate_agent, load_agent_class
from src.evaluation import evaluate_agent, visualize_trajectory

# Environment parameters
env = SailingEnv(**get_initial_windfield('simple_static'))
n_actions = env.action_space.n
d_s = 2054


class FCNet(torch.nn.Module):
  """
  Define the Neural network with the __init__ and forward method.
  It should define a fully connected
  neural network with prescribed input size, hidden size and output size
  """
  def __init__(self, input_size, hidden_size, output_size) -> None:
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
            )

  def forward(self, x):
    return self.linear_relu_stack(x)


class NN(nn.Module):
  """
  A sklearn-like class for the neural net
  """
  def __init__(self, n_iterations, input_size, hidden_size, output_size, alpha, seed, batch_size) -> None:
    """
    Initialize the sklearn class:
    - Record the input parameters using
    self.parameter = parameter
    - Initialize the neural network model and record it in self.model
    - Initialize the Adam optimizer and record it in self.optimizer
    """
    super().__init__()
    self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    self.n_iterations = n_iterations
    self.alpha = alpha
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.seed = seed
    self.batch_size = batch_size
    self.model = FCNet(input_size, hidden_size, output_size).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)


  def partial_fit(self, X, Y):
    """Update parameters
    - Convert numpy input data to torch
    - Define the loss
    - Find the gradient via automatic differentiation
    - Perform a gradient update of the parameters

    Parameters
    ----------
    X: np array of size (n, ds + da)
      state-action pairs
    Y: np array of size (n, 1)
      Estimate of the state action value function
    """
    # DATA
    Xs = torch.from_numpy(X[:, :-1].copy()).float().to(self.device)  # états
    Xa = torch.from_numpy(X[:, -1].copy()).long().to(self.device)    # actions (indices)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y.copy()).float().to(self.device)
    else:
        Y = Y.float().to(self.device)

    # LOSS
    loss_fn = torch.nn.MSELoss()
    Q_all = self.model(Xs)
    Q_pred = Q_all[torch.arange(len(Xs)), Xa]
    loss = loss_fn(Q_pred, Y)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()


  def fit(self, X, Y):
    """Applies n_iterations steps of gradient using function grad_step.
    At each iteration, all samples are shuffled and split into batches of size
    `batch_size`.
    The gradient steps are performed on each batch of data sequentially
    """
    idx_samples = np.arange(len(X))
    for _ in range(self.n_iterations):
      np.random.shuffle(idx_samples)
      for batch_slice in gen_batches(len(X), self.batch_size):
        Xb, Yb = X[idx_samples[batch_slice]].copy(), Y[idx_samples[batch_slice]].copy()
        self.partial_fit(Xb, Yb)

  def predict(self, X):
    """Use the fitted parameter to predict q(s, a)
    - Convert input data to Torch
    - Apply the model
    - Convert the output to numpy

    Parameters
    ---------
    X: np array of shape (n, ds + da)
    Return
    ------
    qSA: np array of shape (n,)
      qSA[i] is an estimate of q(s_i, a_i) computed with the pred function
      where s_i and a_i are given by X[i]
    """
    Xs = X[:, :-1]
    Xa = X[:, -1]
    with torch.no_grad():
      Xs = torch.from_numpy(Xs.copy()).float().to(self.device)
      Xa = torch.from_numpy(Xa.copy()).int().to(self.device)
      y = self.model(Xs)[torch.arange(len(Xs)), Xa]
    return y.cpu().numpy()
  

class DQNAgent(BaseAgent):
    """ DQN Agent """
    
    def __init__(self):
        super().__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.d_s = 2054
        self.n_actions = 9
        self.pi = lambda x: np.random.randint(0, self.n_actions)
        self.capacity = 2000
        self.batch_size = 200
        self.eps = 0.5
        self.gamma = 0.99
        self.C = 20
        self.n_iterations = 1000
        self.nb_gradient_steps = 5  # Number of gradient steps after each iteration
        self.model = NN(n_iterations=1,
                        input_size=2054,
                        hidden_size=24,
                        output_size=env.action_space.n,
                        alpha=0.001,
                        batch_size=None,
                        seed=0)
        self.buffer = np.zeros((self.capacity, self.d_s * 2 + 3))
        self.target_model = deepcopy(self.model)
        self.Qmax = None 
        self.memory = []
        self.criterion = torch.nn.SmoothL1Loss()
        self.reset()

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def gradient_step(self, t):
        if len(self.buffer) > self.batch_size:
            I = np.random.randint(min(t+1, self.capacity), size=self.batch_size)
            data = self.buffer[I]

            # update target network if needed
            if t % self.C == 0:
                self.target_model = deepcopy(self.model)

            Xb = data[:, :len(self.s) + 1]
            Yb = data[:, len(self.s) + 1]

            # Qmax update
            if self.Qmax is None:
                self.model.partial_fit(Xb, Yb)

            self.Qmax = np.max(
                            [self.target_model.predict(np.column_stack([data[:, len(self.s) + 3:], np.ones(len(data)).reshape(-1, 1) * a])) for a in range(n_actions)],
                            axis=0)
            Yb = data[:, len(self.s) + 1] + self.gamma * (1 - data[:, len(self.s) + 2]) * self.Qmax
            Yb = torch.from_numpy(Yb.copy()).to(torch.float).to(self.model.device)
            self.model.partial_fit(Xb, Yb)

    def greedy_action(self, state):
        return np.random.randint(0, self.n_actions)
    
    def definePI(self):
        def pi(s):
            q = [self.model.predict(np.array(self.s.copy().tolist() + [a]).reshape(1,-1))[0] for a in range(self.n_actions)]
            return np.argmax(q)
        return lambda s: pi(s)

    def trainDQN(self, path):
        for initial_windfield_name, initial_windfield in INITIAL_WINDFIELDS.items():
            env = SailingEnv(**get_initial_windfield(initial_windfield_name))
            state, _ = env.reset()
            epsilon = self.eps
            self.s = state.copy()

            for t in tqdm(range(self.n_iterations)):
                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.greedy_action(state)
                # step
                next_state, reward, done, trunc, _ = env.step(action)
                self.buffer[t % self.capacity, :] = np.array(state.copy().tolist() + [action, reward, done] + next_state.copy().tolist()) # [s, a, r, term, s2]

                # train
                for _ in range(self.nb_gradient_steps):   
                    self.gradient_step(t)

                if done or trunc:
                    state, _ = env.reset()
                    self.s = state.copy()
                else:
                    state = next_state
                    self.s = next_state.copy()
        self.save(path)
    
    def act(self, observation: np.ndarray) -> int:
        """ """
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            probs = Categorical(Q)
            action = probs.sample()
        return action
    
    def reset(self) -> None:
        """Reset the agent."""
        self.s0, _ = env.reset()
        self.s = self.s0.copy()
    
    def save(self, path):
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
        else:
            print("No model found to save.")

    def load(self):
        try:
            self.model = NN(n_iterations=1,
                        input_size=2054,
                        hidden_size=24,
                        output_size=env.action_space.n,
                        alpha=0.001,
                        batch_size=None,
                        seed=0).to(self.device)
            self.model.load_state_dict(torch.load("models/DQN", map_location = self.device))
            self.model.eval()
        except:
            print("No saved model found.")
            self.model = None
    
    def seed(self, seed: int = None) -> None:
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)

agentDQN = DQNAgent().load()