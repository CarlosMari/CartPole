import gymnasium as gym
from Qlearning_pole import Qlearning
import numpy as np
if __name__ == '__main__':
    env = gym.make('CartPole-v1',render_mode='human')
    q = Qlearning(env)
    q.Q = np.load('Qmatrix.npy')
    q.simulateLearnedStrategy(render=True,env1=env)
