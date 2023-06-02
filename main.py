
import gymnasium as gym
import numpy
import numpy as np
from Qlearning_pole import Qlearning
import os

# Rendering the environment
# env=gym.make('CartPole-v1',render_mode='human')

# Non rendering
import configparser
#import gym

# Load parameters from config file
config = configparser.ConfigParser()
config.read('config.ini')

cart_velocity_min = float(config['Parameters']['cart_velocity_min'])
cart_velocity_max = float(config['Parameters']['cart_velocity_max'])
pole_angle_velocity_min = float(config['Parameters']['pole_angle_velocity_min'])
pole_angle_velocity_max = float(config['Parameters']['pole_angle_velocity_max'])
number_of_bins_position = int(config['Parameters']['number_of_bins_position'])
number_of_bins_velocity = int(config['Parameters']['number_of_bins_velocity'])
number_of_bins_angle = int(config['Parameters']['number_of_bins_angle'])
number_of_bins_angle_velocity = int(config['Parameters']['number_of_bins_angle_velocity'])
alpha = float(config['Parameters']['alpha'])
gamma = float(config['Parameters']['gamma'])
epsilon = float(config['Parameters']['epsilon'])
number_episodes = int(config['Parameters']['number_episodes'])

# Create the environment
env = gym.make('CartPole-v1')
(state, _) = env.reset()

# Update the observation space bounds
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
upperBounds[1] = cart_velocity_max
upperBounds[3] = pole_angle_velocity_max
lowerBounds[1] = cart_velocity_min
lowerBounds[3] = pole_angle_velocity_min

# Update the number of bins
numberOfBins = [number_of_bins_position, number_of_bins_velocity, number_of_bins_angle, number_of_bins_angle_velocity]




# create an object
Q1 = Qlearning(env, alpha, gamma, epsilon, number_episodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)
import matplotlib.pyplot as plt
# now simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
numpy.save("Qmatrix.npy",Q1.Q)
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.savefig('convergence.png')
plt.title("Convergence of rewards")
plt.show()


# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
obtainedRewardsRandom = []
for i in range(50):
    (rewardsRandom, env2) = Q1.simulateRandomStrategy()
    obtainedRewardsRandom.append(rewardsRandom)
plt.title("Rewards with random strategy")
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()