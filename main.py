
import gymnasium as gym
import numpy
import numpy as np
from Qlearning_pole import Qlearning
import os

# Rendering the environment
# env=gym.make('CartPole-v1',render_mode='human')


Q1 = Qlearning()
# run the Q-Learning algorithm
Q1.train()
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
numpy.save("Qmatrix_new.npy",Q1.Q)
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