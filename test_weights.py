import numpy as np

from Qlearning_pole import Qlearning
import gymnasium as gym
import configparser
#import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
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
agent = Qlearning(env, alpha, gamma, epsilon, number_episodes, numberOfBins, lowerBounds, upperBounds)

iterations = 50

agent.Q = np.load("Qmatrix.npy")
scores = []
for i in tqdm(range(iterations),miniters=1,desc="Trained Agent"):
    a,b = agent.simulateLearnedStrategy()
    scores.append(np.sum(a))

random_scores = []
for i in tqdm(range(iterations),miniters=1,desc="Random Agent"):
    a,b = agent.simulateRandomStrategy()
    random_scores.append(a)

data = [random_scores,scores]
print(data)
"""
plt.title("Rewards with trained agent")
plt.hist(scores)
plt.xlabel('Reward')
plt.ylabel('Percentage')
plt.savefig('./resources/trained_agent.png')
plt.show()
"""


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)


bp = ax.boxplot(data,patch_artist=True,notch=True,vert=0)
plt.title("Trained Agent vs Random Agent (50 episodes)")
plt.savefig("./resources/boxplot.png")
plt.show()