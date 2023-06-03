import random

import numpy as np
import gym
import time
from tqdm import tqdm
import configparser
class Qlearning:
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS:
    # env - Cart Pole environment
    # alpha - step size
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes

    # numberOfBins - this is a 4 dimensional list that defines the number of grid points
    # for state discretization
    # that is, this list contains number of bins for every state entry,
    # we have 4 entries, that is,
    # discretization for cart position, cart velocity, pole angle, and pole angular velocity

    # lowerBounds - lower bounds (limits) for discretization, list with 4 entries:
    # lower bounds on cart position, cart velocity, pole angle, and pole angular velocity

    # upperBounds - upper bounds (limits) for discretization, list with 4 entries:
    # upper bounds on cart position, cart velocity, pole angle, and pole angular velocity
    def __init__(self, env = gym.make('CartPole-v1'), file='config.ini'):
        self.env = env
        self.load_values(file)



    def load_values(self,file):
        config = configparser.ConfigParser()
        config.read(file)

        cart_velocity_min = float(config['Parameters']['cart_velocity_min'])
        cart_velocity_max = float(config['Parameters']['cart_velocity_max'])
        pole_angle_velocity_min = float(config['Parameters']['pole_angle_velocity_min'])
        pole_angle_velocity_max = float(config['Parameters']['pole_angle_velocity_max'])
        number_of_bins_position = int(config['Parameters']['number_of_bins_position'])
        number_of_bins_velocity = int(config['Parameters']['number_of_bins_velocity'])
        number_of_bins_angle = int(config['Parameters']['number_of_bins_angle'])
        number_of_bins_angle_velocity = int(config['Parameters']['number_of_bins_angle_velocity'])
        self.action_number = self.env.action_space.n
        self.alpha = float(config['Parameters']['alpha'])
        self.gamma = float(config['Parameters']['gamma'])
        self.epsilon = float(config['Parameters']['epsilon'])
        self.numEpisodes = int(config['Parameters']['number_episodes'])

        self.upperBounds = self.env.observation_space.high
        self.lowerBounds = self.env.observation_space.low
        self.upperBounds[1] = cart_velocity_max
        self.upperBounds[3] = pole_angle_velocity_max
        self.lowerBounds[1] = cart_velocity_min
        self.lowerBounds[3] = pole_angle_velocity_min

        self.batch_size = int(config['Parameters']['batch_size'])

        self.rewardsEpisode = 0
        self.sumRewardsEpisode = []

        # Update the number of bins
        self.num_bins = [number_of_bins_position, number_of_bins_velocity, number_of_bins_angle,
                         number_of_bins_angle_velocity]

        self.replayBuffer = []
        self.Q = np.random.uniform(0, 1, size=(self.num_bins[0], self.num_bins[1], self.num_bins[2], self.num_bins[3], self.action_number))

    # Observation space is not discrete so we make it discrete
    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.num_bins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.num_bins[1])
        cartAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.num_bins[2])
        cartAngularVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.num_bins[3])

        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, cartAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, cartAngularVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def selectAction(self, state, index):
        # First 10% episodes will be random
        if index < self.numEpisodes * 0.1:
            return np.random.choice(self.action_number)

        # We generate a random number to decide if we are exploring or not.
        randomNumber = np.random.random()

        # Decay starts at 55%
        if index > self.numEpisodes * 0.6:
            self.epsilon = 0.999 * self.epsilon

        # If satisfied we are exploring
        if randomNumber < self.epsilon:
            return np.random.choice(self.action_number)

        # Else we are being greedy
        else:
            return np.random.choice(np.where(
                self.Q[self.returnIndexState(state)] == np.max(self.Q[self.returnIndexState(state)]))[0])

    def train(self):
        for indexEpisode in tqdm(range(self.numEpisodes)):#, miniters=1):
        #for indexEpisode in range(self.numEpisodes):
            rewardsEpisode = []
            (stateS, _) = self.env.reset()
            stateS = list(stateS)
            #print(f'Simulating Episode {indexEpisode}')
            terminalState = False
            steps = 0
            # Add a steps limiter to shorten training time
            while not terminalState and steps < 2000:
                steps += 1
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS, indexEpisode)

                (stateSprime, reward, terminalState, _, _) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime = list(stateSprime)

                # Store the experience in the buffer
                self.replayBuffer.append([stateS,actionA,reward,stateSprime,terminalState])

                stateSprimeIndex = self.returnIndexState(stateSprime)

                QmaxPrime = np.max(self.Q[stateSprimeIndex])
                if not terminalState:
                    error = reward + self.gamma * QmaxPrime - self.Q[stateSIndex + (actionA,)]
                    self.Q[stateSIndex + (actionA,)] = self.Q[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    error = reward - self.Q[stateSIndex + (actionA,)]
                    self.Q[stateSIndex + (actionA,)] = self.Q[stateSIndex + (actionA,)] + self.alpha * error

                stateS = stateSprime

            if indexEpisode % 5 == 0:
                self.updateQValues()
            #print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))


    def updateQValues(self):
        if len(self.replayBuffer)<self.batch_size:
            return

        # Select a random batch of experiences
        batch = random.sample(self.replayBuffer, self.batch_size)

        for experience in batch:
            state,action,reward,next_state,done = experience
            stateIndex = self.returnIndexState(state)
            actionIndex = action

            if not done:
                next_stateIndex = self.returnIndexState(next_state)
                QmaxPrime = np.max(self.Q[next_stateIndex])
                error = reward + self.gamma * QmaxPrime - self.Q[stateIndex + (actionIndex,)]
            else:
                error = reward - self.Q[stateIndex + (actionIndex,)]
            self.Q[stateIndex + (actionIndex,)] += self.alpha * error

    def simulateLearnedStrategy(self,env1 = gym.make("CartPole-v1"), render=False):
        import gym
        import time
        # Choose this line if you want to see how it behaves
        #env1 = gym.make("CartPole-v1", render_mode='human')
        (currentState, _) = env1.reset()
        if render:
            env1.render()
        timeSteps = 3000
        steps = 0
        # obtained rewards at every time step
        obtainedRewards = []
        terminated = False
        truncated = False
        while (not (terminated or truncated)) or steps < timeSteps:
            steps+=1
            #print(timeIndex)
            # select greedy actions
            actionInStateS = np.random.choice(np.where(self.Q[self.returnIndexState(currentState)] == np.max(
                self.Q[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards, env1

    def simulateRandomStrategy(self):
        env2 = gym.make('CartPole-v1')
        (currentState, _) = env2.reset()
        #env2.render()
        # number of simulation episodes
        episodeNumber = 100
        # time steps in every episode
        timeSteps = 1000
        # sum of rewards in each episode
        rewardsEpisode = []


        for timeIndex in range(timeSteps):
            random_action = env2.action_space.sample()
            observation, reward, terminated, truncated, info = env2.step(random_action)
            rewardsEpisode.append(reward)
            if (terminated):
                break

        return np.sum(rewardsEpisode), env2


