import numpy as np
import gym
import time


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
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins, lower_bounds, upper_bounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_number = env.action_space.n
        self.numEpisodes = num_episodes
        self.numBins = num_bins
        self.lowerBounds = lower_bounds
        self.upperBounds = upper_bounds

        self.sumRewardsEpisode = []

        self.Q = np.random.uniform(0, 1, size=(num_bins[0], num_bins[1], num_bins[2], num_bins[3], self.action_number))

    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numBins[1])
        cartAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numBins[2])
        cartAngularVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numBins[3])

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
        if index > self.numEpisodes * 0.55:
            self.epsilon = 0.999 * self.epsilon

        # If satisfied we are exploring
        if randomNumber < self.epsilon:
            return np.random.choice(self.action_number)

        # Else we are being greedy
        else:
            return np.random.choice(np.where(
                self.Q[self.returnIndexState(state)] == np.max(self.Q[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        for indexEpisode in range(self.numEpisodes):
            rewardsEpisode = []
            (stateS, _) = self.env.reset()
            stateS = list(stateS)
            print(f'Simulating Episode {indexEpisode}')
            terminalState = False
            while not terminalState:
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS, indexEpisode)

                (stateSprime, reward, terminalState, _, _) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime = list(stateSprime)

                stateSprimeIndex = self.returnIndexState(stateSprime)

                QmaxPrime = np.max(self.Q[stateSprimeIndex])
                if not terminalState:
                    error = reward + self.gamma * QmaxPrime - self.Q[stateSIndex + (actionA,)]
                    self.Q[stateSIndex + (actionA,)] = self.Q[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    error = reward - self.Q[stateSIndex + (actionA,)]
                    self.Q[stateSIndex + (actionA,)] = self.Q[stateSIndex + (actionA,)] + self.alpha * error

                stateS = stateSprime


            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))


    def simulateLearnedStrategy(self):
        import gym
        import time
        env1 = gym.make("CartPole-v1", render_mode='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        # obtained rewards at every time step
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            print(timeIndex)
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
        env2.render()
        # number of simulation episodes
        episodeNumber = 100
        # time steps in every episode
        timeSteps = 1000
        # sum of rewards in each episode
        sumRewardsEpisodes = []

        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode = []
            initial_state = env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes, env2


