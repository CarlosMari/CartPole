import numpy as np
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
    def __init__(self,env,alpha,gamma,epsilon,numEpisodes,numBins,lowerBounds,upperBounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numEpisodes = numEpisodes
        self.numBins = numBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        self.sumRewardsEpisode = []

        self.Qmatrix = np.random.uniform(0,1,size=(numBins[0],numBins[1],numBins[2],numBins[3],self.actionNumber))

    def returnIndexState(self,state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numBins[1])
        cartAngleBin = np.linspace(self.lowerBounds[2],self.upperBounds[2],self.numBins[2])
        cartAngularVelocityBin = np.linspace(self.lowerBounds[3],self.upperBounds[3],self.numBins[3])

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], cartAngularVelocityBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3],cartAngularVelocityBin)-1,0)

        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])

    def selectAction(self,state,index):
        #First 500 episodes will be random
        if index<500:
            return np.random.choice(self.actionNumber)

        randomNumber = np.random.random()
        if index > 7000:
            self.epsilon = 0.999*self.epsilon

        # If satisfied we are exploring
        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)

        # Else we are being greedy
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)]==np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        for indexEpisode in range(self.numEpisodes):
            rewardsEpisode=[]

            (stateS,_) = self.env.reset()
            stateS = list(stateS)
            print(f'Simulating Episode {indexEpisode}')

            terminalState = False
            while not terminalState:
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS,indexEpisode)

                (stateSprime,reward,terminalState,_,_) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime=list(stateSprime)

                stateSprimeIndex = self.returnIndexState(stateSprime)

                QmaxPrime = np.max(self.Qmatrix[stateSprimeIndex])
                if not terminalState:
                    error = reward+self.gamma*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)] = self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                else:
                    error = reward - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error

                stateS = stateSprime
                print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
                self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    def simulateLearnedStrategy(self):
        import gym
        import time
        env1 = gym.make("CartPole-v1",render_mode='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        # obtained rewards at every time step
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS = np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)] == np.max(
                self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,env1


# Note:
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)

# tested on
# gym==0.26.2
# gym-notices==0.0.8

# gymnasium==0.27.0
# gymnasium-notices==0.0.1

# classical gym
import gym
# instead of gym, import gymnasium
# import gymnasium as gym
import numpy as np
import time


# import the class that implements the Q-Learning algorithm


# env=gym.make('CartPole-v1',render_mode='human')
env = gym.make('CartPole-v1')
(state, _) = env.reset()
# env.render()
# env.close()

# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 15000

# create an object
Q1 = Qlearning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()


# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()

# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()


