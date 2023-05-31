import gym
import random
import numpy as np

env = gym.make("CartPole-v0", render_mode="human")
# As the observation space is not discrete we need to modyfy
# S = (x,dx,Theta,dTheta)

# We divide the x into 4 segments
# Soruce: https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/



Q = np.zeros((env.observation_space.shape[0], env.action_space.n))

# We define the Q learning hyperparameters

alpha = 0.7  # learning rate
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

episodes = 2000

training_rewards = []
epsilons = []

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()

        # Bellman equation
        new_state, reward, terminated, truncated, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + discount_factor *
                                                       np.max(Q[new_state, :])) - Q[state, action]

        score += reward
        state = new_state

        done = truncated or terminated

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    training_rewards.append(score)
    epsilons.append(epsilon)
    print(f'Episode: {episode} Score: {score}')

env.close()
