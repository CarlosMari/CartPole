# Cartpole
### Introduction
This repository is a project in which I seek to take my first steps in reinforcement learning via OpenAI gym. 
My objective is to compare different reinforcement learning ideas from simple algorithms to more complex ones.

### Objectives
- Achieve a working model in which there is evidence that training increases survival time
- Experiment with different algorithms and compare training time, complexity and achieved score.
- Fine tune the parameters and required number of bins to achieve an optimal training.
- Achive a more consistent strategy.

### Results
So far only the only RL algorithm used has been Q-Learning obtaining the following results.
![plot](./resources/convergence_old.png)
It can be seen that there is an increase in score, with 3 completely different parts. The first part belongs to random inputs, second part to a model that exploresa lot. The third part is when the epsilon starts to decay.
![plot](./resources/histogram_old.png)

When comparing the results of the trained agent (with 20.000 episodes) to the random agent, the difference becomes clear
![plot](./resources/boxplot_old.png)]

As mentioned before the trained agent is still really unconsisted. It is believed that this is due to the randomness of cartpole.

This repository has been heavily inspired in the following:
https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learnin

