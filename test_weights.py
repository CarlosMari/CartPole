import numpy as np
from Qlearning_pole import Qlearning
from tqdm import tqdm
import matplotlib.pyplot as plt


# Number of games the agent will play.
iterations = 50
agent = Qlearning()
# Insert the weights of the Agent to plot.
agent.Q = np.load("Qmatrix.npy")

scores = []
for i in tqdm(range(iterations),miniters=1,desc="Trained Agent"):
    a, b = agent.simulateLearnedStrategy()
    scores.append(np.sum(a))

random_scores = []
for i in tqdm(range(iterations),miniters=1,desc="Random Agent"):
    a, b = agent.simulateRandomStrategy()
    random_scores.append(a)

data = [random_scores,scores]
print(data)

plt.title("Rewards with trained agent")
plt.hist(scores)
plt.xlabel('Reward')
plt.ylabel('Percentage')
# plt.savefig('./resources/new.png')
plt.show()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)


bp = ax.boxplot(data,patch_artist=True,notch=True,vert=0)
plt.title("Trained Agent vs Random Agent (50 episodes)")
# plt.savefig("./resources/old_boxplot.png")
plt.show()
