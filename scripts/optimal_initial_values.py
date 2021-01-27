import numpy as np
import matplotlib.pyplot as plt


NUM_TRIALS = 10000
BANDIT_PROBS = [0.2, 0.5, 0.75]

class Bandit:

	def __init__(self, p):
		self.p = p
		self.p_estimate = 5
		self.N = 1.


	def pull(self):

		return np.random.random() < self.p

	def update(self, x):
		self.N += 1
		self.p_estimate += (x - self.p_estimate) / self.N



def experiment():

	bandits = [Bandit(p) for p in BANDIT_PROBS]

	optimal_bandit = np.argmax([b.p for b in bandits])

	print(f"The optimal bandit : {optimal_bandit}")

	rewards = np.zeros(NUM_TRIALS)

	for i in range(NUM_TRIALS):

		j = np.argmax([b.p_estimate for b in bandits])

		x = bandits[j].pull()

		rewards[i] = x
		bandits[j].update(x)


	for i, b in enumerate(bandits):
		print(f"Mean estimate for bandit : {i} is : {b.p_estimate}")


	print(f"total reward : {rewards.sum()}")
	print(f"win rate : {rewards.sum()/NUM_TRIALS}")
	print(f"Percentage of times each bandits were selected : {[b.N / NUM_TRIALS * 100 for b in bandits]}")


	cum_rewards = np.cumsum(rewards)
	cum_avg_rewards = cum_rewards / (np.arange(NUM_TRIALS) + 1)

	# plot the results
	plt.ylim([0, 1])
	plt.plot(cum_avg_rewards)
	plt.plot(np.ones(NUM_TRIALS) * np.max([b.p_estimate for b in bandits]))
	# plt.xscale('log')
	plt.show()


if __name__ == "__main__":

	experiment()


