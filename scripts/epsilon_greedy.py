import numpy as np
import matplotlib.pyplot as plt


NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:

	def __init__(self, p):

		# p : win rate of the bandit 
		self.p = p
		self.p_estimate = 0 
		self.N = 0 # number of data collected for that bandit so far

	def pull(self):
		# draw a 1 with probability p
		return np.random.random() < self.p


	def update(self, x):
		self.N += 1 
		self.p_estimate = self.p_estimate + (x - self.p_estimate) / self.N


def experiment():
	
	bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

	rewards = np.zeros(NUM_TRIALS)
	num_times_explored = 0
	num_times_exploited = 0
	num_optimal = 0

	optimal_j = np.argmax([b.p for b in bandits])

	print(f"Optimal bandit : {optimal_j}")


	for i in range(NUM_TRIALS):

		# Use epsilon-greedy method

		if np.random.random() < EPS:
			num_times_explored += 1

			j = np.random.choice(len(bandits))

		else:
			num_times_exploited += 1

			j = np.argmax([b.p_estimate for b in bandits])

		if j == optimal_j:
			num_optimal += 1


		x = bandits[j].pull()

		rewards[i] = x

		bandits[j].update(x)


	for i, b in enumerate(bandits):
		print(f"mean estimate for bandit {i} : {b.p_estimate}")


	print(f"total reward earned : {rewards.sum()}")
	print(f"overall win rate : {rewards.sum() / NUM_TRIALS}")
	print(f"num times explored : {num_times_explored}")
	print(f"num times exploited : {num_times_exploited}")
	print(f"num of times we selected the optimal bandit : {num_optimal}")


	# plot the results
	cumulative_rewards = np.cumsum(rewards)
	cum_win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
	plt.plot(cum_win_rates)
	plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
	plt.show()


if __name__ == "__main__":

	experiment()





 