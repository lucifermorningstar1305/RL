import numpy as np
import matplotlib.pyplot as plt


NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2 , 0.5, 0.75]


class Bandit:

	def __init__(self, p):

		# p : the win rate of the bandit
		self.p = p
		self.p_estimate = 0
		self.N = 0 # the number of times the bandit was used


	def pull(self):

		return np.random.random() < self.p

	def update(self, x):

		self.N += 1
		self.p_estimate = self.p_estimate + (x - self.p_estimate) / self.N



def experiment():

	bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
	optimal_bandit = np.argmax([b.p for b in bandits])

	print(f"Optimal bandit : {optimal_bandit}")

	rewards = np.zeros(NUM_TRIALS)
	num_times_explored = 0
	num_times_exploited = 0
	num_times_optimal = 0

	eps = EPS

	for i in range(NUM_TRIALS):

		# Use epsilon greedy approach

		if np.random.random() < eps:

			num_times_explored += 1	
			j = np.random.choice(len(BANDIT_PROBABILITIES))


		else:

			num_times_exploited += 1
			j = np.argmax([b.p_estimate for b in bandits])


		if j == optimal_bandit:
			num_times_optimal += 1


		x = bandits[j].pull()

		rewards[i] = x

		bandits[j].update(x)

		eps = 1 / np.log(i + 1e-10) # Epsilon decay


	for i, b in enumerate(bandits):

		print(f"Mean estimate for bandit : {i} is : {b.p_estimate} ")

	print(f"total rewards : {rewards.sum()}")
	print(f"overall win rate : {rewards.sum() / NUM_TRIALS}")
	print(f"num of times explored : {num_times_explored}")
	print(f"num of times exploited : {num_times_exploited}")
	print(f"num of times optimal bandit was selected: {num_times_optimal}")


	# plot the results
	cumulative_rewards = np.cumsum(rewards)
	cum_win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
	plt.plot(cum_win_rates)
	plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
	plt.show()



if __name__ == "__main__":

	experiment()
