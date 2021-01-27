import numpy as np
import matplotlib.pyplot as plt


class Bandit:

	def __init__(self, p):

		self.p = p
		self.p_estimate = 0
		self.N = 0


	def pull(self):

		return np.random.random() + self.p # Reward will be gaussian distributed


	def update(self, x):

		self.N += 1
		self.p_estimate = self.p_estimate + (x - self.p_estimate) / self.N



def run_experiment(bandit_probs, eps, num_trials):

	bandits = [Bandit(p) for p in bandit_probs]

	optimal_bandit = np.argmax([b.p for b in bandits])

	print(f">>>>>>>>>>>>>>>> RUNNING EXP FOR EPS : {eps} <<<<<<<<<<<<<<<<<<<<<<<")

	print(f"Optimal bandit : {optimal_bandit}")

	rewards = np.zeros(num_trials)
	num_times_explored = 0
	num_times_exploited = 0
	num_optimal = 0

	for i in range(num_trials):

		# Use epsilon greedy method

		if np.random.random() < eps:

			num_times_explored += 1
			j = np.random.choice(len(bandit_probs))

		else:
			num_times_exploited += 1 
			j = np.argmax([b.p_estimate for b in bandits])

		if j == optimal_bandit:
			num_optimal += 1

		x = bandits[j].pull()

		rewards[i] = x

		bandits[j].update(x)


	for i, b in enumerate(bandits):
		print(f"mean estimate for bandit : {i} is : {b.p}")

	print(f"total rewards obtained : {rewards.sum()}")
	print(f"win rate : {rewards.sum()/num_trials}")
	print(f"percent of times explored : {float(num_times_explored) / num_trials * 100}%")
	print(f"percent of times exploited : {float(num_times_exploited) / num_trials * 100}%")
	print(f"percent of times we hit optimal bandit : {(float(num_optimal) / num_trials) * 100}%")
	print(f"percent of times we hit suboptimal bandit : {(1 - (float(num_optimal) / num_trials)) * 100}%")

	# plot the result

	cumulative_rewards = np.cumsum(rewards)
	cumulative_rewards_average = cumulative_rewards / (np.arange(num_trials) + 1)


	plt.plot(cumulative_rewards_average)

	for p in bandit_probs:
		plt.plot(np.ones(num_trials) * p)

	plt.xscale('log')
	plt.show()
	print("\n")
	
	return cumulative_rewards_average




if __name__ == "__main__":

	BANDIT_PROBS = [1.5, 2.5, 3.5]
	NUM_TRIALS = 100000
	c_01 = run_experiment(BANDIT_PROBS, 0.1, NUM_TRIALS)
	c_005 =run_experiment(BANDIT_PROBS, 0.05, NUM_TRIALS)
	c_001 =run_experiment(BANDIT_PROBS, 0.01, NUM_TRIALS)


	# Log plot
	plt.plot(c_01, label="epsilon = 0.1")
	plt.plot(c_005, label="epsilon = 0.05")
	plt.plot(c_001, label="epsilon = 0.01")
	plt.legend()
	plt.xscale("log")
	plt.show()




	# Linear plot
	plt.plot(c_01, label="epsilon = 0.1")
	plt.plot(c_005, label="epsilon = 0.05")
	plt.plot(c_001, label="epsilon = 0.01")
	plt.legend()
	plt.show()
