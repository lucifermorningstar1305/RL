import numpy as np

class Bandit:

	def __init__(self, p, p_estimate=0, N=0):

		"""
		The constructor for the Bandit class

		Parameters:
		-----------
		:param p: the win rate of a bandit
		:param p_estimate: the sampled estimated win rate of a bandit
		:param N: the number of times that bandit was used
		"""

		self.p = p
		self.p_estimate = p_estimate 
		self.N = N


	def pull(self):
		"""
		Function to return a reward of 1 with probability p
		"""
		return np.random.random() < self.p

	def update(self, x):

		"""
		Function to update the estimated win rate and the number of times the bandit was used

		Parameters:
		----------
		:param x: the reward at a particular period of time.
		"""

		self.N += 1

		self.p_estimate += (x - self.p_estimate) / self.N




