import numpy as np

class Bandit:

	def __init__(self, p):
		self.p = p
		self.p_estimate = 0
		self.N = 0


	def pull(self):
		return np.random.random() < self.p

	def update(self, x):

		self.N += 1

		self.p_estimate += (x - self.p_estimate) / self.N




