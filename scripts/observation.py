import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":

	random_data = np.random.random(size=10000)

	data = {"0.0-0.05":0, "0.05-0.1":0, "0.1-0.5":0, "0.5-1.0":0}

	for i in random_data:

		if i <= 0.05:
			data["0.0-0.05"] += 1

		elif i > 0.05 and i <= 0.1:
			data["0.05-0.1"] += 1

		elif i > 0.1 and i <= 0.5:
			data["0.1-0.5"] += 1

		else:
			data["0.5-1.0"] += 1


	final_data = {"observations":list(data.keys()), "freq":list(data.values())}

	df = pd.DataFrame(final_data)

	sns.barplot(x="observations", y="freq", data=df)

	plt.show()









