import numpy as np
import matplotlib.pyplot as plt

def distance_to_reward(d, alpha=0.006, b=500.0, beta=0.001):
	norm_c = - b * np.log(beta)
	d = np.asarray(d, dtype=float)
	rew = - alpha * d**2 - b * np.log(d**2 + beta)
	return rew / norm_c

def distance_raw(d, alpha=0.006, b=500.0, beta=0.001):
	d = np.asarray(d, dtype=float)
	return - alpha * d**2 - b * np.log(d**2 + beta)

def main():
	xs = np.linspace(0.0, 0.3, 301)
	ys_norm = distance_to_reward(xs)
	ys_raw = distance_raw(xs)
	fig, ax1 = plt.subplots(figsize=(6, 4))
	color1 = 'tab:blue'
	color2 = 'tab:orange'
	l1, = ax1.plot(xs, ys_norm, lw=2, color=color1, label='normalized')
	ax1.set_xlabel('distance')
	ax1.set_ylabel('normalized reward', color=color1)
	ax1.tick_params(axis='y', labelcolor=color1)
	ax2 = ax1.twinx()
	l2, = ax2.plot(xs, ys_raw, lw=1.5, color=color2, label='raw (unnormalized)')
	ax2.set_ylabel('raw reward', color=color2)
	ax2.tick_params(axis='y', labelcolor=color2)
	# combine legends from both axes
	handles = [l1, l2]
	labels = [h.get_label() for h in handles]
	ax1.legend(handles, labels, loc='best')
	plt.title('Distance to Reward (normalized and raw)')
	ax1.grid(True)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

