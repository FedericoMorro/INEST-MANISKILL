import numpy as np

import matplotlib.pyplot as plt

# Parameters
HORIZON = 100
discount_values = [0.9, 0.91, 0.93, 0.95, 0.97, 0.99, 0.995]

# Create figure
plt.figure(figsize=(10, 6))

# Plot discount decay for each discount value
for discount in discount_values:
    steps = np.arange(HORIZON)
    decay = discount ** steps
    plt.plot(steps, decay, label=f'discount={discount}', linewidth=2)

# Customize plot
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Discount Factor Value', fontsize=12)
plt.title('Discount Factor Decay Over Steps', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.xlim(0, HORIZON - 1)
plt.ylim(0, 1.05)

# Display plot
plt.tight_layout()
plt.show()