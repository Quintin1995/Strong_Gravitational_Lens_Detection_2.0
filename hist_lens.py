
## To emperically determine the noise galaxy count.
import matplotlib.pyplot as plt
import numpy as np

# Observations
data = 101*[0] + 53*[1] + 28*[2] + 17*[3] + 8*[4] + 1*[5]
data = 10*data
print("num samples: {}".format(len(data)))
print("mean: {}".format(np.mean(np.asarray(data))))


## Theoretical
mu, sigma = 0, 2
n_sam = len(data)
theory_dat = []
for i in range(n_sam):
    theory_dat.append(abs(int(np.random.normal(mu, sigma, 1))))

# plotting
count, bins, ignored = plt.hist(data, max(data), density=True, alpha=0.5, label="observations")
count, bins, ignored = plt.hist(theory_dat, max(theory_dat), density=True, alpha=0.5, label="theory")
plt.legend()
plt.show()
