import numpy as np

x = [[1,2],[3,4],[5,6]]
x = np.asarray(x)
print(x.shape)
print(x)

#means
mu_vec = np.mean(x, axis=0)
print()
print("means")
print(mu_vec)
print(mu_vec.shape)

#stds
std_vec = np.std(x, axis=0)
print()
print("stds")
print(std_vec)
print(std_vec.shape)