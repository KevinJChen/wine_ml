import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data/combined_wine.csv", delimiter=',')
data = data[~np.isnan(data).any(axis=1)]

# print(data)

feat_num = 11

X = data[:, 0]
X1 = data[:, feat_num]

Xr = data[:1600, 0]
Xr1 = data[:1600, feat_num]

Xw = data[1600:, 0]
Xw1 = data[1600:, feat_num]

Y = data[-1]                    # classification of 0 or 1
print(X)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#ax.scatter(X, X1, s=10, color='black', alpha=0.75, label="both")
ax.scatter(Xr, Xr1, s=10, color='red', alpha=0.75, label="red")
ax.scatter(Xw, Xw1, s=10, color='blue', alpha=0.75, label="white")


ax.legend(fontsize=12, loc=1)

plt.show()


