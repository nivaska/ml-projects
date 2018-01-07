from bottomup_cluster import *
import matplotlib.pyplot as plt

X = np.loadtxt("s1.txt")
c = AggloClustering(X[0:5000, :])
clusters = c.cluster(15, linkage="average")

LABEL_COLOR_MAP = ['b','g','r','c','m','y','k']
[plt.scatter(cluster[:, 0], cluster[:, 1], c=LABEL_COLOR_MAP[i], marker='+') for i, cluster in enumerate(clusters[0:7])]
[plt.scatter(cluster[:, 0], cluster[:, 1], c=LABEL_COLOR_MAP[i], marker='^') for i, cluster in enumerate(clusters[7:14])]
plt.scatter(clusters[14][:, 0], clusters[14][:, 1], c='b', marker='o')
plt.show()


#Example with Scipy hierarchy clustering
# from matplotlib import pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage

# import numpy as np
# np.random.seed(4711)  # for repeatability of this tutorial
# a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
# b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
# X = np.concatenate((a, b),)

# Z = linkage(X, 'average')
# from scipy.cluster.hierarchy import fcluster
# k=15
# clusters = fcluster(Z, k, criterion='maxclust')
#
# plt.figure(figsize=(10, 8))
# plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
# plt.show()