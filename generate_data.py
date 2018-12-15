""" generate_data.py generates toy data for the Subtype-Oriented Disease Axes 
(SODA) project. It generates two files: 
"""
import numpy as np
import scipy.stats

np.random.seed(0)

n_features = 20
n_id = 100
n_class = 5

class_ii = np.zeros([n_id, n_class], int)

class_ii[:, :n_class - 1] = 1 + np.random.choice(2, [n_id, n_class - 1], p = [.3, .7])

class_ii[:, -1] = 1 + np.random.choice(4, n_id, p = [.2, .2, .2, .4])

feature_ii = np.zeros([n_id, n_features])

feature_class = n_features // n_class

for cc in range(n_class):
    unique_cc = np.unique(class_ii[:, cc])
    for cccc in unique_cc:
        Sigma = scipy.stats.wishart.rvs(feature_class, np.eye(feature_class))
        mu = np.random.normal(0, size = 4)
        n_cccc = (class_ii[:, cc] == cccc).sum()
        feature_ii[class_ii[:, cc] == cccc, cc * feature_class : (cc + 1) * feature_class] = \
            np.random.multivariate_normal(mu, Sigma, size = n_cccc)
        

f = open("cluster.csv", "w")

header = "id" + (",Cluster{}" * n_class).format(
            *[ii + 1 for ii in range(n_class)]
        ) + "\n"

f.write(header)
for ii in range(n_id):
    f.write("XX{0:06d}".format(ii + 1))
    f.write( (",{}" * n_class).format(*class_ii[ii, :]) )
    f.write("\n")

f.close()

f = open("data.csv", "w")

header = "id" + (",Feature{}" * n_features).format(
            *[ii + 1 for ii in range(n_features)]
        ) + "\n"
        

f.write(header) 
for ii in range(n_id):
    f.write("XX{0:06d}".format(ii + 1))
    f.write( (",{:.2f}" * n_features).format(*feature_ii[ii, :]) )
    f.write("\n")


f.close()