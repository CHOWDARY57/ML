from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
import numpy as np

X = load_iris().data

gmm = GaussianMixture(n_components=3, random_state=42, init_params='random').fit(X)


weights, means, covariances = gmm.weights_, gmm.means_, gmm.covariances_


density = np.exp(gmm.score_samples(X))


print("Estimated Weights:\n", weights)
print("\nEstimated Means:\n", means)
print("\nEstimated Covariances:\n", covariances)
print("\nExpected Density for the first few samples:\n", density[:5])
