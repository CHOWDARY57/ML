from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
import numpy as np

# Load dataset
X = load_iris().data

# Initialize and fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42, init_params='random').fit(X)

# Retrieve parameters estimated by the EM algorithm
weights, means, covariances = gmm.weights_, gmm.means_, gmm.covariances_

# Calculate density for each sample
density = np.exp(gmm.score_samples(X))

# Print estimated parameters and density for the first few samples
print("Estimated Weights:\n", weights)
print("\nEstimated Means:\n", means)
print("\nEstimated Covariances:\n", covariances)
print("\nExpected Density for the first few samples:\n", density[:5])
