from pathlib import Path
from typing import Optional
from time import monotonic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics.pairwise import distance_metrics
from sklearn.metrics import normalized_mutual_info_score

from log_config import get_logger
from lsh import LSH


class KMeans:
    def __init__(self, X: pd.DataFrame, y: pd.Series, k, path_to_log: Optional[Path] = None, max_iterations: int = 100_000):
        self.logger = get_logger(name="KMeans", path_to_log=path_to_log)

        self.X = X.to_numpy(dtype=np.float32) # grabbing the samples
        self.k = k # num of clusters
        self.true_y = y.to_numpy().flatten() # true labels

        self.max_iterations = max_iterations # hard stop in case of no convergence
        max_k = X.shape[0]
        # NOTE: what if k > entries? or k is negative
        if k > max_k:
            self.k = max_k
        elif k < 1:
            self.k = 1

        self.centroids = X.iloc[:k].to_numpy() # initializing the first k points as clusters
        self.n = X.shape[0] # num of samples

        self.labels = None # future label assignment
        
        # performance markers
        self.iterations_took = 0
        self.start_time = None
        self.end_time = None
        self.distance_calculations = 0
        self.score = None

    def fit_classic(self):
        self.logger.info(f"Fitting KMeans model with {self.k} clusters on {self.n} samples with dimensions {self.X.shape}")
        self.start_time = monotonic()
        # initialization
        cluster_assignments = np.zeros(self.n, dtype=np.int32)
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            # distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
            distances = np.empty((self.n, self.k), dtype=np.float32)
            diff = np.empty_like(self.X, dtype=np.float32) # pre alocating
            for j in range(self.k):
                diff = self.X - self.centroids[j]
                distances[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
                self.distance_calculations += self.n

            
            new_assignments = np.argmin(distances, axis=1)

            # converged ?
            if np.array_equal(new_assignments, cluster_assignments):
                self.logger.info(f"Converged after {iterations} iterations")
                break

            cluster_assignments = new_assignments

            # recalc centroids
            for j in range(self.k):
                assigned_points = self.X[cluster_assignments == j]
                if len(assigned_points) > 0:
                    self.centroids[j] = assigned_points.mean(axis=0)

        self.end_time = monotonic()

        if iterations == self.max_iterations:
            self.logger.warning("Maximum number of iterations reached")

        self.labels = cluster_assignments
        self.iterations_took = iterations
        self.score = normalized_mutual_info_score(self.true_y, self.labels)
        self.logger.info(f"KMeans Lloyd DONE - took {self.end_time - self.start_time} seconds with {self.iterations_took} iterations - NMI: {self.score} - num distances: {self.distance_calculations}")


    def fit_with_lsh(self, num_tables: int, hash_size: int):
        self.logger.info(f"Starting KMeans with LSH with {num_tables} tables and {hash_size} hash size for n={self.n} samples")

        self.start_time = monotonic()

        cluster_assignments = np.full(self.n, -1)
        iterations = 0
        lsh = LSH(hash_size=hash_size, num_tables=num_tables, input_dimension=self.X.shape[1])

        # init all points
        for index, vector in enumerate(self.X):
            lsh.set_item(vector, label=str(index))

        while iterations < self.max_iterations:
            iterations += 1
            new_assignments = np.full(self.n, -1)

            # finding near neighbor points for each centroid
            for j, centroid in enumerate(self.centroids):
                hashes, bucket = lsh.query(centroid)
                indices = [int(i) for i in bucket.genres]

                for i in indices:
                    distance = np.linalg.norm(self.X[i, :] - centroid)
                    self.distance_calculations += 1
                    if new_assignments[i] == -1:
                        new_assignments[i] = j
                    else:
                        old_centriod = self.centroids[new_assignments[i]]
                        if np.linalg.norm(self.X[i, :] - old_centriod) > distance:
                            new_assignments[i] = j

            # converged?
            if np.array_equal(new_assignments, cluster_assignments):
                self.logger.info(f"LSH-KMeans converged in {iterations} iterations.")
                break

            cluster_assignments = new_assignments

        # recalc centriods
        for j in range(self.k):
            assigned_points = self.X[cluster_assignments == j]
            if len(assigned_points) > 0:
                self.centroids[j] = assigned_points.mean(axis=0)

        self.end_time = monotonic()

        if iterations == self.max_iterations:
            self.logger.warning("Maximum number of iterations reached")

        self.labels = cluster_assignments
        self.iterations_took = iterations
        self.score = normalized_mutual_info_score(self.true_y, self.labels)
        self.logger.info(f"KMeans with LSH DONE - took: {self.end_time - self.start_time} seconds - {self.iterations_took} iterations - NMI: {self.score} - num distance calculations: {self.distance_calculations}")

    def fit_with_core_sets(self, m, alpha: int = 2):
        self.logger.info(f"Starting KMeans with core sets of m={m}, k={self.k}, alpha={alpha}, datashape shape={self.X.shape}")

        self.start_time = monotonic()

        n = self.X.shape[0]

        # find initial clusters, we use k-means++ for that - it only has one iteration
        k_prime = self.k * alpha
        k_means = SKKMeans(n_clusters=k_prime, init="k-means++", n_init=1, random_state=1, max_iter=10)
        k_means.fit(self.X)
        assignments = k_means.labels_
        centroids = k_means.cluster_centers_

        # calculate the distances for each point to the nearest cluster
        distances = self.X[:, None, :] - centroids[None, :, :]
        distances_squared = np.min((distances **2).sum(axis=2), axis=1) # d_i = ||x_i - mu_j||**2

        # getting the sum of all distances
        distances_sum = np.sum(distances_squared)

        # now calculating the sensitivity's
        # s_i = (d_i / sum d_j) + 1/n
        sensitivities = (distances_squared / distances_sum) + (1 / n)
        sensitivities_sum = np.sum(sensitivities) # S

        # getting the probability for sampling with q_i = s_i / S
        probabilities = sensitivities / sensitivities_sum # q_i
        # now choosing the indicies
        random_indices = np.random.choice(n, size=m, replace=True, p=probabilities)

        # now we have the core set
        X_coreset = self.X[random_indices]

        # calculating the weights of the coreset points with w_i = 1 / m * q_i
        weights = 1 / (m*probabilities[random_indices])

        # finally we can run k-means on the core set
        coreset_kmeans = SKKMeans(n_clusters=self.k, init='k-means++', n_init=10, random_state=1, max_iter=self.max_iterations)
        coreset_kmeans.fit(X_coreset, sample_weight=weights)

        # getting the centers
        self.centroids = coreset_kmeans.cluster_centers_
        # prediction the labels from the coreset clustered dataset
        self.labels = coreset_kmeans.predict(self.X)

        self.end_time = monotonic()
        self.iterations_took = coreset_kmeans.n_iter_
        self.score = normalized_mutual_info_score(self.true_y, self.labels)
        self.logger.info(f"Coreset-based KMeans fit with m={m} coreset samples DONE - Time took {self.end_time - self.start_time} - {self.iterations_took} iterations - NMI = {self.score}")


    def plot(self):
        X = self.X
        centroids = self.centroids
        labels = self.labels

        runtime = self.end_time - self.start_time if self.start_time and self.end_time else None # might be null
        self.logger.info(f"NMI score: {self.score:.4f}")
        self.logger.info(f"Runtime: {runtime:.4f} seconds")
        self.logger.info(f"Iterations: {self.iterations_took}")
        self.logger.info(f"Distance calculations: {self.distance_calculations}")

        if self.X.shape[1] > 2:
            self.logger.info("Applying PCA for plotting since data has more then 2 dimensions")
            pca = PCA(n_components=2)
            X = pca.fit_transform(self.X)
            centroids = pca.transform(self.centroids)

        if self.k <= 10:
            cmap = plt.get_cmap("tab10")
        elif self.k <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("nipy_spectral")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=10, alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')

        plt.title("KMeans Clustering Results")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)

        # Only show legend if k is small enough
        if self.k <= 20:
            plt.legend()
        else:
            self.logger.info("Omitting legend due to large number of clusters")

        plt.colorbar(scatter, label="Cluster ID")
        plt.show()