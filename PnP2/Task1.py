import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
import scipy.spatial
import scipy.sparse
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm

def jaccard_similarity(X):
    """
    Computes the Jaccard similarity of matrix X of size n x d,
    where n is the number of samples and d the dimensionality of
    the samples.
    """
    # compute interesction
    intersection = (X @ X.T)

    # cardinalities of vectors
    cardinalities = intersection.diagonal()

    # Compute union using |A u B| = |A| + |B| - |A n B|
    unions = cardinalities[:,None] + cardinalities - intersection

    return intersection / unions


if __name__ == "__main__":
    # load data
    f = h5py.File("data.hdf5", "r")
    X_data = np.array(f['data-data']).astype('float')
    X_indices = np.array(f['data-indices'])
    X_indptr = np.array(f['data-indptr'])
    X = scipy.sparse.csr_matrix((X_data, X_indices, X_indptr))
    signature = np.array(f['signature'])
    f.close()

    # compute Jaccard similarities
    sim = np.array(jaccard_similarity(X))

    ### >> TODO
    # conv the spare matrix into dense form
    # so the calculations a a bit more straigt forward
    sim = np.array(jaccard_similarity(X).todense())

    errors = []
    num_hashes = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for k in num_hashes:
        # taking the subsets of the signatures
        sig_subset = signature[:k, :]


        dist =  pdist(sig_subset.T, metric="hamming")
        approx_sim = 1 - squareform(dist)

        sse = np.sum((sim - approx_sim)**2)
        errors.append(sse)

        true_neighbors = np.argsort(-sim[0])[1:11]
        approx_neighbors = np.argsort(-approx_sim[0])[1:11]
        union = list((set(approx_neighbors).intersection(set(true_neighbors))))

        print(f"For k={k} - True neighbors    : {', '.join(map(str, true_neighbors))}")
        print(f"For k={k} - Approx neighbors  : {', '.join(map(str, approx_neighbors))}")
        print(f"For k={k} - Union neighbors   : {', '.join(map(str, union))}")


    plt.plot(num_hashes, errors, marker="o")
    plt.xlabel("# Hash Functions")
    plt.ylabel("SSE")
    plt.grid(True)
    plt.show()

    ### << TODO