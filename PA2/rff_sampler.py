import numpy as np
from scipy.sparse import issparse


class RFFSampler:
    """used to get the feature map if rff is turned on"""
    def __init__(self, input_dim: int,
                 num_features: int,
                 sigma: float,
                 seed: int = 1):

        rng = np.random.default_rng(seed)

        self.omega = rng.normal(
            loc=0.0,
            scale=1.0 /sigma,
            size=(num_features, input_dim)
        ).astype(np.float32)

        self.phase = rng.uniform(0.0, 2*np.pi, size=num_features).astype(np.float32)

        self.scale = np.float32(np.sqrt(2.0 / num_features))

    def transform(self, features: np.ndarray):
        """returns the feature map"""

        projection = features @ self.omega.T if issparse(features) else features.dot(self.omega.T)

        random_phase = np.cos(projection + self.phase)

        return (self.scale * random_phase).astype(np.float32)
