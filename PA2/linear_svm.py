import numpy as np
from scipy.sparse import issparse, csr_matrix


class LinearSvmSGD:

    def __init__(self,
                 dimensions: int,
                 learning_rate: dict[str, float],
                 l2_radius: dict[str, float] | None = None,  # might be always 1.0
                 optimiser: str = 'sgd',
                 momentum_beta: float = 0.9,
                 adagrad_epsilon: float = 1e-8):

        self.dimensions = dimensions
        self.weight_vector = np.zeros(dimensions, dtype=np.float32)
        self.learning_rate = learning_rate
        self.l2_radius = l2_radius
        self.optimiser = optimiser.lower()

        self.velocity = np.zeros(dimensions, dtype=np.float32)
        self.gradient_square_sum = np.zeros(dimensions, dtype=np.float32)
        self.momentum_beta = momentum_beta
        self.adagrad_epsilon = adagrad_epsilon
        self.update_counter = 0

    def _current_learning_rate(self) -> float:
        """ eta_t = eta / sqrt(t)"""
        if self.optimiser == "sgd":
            return self.learning_rate['sgd'] / np.sqrt(max(1, self.update_counter))
        elif self.optimiser == "momentum":
            return self.learning_rate['momentum'] / np.sqrt(max(1, self.update_counter))
        elif self.optimiser == "adagrad":
            return self.learning_rate['adagrad'] / np.sqrt(max(1, self.update_counter))
        else:
            raise NotImplementedError

    def _project_l2(self, weight_vector: np.ndarray, radii: dict[str, float]) -> np.ndarray:
        norm = np.linalg.norm(weight_vector)
        rad = radii[self.optimiser]
        if norm <= rad:
            return weight_vector
        return weight_vector * (rad / norm)

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        """training step per batch"""
        self.update_counter += 1
        labels = labels.ravel()
        margins = labels * features.dot(self.weight_vector)
        mask = margins < 1
        if issparse(features):
            selected = features[mask]
            gradient = -selected.multiply(labels[mask][:, np.newaxis]).mean(axis=0).A1 if np.any(mask) else np.zeros(
                self.dimensions)
        else:
            gradient = - np.mean(labels[mask, None] * features[mask], axis=0) if np.any(mask) else np.zeros(
                self.dimensions)

        # calc the step
        learning_rate = self._current_learning_rate()
        # vanilla sgd
        if self.optimiser == 'sgd':
            step = learning_rate * gradient

        # momentum step
        elif self.optimiser == 'momentum':
            self.velocity = self.momentum_beta * self.velocity + learning_rate * gradient
            step = self.velocity

        # step with adagrad
        elif self.optimiser == 'adagrad':
            self.gradient_square_sum += gradient ** 2
            adjusted_learining_rate = learning_rate / (np.sqrt(self.gradient_square_sum) + self.adagrad_epsilon)
            step = adjusted_learining_rate * gradient
        else:
            raise NotImplementedError

        # update weights
        self.weight_vector -= step
        # project into the l2 ball
        self.weight_vector = self._project_l2(self.weight_vector, self.l2_radius)
