import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Iterator, Tuple, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import issparse, csr_matrix, vstack
from tqdm import tqdm

from linear_svm import LinearSvmSGD


def get_dim(dataset_path: Path) -> int:
    """
    Trying to read out the dimensionality of the dataset and return it, which is needed for the LinearSvmSGD initialization
    :param dataset_path: path to dataset
    :return: dimension of dataset
    """
    if dataset_path.suffix == '.csv':
        df = pd.read_csv(dataset_path, nrows=1)
        return df.shape[1] - 1

    elif dataset_path.suffix == '.npz':
        data = np.load(dataset_path, allow_pickle=True)
        if "train" in data and "train_labels" in data:
            feature_matrix = data["train"]
            if isinstance(feature_matrix, np.ndarray) and feature_matrix.dtype == object and feature_matrix.shape == ():
                return feature_matrix.item().shape[1]
            return feature_matrix.shape[1]
        else:
            raise ValueError("Something went wrong while trying to read dim for data with suffix .npz")
    else:
        raise NotImplementedError("Dont know this data suffix - please use csv or npz")


def normalize_rows(features: np.ndarray):
    if issparse(features):
        row_norm = np.sqrt(features.multiply(features).sum(axis=1)).A1
        row_norm[row_norm == 0] = 1
        return features.multiply(1.0 / row_norm[:, np.newaxis]).tocsr()
    else:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms


def load_data_as_matrix_and_vector_chunked_iter(dataset_path: Path, chunk_size: int, delimiter=',',
                                                max_rows: int = 100_000):
    """
    yielding and formating chunks of the dataset.
    :param dataset_path: Path to dataset
    :param chunk_size: chunk size which is loaded into memory
    :param delimiter: delimiter in the csv file
    :param max_rows: hard stop of iteration if total_rows_yielded >= max_rows. NOTE: total_rows_yielded can be larger than max_rows depending on chunk_size
    :return:
    """
    MAX_ROWS = max_rows
    total_rows_yielded = 0

    if dataset_path.suffix == ".csv":
        for chunk in pd.read_csv(dataset_path, delimiter=delimiter, chunksize=chunk_size):

            # early return
            remaining_rows = MAX_ROWS - total_rows_yielded
            if remaining_rows <= 0:
                break

            feature_matrix = chunk.iloc[:, :-1].astype(float).values
            labels = chunk.iloc[:, -1].astype(float).values
            feature_matrix = normalize_rows(feature_matrix)

            total_rows_yielded += len(labels)
            yield feature_matrix, labels

    elif dataset_path.suffix == ".npz":
        data = np.load(dataset_path, allow_pickle=True)
        if "train" in data and "train_labels" in data:

            feature_matrix = data["train"]
            if isinstance(feature_matrix, np.ndarray) and feature_matrix.dtype == object and feature_matrix.shape == ():
                feature_matrix = feature_matrix.item()

            labels = data["train_labels"]

            if labels.dtype == bool:
                labels = labels.astype(np.int8) * 2 - 1
            elif np.array_equal(np.unique(labels), [0, 1]):
                labels = labels.astype(np.int8) * 2 - 1

        else:
            raise ValueError("Something went wrong while loading the data with suffix .npz")

        for start in range(0, feature_matrix.shape[0], chunk_size):

            if total_rows_yielded >= MAX_ROWS:
                break

            end = min(start + chunk_size, feature_matrix.shape[0])

            # early return
            remaining_rows = MAX_ROWS - total_rows_yielded
            if (end - start) > remaining_rows:
                end = start + remaining_rows

            chunk_features = feature_matrix[start:end, :]
            chunk_labels = labels[start:end]
            chunk_features = normalize_rows(chunk_features)

            total_rows_yielded += len(chunk_labels)
            yield chunk_features, chunk_labels

    else:
        raise NotImplementedError("Dont know this data suffix - please use csv or npz")


def mini_batch_iter(feature_matrix: np.ndarray, labels: np.ndarray, batch_size: int) -> Iterator[
    Tuple[np.ndarray, np.ndarray]]:
    """yielding batches of size batch_size in random order for SGD"""
    feature_matrix_len = feature_matrix.shape[0]
    indices = np.arange(feature_matrix_len)

    np.random.shuffle(indices)
    for i in range(0, feature_matrix_len, batch_size):
        batch_indices = indices[i: i + batch_size]
        yield feature_matrix[batch_indices, :], labels[batch_indices]


def mean_hinge_loss(weight_vector: np.ndarray, features: np.ndarray, labels: np.ndarray):
    margins = labels * features.dot(weight_vector)
    losses = np.maximum(0, 1 - margins)
    return np.mean(losses)


def predict_scores(weight_vector: np.ndarray, features: np.ndarray) -> np.ndarray:
    # w^T * x
    return features.dot(weight_vector)


def train_svm(model: LinearSvmSGD, batch_size: int, epoches: int, dataset_name: str, data_set: Path, chunk_size: int,
              feature_map: Callable | None = None, max_rows: int = 100_000) -> dict:
    """
    Trains the given model on the dataset for the given number of epochs. Then calculating the accuracy score and returning the reuslts.
    :param model: svm model
    :param batch_size: batch size for mini-batches
    :param epoches: number of epochs
    :param dataset_name: dataset name
    :param data_set: patch to the dataset
    :param chunk_size: number of rows to be loaded into memory at once
    :param feature_map: feature map to use when rff is turned on
    :param max_rows: hard stop of iteration if total_rows_yielded >= max_rows
    :return:
    """
    start_time = monotonic()
    loss_history: list[float] = []

    for epoch in range(epoches):
        cumulative_loss = 0
        cumulative_count = 0

        for features, labels in load_data_as_matrix_and_vector_chunked_iter(dataset_path=data_set, chunk_size=chunk_size, max_rows=max_rows):
            # RFF
            if feature_map is not None:
                features = feature_map(features)

            for batch_feature, batch_label in mini_batch_iter(features, labels, batch_size):
                model.update(batch_feature, batch_label)

                batched_loss = mean_hinge_loss(model.weight_vector, batch_feature, batch_label)
                cumulative_loss += batched_loss * batch_feature.shape[0]
                cumulative_count += batch_feature.shape[0]

        average_loss = cumulative_loss / cumulative_count
        loss_history.append(average_loss)

    time_elapsed = monotonic() - start_time

    all_features, all_labels = [], []
    for features, labels in load_data_as_matrix_and_vector_chunked_iter(data_set, chunk_size):
        all_features.append(features)
        all_labels.append(labels)

    if issparse(all_features[0]):
        full_feature_matrix = vstack(all_features)
    else:
        full_feature_matrix = np.vstack(all_features)
    # RFF
    if feature_map is not None:
        full_feature_matrix = feature_map(full_feature_matrix)

    full_labels = np.concatenate(all_labels)

    scores = predict_scores(model.weight_vector, full_feature_matrix)
    if dataset_name.lower() == "higgs":
        performance = roc_auc_score(full_labels, scores)
    else:
        predictions = np.sign(scores)
        performance = accuracy_score(full_labels, predictions)

    return {
        "loss_history": loss_history,
        "performance": performance,
        "time_elapsed": time_elapsed,
        "labels": full_labels,
        "features": full_feature_matrix,
        "weights": model.weight_vector.copy(),
    }


def plot(all_results: list[dict], dataset_name: str) -> None:
    # ----------------- Ergebnisse gruppieren -----------------------------
    results_by_opt = defaultdict(list)
    for res in all_results:
        results_by_opt[res["optimiser_variant"]].append(res)

    n_opts = len(results_by_opt)
    cols = min(3, n_opts)
    rows = math.ceil(n_opts / cols)

    # Abbildung groß anlegen
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(7 * cols, 5 * rows),
        squeeze=False
    )

    # ----------------- Kurven zeichnen -----------------------------------
    for idx, (opt, res_list) in enumerate(sorted(results_by_opt.items())):
        ax = axes.flat[idx]
        for res in sorted(res_list, key=lambda r: r["batch_size"]):
            label = (f"bs: {res['batch_size']} | "
                     f"Perf: {res['performance']:.3f} | "
                     f"time: {res['time_elapsed']:.1f}s")
            ax.plot(res["loss_history"], label=label)

        ax.set_title(opt.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean hinge loss")
        ax.grid(True)
        ax.legend(fontsize="small")

    # überzählige Achsen entfernen
    for j in range(idx + 1, len(axes.flat)):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"Mean Hinge Loss {dataset_name}", fontsize=18, y=1.02)
    fig.tight_layout()

    # ----------------- Abbildung speichern -------------------------------
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"hinge_loss_{dataset_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert unter {out_path.resolve()}")


def enhanced_plot(all_results: list[dict], dataset_name: str) -> None:
    # Group results by optimiser
    results_by_optimiser = defaultdict(list)
    for result in all_results:
        results_by_optimiser[result["optimiser_variant"]].append(result)

    active_optimisers = sorted(results_by_optimiser.keys())
    num_optimisers = len(active_optimisers)
    cols = min(3, num_optimisers)
    rows = 2  # hinge + margin

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for i, optimiser_variant in enumerate(active_optimisers):
        col = i % cols

        results = sorted(results_by_optimiser[optimiser_variant], key=lambda r: r["batch_size"])

        # Hinge loss plot
        ax_loss = axes[0][col]
        for result in results:
            label = f"bs={result['batch_size']} time={result['time_elapsed']:.2f}"
            ax_loss.plot(result["loss_history"], label=label)

        ax_loss.set_title(f"{optimiser_variant.upper()} on {dataset_name}")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Mean hinge loss")
        ax_loss.grid(True)
        ax_loss.legend(fontsize="small")

        performance_text = "\n".join(
            f"bs={r['batch_size']}: {r['performance']:.3f}" for r in results
        )
        ax_loss.text(0.5, -0.35, f"Performance:\n{performance_text}",
                     transform=ax_loss.transAxes, ha='center', va='top', fontsize=9)

        # Margin distribution plot
        ax_dist = axes[1][col]
        for result in results:
            margins = result["labels"] * (result["weights"] @ result["features"].T)
            ax_dist.hist(margins, bins=30, alpha=0.5, label=f"bs={result['batch_size']}")

        ax_dist.set_title(f"Margin dist ({optimiser_variant})")
        ax_dist.set_xlabel("y * f(x)")
        ax_dist.set_ylabel("Count")
        ax_dist.grid(True)
        ax_dist.legend(fontsize="small")

    # Remove unused axes
    for row in range(rows):
        for col in range(num_optimisers, cols):
            fig.delaxes(axes[row][col])

    fig.suptitle("Convergence + Margin Distribution by Optimiser", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
