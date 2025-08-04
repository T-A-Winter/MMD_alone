import csv
from pathlib import Path

from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_iter(file_path: Path):
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            yield row

def get_w_star(file_path: Path):
    df = pd.read_csv(file_path, delimiter=",")
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(float).values

    model = LinearSVC(fit_intercept=False, loss="hinge")#C=1e16, fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    w_star = model.coef_.flatten()

    return w_star

def project(w, rad=1):
    norm = np.linalg.norm(w)
    if norm <= rad:
        return w
    return w * (rad / norm)

if __name__ == "__main__":
    data_path = Path("toydata_small.csv")
    dim = 4
    w_star = project(get_w_star(data_path)) # project w_star into the feasibility set

    step_sizes = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    for step in step_sizes:
        w = np.zeros(dim)
        cumulated_regret = 0
        regret_history = []
        for t, (x_1, x_2, x_3, x_4, y) in enumerate(data_iter(data_path), start=1):
            x = np.array([float(x_1), float(x_2), float(x_3), float(x_4)])
            y = float(y)

            m = y * np.dot(w, x)
            gradient = -y * x if m < 1 else np.zeros(dim)

            eta = step / np.sqrt(t)
            w_tilde = w - eta * gradient
            w = project(w_tilde) # project w into the feasibility set

            online_loss = max(0, 1 - m)
            optimal_loss = max(0, 1 -(y * np.dot(w_star, x)))

            cumulated_regret += online_loss - optimal_loss
            regret_history.append(cumulated_regret)

        plt.plot(range(1, len(regret_history)+1), regret_history, label=f"eta={step}")
        plt.xlabel("Runde t")
        plt.ylabel("Kumulierter Regret")
        plt.title("Regret-Verlauf")
        plt.legend()
        plt.grid(True)
        plt.show()