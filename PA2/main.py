import itertools
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import csv
from time import monotonic

import statistics as stats

from file_iteration_and_general_utilities import train_svm, plot, get_dim, enhanced_plot
from linear_svm import LinearSvmSGD
from rff_sampler import RFFSampler


def summarise_best_configs():
    """
    Iterate over the produced csv files from the parameter tuning and
    write the best configurations to a csv file.
    """
    summary_file = Path("datasets/best_parameters_for_each_dataset.txt")
    best_config = {}

    for csv_file in Path().glob("hyper_parameters_*.csv"):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["dataset"], row["optimiser_variant"])
                try:
                    score = float(row["tail_avg_loss"])
                except ValueError:
                    continue

                if key not in best_config or score < best_config[key][1]:
                    best_config[key] = (row, score)

    # write summary
    with open(summary_file, "w") as fh:
        fh.write(f"# Best configurations by tail-avg-loss\n")
        fh.write(f"# Generated at {datetime.now().isoformat(timespec='seconds')}\n")
        for (dataset, optimiser), (row, _) in sorted(best_config.items()):
            fh.write(
                f"{dataset}: optimiser={optimiser}, batch_size={row['batch_size']}, "
                f"with_rff={row['with_rff']}, lr={row['learning_rate']}, "
                f"l2_radius={row['l2_radius']}, tail_avg_loss={float(row['tail_avg_loss']):.6f}\n"
            )

    print(f"Summary written to {summary_file}")


def process_dataset(dataset_path: str, epoches: int, chunk_size: int, optimiser_variants: list[str],
                    batch_sizes: list[int]):
    """
    Function executed per dataset.
    Iterate over a grid.
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem.lower()
    output_path = Path(f"hyper_parameters_{dataset_name}.csv")

    with_rff_options = [False, True]

    # grid
    # lr_values = [i / 100 for i in range(1, 201)]  # 0.01 to 2.00
    # l2_values = [i / 10 for i in range(1, 51)]  # 0.1 to 5.0
    lr_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    l2_values = [0.1, 1.0, 2.5, 5.0]

    with open(output_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "dataset", "optimiser_variant", "batch_size", "with_rff",
            "learning_rate", "l2_radius", "tail_avg_loss", "time_elapsed_sec"
        ])
        writer.writeheader()

        for lr, l2, use_rff in itertools.product(lr_values, l2_values, with_rff_options):
            lr_dict = {opt: lr for opt in optimiser_variants}
            l2_dict = {opt: l2 for opt in optimiser_variants}

            results_list, _ = start_svm_run(
                dataset_path=dataset_path,
                batch_sizes=batch_sizes,
                optimiser_variants=optimiser_variants,
                epoches=epoches,
                learning_rate=lr_dict,
                l2_radius=l2_dict,
                dataset_name=dataset_name,
                chunk_size=chunk_size,
                with_rff=use_rff,
                with_plots=False,
            )

            for r in results_list:
                losses = r["loss_history"]
                if not losses:
                    continue
                tail_avg = stats.mean(losses[-2:]) if len(losses) >= 2 else losses[-1]

                writer.writerow({
                    "dataset": dataset_name,
                    "optimiser_variant": r["optimiser_variant"],
                    "batch_size": r["batch_size"],
                    "with_rff": use_rff,
                    "learning_rate": lr,
                    "l2_radius": l2,
                    "tail_avg_loss": tail_avg,
                    "time_elapsed_sec": r["time_elapsed"]
                })

    return f"{dataset_name} done ({output_path})"


def hyper_parametrisation(batch_sizes: list[int], epoches: int, chunk_size: int, optimiser_variants: list[str],
                          datasets: list[str]):
    """
    Parameter Tuning.
    Launches one process per dataset, each evaluating a grid.
    """

    print(f"Starting hyper-parameter search on {len(datasets)} datasets...")

    with ProcessPoolExecutor(max_workers=len(datasets)) as pool:
        futures = [pool.submit(process_dataset, ds, epoches, chunk_size, optimiser_variants, batch_sizes) for ds in
                   datasets]
        for future in futures:
            print(future.result())  # print done message

    print("All datasets completed.")


def start_svm_run(dataset_path: Path, batch_sizes: list[int], optimiser_variants: list[str] | str, epoches: int,
                  learning_rate: dict[str, float], l2_radius: dict[str, float], dataset_name: str, chunk_size: int,
                  with_rff: bool = False, with_plots: bool = True, max_rows: int = 100_000):
    """
    :param dataset_path: path to dataset
    :param batch_sizes: list of batch sizes for minibatching
    :param optimiser_variants: list of optimiser variants. Currently supported variants: sgd, momentum, adagrad
    :param epoches: number of epoches to run
    :param learning_rate: dict of learning rates for each optimiser variant
    :param l2_radius: dict of l2 radius for each optimiser variant
    :param dataset_name: str name of dataset. Used to calc the accuracy after learning
    :param chunk_size: used to determine how many rows at once should be loaded into memory. This should be larger then batch_size
    :param with_rff: with or without random fourier features
    :param with_plots: choose False for debugging
    :param max_rows: maximum number of rows which should be iterated through in the dataset

    """
    # always list patterns
    if isinstance(optimiser_variants, str):
        optimiser_variants = [optimiser_variants]

    dimensions = get_dim(dataset_path)

    # for RFF
    feature_map = None
    if with_rff:
        D = 1000
        sigma = 1.0
        rff_sampler = RFFSampler(input_dim=dimensions,
                                 num_features=D,
                                 sigma=sigma)
        dimensions = D
        feature_map = rff_sampler.transform

    all_results = []
    for optimiser_variant in optimiser_variants:
        for batch_size in batch_sizes:
            model = LinearSvmSGD(dimensions=dimensions,
                                 learning_rate=learning_rate,
                                 l2_radius=l2_radius,
                                 optimiser=optimiser_variant)

            results = train_svm(model=model,
                                batch_size=batch_size,
                                epoches=epoches,
                                dataset_name=dataset_name,
                                data_set=dataset_path,
                                chunk_size=chunk_size,
                                feature_map=feature_map,
                                max_rows=max_rows)

            results.update({
                "optimiser_variant": optimiser_variant,
                "batch_size": batch_size,
                "with_rff": with_rff,
            })
            all_results.append(results)

    if with_plots:
        plot(all_results, dataset_name)

    return all_results, dataset_name


def parameter_tuning():
    """
    Find the optimal hyper-parameters for a all datasets in the folder. All tested parameters will be saved in a csv file and the
    optimal solution be writen to a txt file.
    The grid search is as follows: learning rate 0.01 up to 2.00 with a step size of 0.02. L2 radius 0.1 up to 5.0 with a step size of 0.5
    NOTE: not to be used since it would take to long.
    """
    chunk_size = 32768
    epoches = 10
    batch_sizes = [1, 8, 32, 128, 256]
    datasets = [
        "toydata_tiny.csv",
        "toydata_small.csv",
        "imdb.npz",
        "higgs.npz",
    ]

    optimiser_variants = ["sgd", "momentum", "adagrad"]

    hyper_parametrisation(batch_sizes=batch_sizes,
                          epoches=epoches,
                          chunk_size=chunk_size,
                          optimiser_variants=optimiser_variants,
                          datasets=datasets)

    summarise_best_configs()


if __name__ == "__main__":
    # adjust parameters here

    # TODO: choose your dataset - you it would be beneficial to use absolute paths
    data_set_path = Path("datasets/toydata_large.csv")
    # TODO: adjust accordingly
    dataset_name = "toydata_large"
    # if batch_size == 1 -> we are in online mode
    batch_sizes = [1, 8, 32, 128]
    # to not wait to long, you can also just pass "sgd" as a str to omit the other optimisers
    optimiser_variants = ["sgd", "momentum", "adagrad"]
    # TODO: you can always set higher chunk sizes. This works like mini batching but we now really only
    #  load 'chunk_size' of rows into memory. -> if 'chunk_size' is set to dataset feature matrix.shape[0] then we
    #  load the whole data at once. This might
    chunk_size = 32768
    # TODO: you might wanna choose lower epoches if running takes to long
    epoches = 10

    # Here you can adjust the parameters
    learning_rate = {
        "sgd": 1.87,
        "momentum": 1.83,
        "adagrad": 1.77,
    }

    l2_radius = {
        "sgd": 4.9,
        "momentum": 4.9,
        "adagrad": 4.9,
    }

    # with/ without random fourier features
    with_rff = True

    # NOTE: choose only few rows for a quick check and debugging
    MAX_ROWS = 100

    start_svm_run(dataset_path=data_set_path,
                  batch_sizes=batch_sizes,
                  optimiser_variants=optimiser_variants,
                  epoches=epoches,
                  learning_rate=learning_rate,
                  l2_radius=l2_radius,
                  dataset_name=dataset_name,
                  chunk_size=chunk_size,
                  with_rff=with_rff,
                  max_rows=MAX_ROWS)

