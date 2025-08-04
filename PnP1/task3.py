import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics

# >> TODO: Adjust the code of the mapper here
def mapper(key, value, weights):
    """
    key ... index of data split
    value ... a tuple of the form (X, y) where y is a vector of target values (shape: n_samples x 1) and X a matrix of features (shape: n_samples x n_dim)
    weights ... parameters of the OLS model

    yields a generator of tuples, where the first entry in the tuple is an n-gram and the second entry is the count
    """
    X, y = value
    #yield ("count", len(y))
    yield ("error", (len(y), np.sum(np.abs(y - np.matmul(X, weights)))))

def reducer(key, values):
    """
    key ... "MSE"
    values ... number of samples in batch
    """
    AE = 0
    n = 0
    for n_, se_ in values:
        AE += se_
        n += n_
    yield ("MAE", AE / n)

# << TODO: End of code you have to change

if __name__ == "__main__":
    print("Running MapReduce for evaluating an OLS model...")

    # (1) Load Diabetes housing data and fit OLS model
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    reg.fit(X, y)

    # (2) Split data into blocks of 50 samples and run mappers
    idx_list = []
    X_list = []
    y_list = []
    weights_list = []
    for i in range(int(np.ceil(len(X) / 50))):
        sidx = i * 50
        eidx = (i + 1) * 50
        X_list.append(X[sidx:eidx])
        y_list.append(y[sidx:eidx].reshape(-1,1))
        weights_list.append(reg.coef_.reshape(len(reg.coef_), 1))
        idx_list.append(i)
    mapper_results = map(mapper, idx_list, zip(X_list, y_list), weights_list)

    # (3) Gather results from mappers, sort and run reducers
    mapper_results = list(mapper_results)
    mapper_results_dict = {}
    for mapper_result in mapper_results:
        for key, value in mapper_result:
            if key not in mapper_results_dict:
                mapper_results_dict[key] = []
            mapper_results_dict[key].append(value)
    mapper_results_dict = mapper_results_dict.items()
    reducer_results = map(reducer, [x[0] for x in mapper_results_dict], [x[1] for x in mapper_results_dict])

    # (4) Gather restults form reducers and output them
    reducer_results = list(reducer_results)
    reducer_results = [list(x) for x in reducer_results]
    
    print("MAE...")
    print(reducer_results)

    print("MAE according to sklearn (should equal your result):")
    print(sklearn.metrics.mean_absolute_error(y, reg.predict(X)))
