from __future__ import annotations

import autosklearn
import pandas as pd

import pdcast
from pdcast.util.type_hints import datetime_like, timedelta_like

from .models.sklearn import AutoModel, SklearnClassifier, SklearnRegressor
from .util.parse import parse_memory_limit, parse_n_jobs, parse_time_limit
from .util.reshape import column_specifier, extract_columns, split_train_test


def attach() -> None:
    """Attach ``pdlearn`` functionality to pandas DataFrame objects."""
    pd.DataFrame.fit = fit
    pd.DataFrame.split_train_test = split_train_test


def detach() -> None:
    """Detach ``pdlearn`` functionality from pandas DataFrame objects."""
    del pd.DataFrame.fit
    del pd.DataFrame.split_train_test


def fit(
    df: pd.DataFrame,
    target: column_specifier,
    features: column_specifier = None,
    algorithms: str | list[str] = None,
    data_preprocessors: str | list[str] = None,
    balancers: str | list[str] = None,
    feature_preprocessors: str | list[str] = None,
    train_size: float = 1.0,  # for performance over time
    ensemble_size: int = 50,
    resampling: str = "cv",
    folds: int = 5,
    time_limit: int | datetime_like | timedelta_like = 3600,
    memory_limit: int | float = 3072,
    seed: int = 1,
    n_jobs: int = -1,
    metric: autosklearn.metrics.Scorer = None,
    dask_client = None,
    smac_scenario_args: dict = None,
    logging_config: dict = None
) -> AutoModel:
    """Do a generalized automl fit on a DataFrame.

    Model selection is determined by the inferred type of the ``target``
    column(s).
    """
    # parse model settings
    time_limit = parse_time_limit(time_limit)
    n_jobs = parse_n_jobs(n_jobs)
    memory_limit = parse_memory_limit(memory_limit)
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    if isinstance(data_preprocessors, str):
        data_preprocessors = [data_preprocessors]
    if isinstance(balancers, str):
        balancers = [balancers]
    if isinstance(feature_preprocessors, str):
        feature_preprocessors = [feature_preprocessors]

    # extract columns
    target = extract_columns(df, target)
    if features is None:
        features = df.drop(target.columns, axis=1)
    else:
        features = extract_columns(df, features)
        overlap = [x for x in features.columns if x in target.columns]
        if overlap:
            raise ValueError(f"features overlap with target: {overlap}")

    # iterate through target columns to select model
    model_select = None
    for col_name, col in target.items():
        # detect type of column
        col_type = pdcast.detect_type(col)

        # if column is categorical, use an SklearnClassifier
        if col_type.is_subtype("bool, str, object") or col_type.is_categorical:
            model_select = "classify" if model_select is None else model_select
            if model_select != "classify":
                raise TypeError(
                    f"target columns must be uniformly categorical or "
                    f"numeric, not a mixture of both:\n{target}"
                )

        # if column is numeric, use an SklearnRegressor
        else:
            model_select = "regress" if model_select is None else model_select
            if model_select != "regress":
                raise TypeError(
                    f"target columns must be uniformly categorical or "
                    f"numeric, not a mixture of both:\n{target}"
                )

    # choose a model
    model_kwargs = {
        "target": target.columns.tolist(),
        "features": features.columns.tolist(),
        "data_preprocessors": data_preprocessors,
        "balancers": balancers,
        "feature_preprocessors": feature_preprocessors,
        "ensemble_size": ensemble_size,
        "resampling": resampling,
        "folds": folds,
        "metric": metric,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "seed": seed,
        "n_jobs": n_jobs,
        "dask_client": dask_client,
        "smac_scenario_args": smac_scenario_args,
        "logging_config": logging_config
    }
    if model_select == "classify":
        model = SklearnClassifier(classifiers=algorithms, **model_kwargs)
    elif model_select == "regress":
        model = SklearnRegressor(regressors=algorithms, **model_kwargs)
    else:
        raise ValueError(f"no model could be chosen for target:\n{target}")

    print(f"Generating {model}...")

    # return fitted model
    train, test = split_train_test(df, train_size=train_size, seed=seed)
    return model.fit(train, test)
