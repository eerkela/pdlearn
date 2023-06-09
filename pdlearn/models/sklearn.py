from __future__ import annotations

import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
import pandas as pd

from pdlearn.encode import FeatureEncoder, TargetEncoder
from pdlearn.util.parse import shorten_labels
from pdlearn.util.reshape import extract_columns


######################
####    PUBLIC    ####
######################


class AutoModel:
    """Mixin class that adds stats/formatting for pandas-linked AutoML models.
    """

    ###########################
    ####    FIT/PREDICT    ####
    ###########################

    def fit(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None = None
    ) -> AutoModel:
        """Fit the classifier to the given training set (X, y)."""
        print("Extracting columns...")

        # extract and reorder columns
        x_train = extract_columns(train, self.features)
        y_train = extract_columns(train, self.target)
        x_test = None if test is None else extract_columns(test, self.features)
        y_test = None if test is None else extract_columns(test, self.target)

        print("Encoding features...")

        # encode
        x_train = self.feature_encoder.fit_transform(x_train)
        y_train = self.target_encoder.fit_transform(y_train)
        if x_test is not None:
            x_test = self.feature_encoder.transform(x_test)
            y_test = self.target_encoder.transform(y_test)

        print("Fitting...")

        # do fit
        return super().fit(
            X=x_train,
            y=y_train,
            X_test=x_test,
            y_test=y_test,
            dataset_name=getattr(train, "name", None)
        )

    def predict(
        self,
        df: pd.DataFrame,
        batch_size: int | None = None,
        n_jobs: int | None = None
    ) -> pd.DataFrame:
        """Predict target values based on new features."""
        # extract features
        features = extract_columns(df, self.features)

        # encode features
        features = self.feature_encoder.transform(features)

        # generate prediction
        result = pd.DataFrame(
            super().predict(
                features,
                batch_size=batch_size,
                n_jobs=n_jobs
            ),
            index=features.index,
            columns=self.target
        )

        # decode predictions
        return self.target_encoder.inverse_transform(result)

    def predict_proba(
        self,
        df: pd.DataFrame,
        batch_size: int | None = None,
        n_jobs: int | None = None
    ) -> pd.DataFrame:
        """Predict probabilities for target values based on new features."""
        # extract features
        features = extract_columns(df, self.features)

        # encode features
        features = self.feature_encoder.transform(features)

        # generate prediction
        return pd.DataFrame(
            super().predict_proba(
                features,
                batch_size=batch_size,
                n_jobs=n_jobs
            ),
            index=features.index,
            columns=self.target_encoder.get_feature_names()
        )

    def refit(self, df: pd.DataFrame) -> AutoModel:
        """Refit all models found with :meth:`AutoModel.fit` to new data.

        This method is necessary when using cross-validation.  During training,
        auto-sklearn fits each model k times on the dataset, but does not keep
        any of the trained models and can therefore not be used to predict for
        new data points.  This method fits all models found during a call to
        fit on the given data.  It may also be used together with holdout to
        avoid only using 66% of the training data to fit the final model.
        """
        # extract columns
        features = extract_columns(df, self.features)
        target = extract_columns(df, self.target)

        # encode
        features = self.feature_encoder.transform(features)
        target = self.target_encoder.transform(target)

        # refit
        return super().refit(X=features, y=target)

    def score(self, df: pd.DataFrame) -> float:
        """Return the mean accuracy on the given test data.

        In multi-label classification, this is a measure of subset accuracy,
        which is a harsh metric.  For each sample, the complete label set must
        be correctly predicted to count as a success.
        """
        # extract columns
        features = extract_columns(df, self.features)
        target = extract_columns(df, self.target)

        # encode
        features = self.feature_encoder.transform(features)
        target = self.target_encoder.transform(target)

        # score
        return super().score(features, target)

    ####################
    ####    INFO    ####
    ####################

    @property
    def ensemble(self) -> dict:
        """The final ensemble constructed by this classifier.
        """
        return self.show_models()

    def get_configuration_space(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame = None
    ):
        """Get the configuration space used to train this model."""
        # extract and reorder columns
        x_train = extract_columns(train, self.features)
        y_train = extract_columns(train, self.target)
        x_test = None if test is None else extract_columns(test, self.features)
        y_test = None if test is None else extract_columns(test, self.target)

        # encode
        x_train = self.feature_encoder.transform(x_train)
        y_train = self.target_encoder.transform(y_train)
        if x_test is not None:
            x_test = self.feature_encoder.transform(x_test)
        if y_test is not None:
            y_test = self.target_encoder.transform(y_test)

        return super().get_configuration_space(
            X=x_train,
            y=y_train,
            X_test=x_test,
            y_test=y_test,
            dataset_name=getattr(train, "name", None)
        )

    @property
    def models(self) -> pd.DataFrame:
        """A DataFrame showing the relative performance of each model generated
        By this classifier.
        """
        return self.leaderboard()

    @property
    def params(self) -> dict:
        """Get the AutoSklearn parameters used by this estimator"""
        return self.get_params()

    @params.setter
    def params(self, kwargs) -> None:
        self.set_params(**kwargs)

    @property
    def performance(self) -> pd.DataFrame:
        """A DataFrame containing performance over time for the training of
        this model.
        """
        return self.performance_over_time_

    @property
    def stats(self) -> str:
        """A formatted string containing basic diagnostics for this model."""
        return self.sprint_statistics()

    @property
    def weights(self) -> list:
        """A list of all models included in the final ensemble along with their
        respective weights.
        """
        return self.get_models_with_weights()

    #######################
    ####    SCORING    ####
    #######################

    # https://scikit-learn.org/stable/modules/model_evaluation.html
    # TODO: r_square()
    # TODO: mean_absolute_error
    # TODO: mean_square_error
    # TODO: error()

    #############################
    ####    MAGIC METHODS    ####
    #############################

    def __repr__(self) -> str:
        targets = shorten_labels(self.target, max_length=4, sep=" + ")
        features = shorten_labels(self.features, max_length=4, sep=" + ")
        return f"{type(self).__name__}({targets} ~ {features})"


class SklearnClassifier(AutoModel, AutoSklearnClassifier):
    """An ``AutoSklearnClassifier`` that automatically encodes and decodes
    its input data and prediction results.
    """

    def __init__(
        self,
        target: list,
        features: list,
        encode_features: bool = True,
        classifiers: list[str] = None,
        data_preprocessors: list[str] = None,
        balancers: list[str] = None,
        feature_preprocessors: list[str] = None,
        ensemble_size: int = 50,
        resampling: str = "cv",
        folds: int = 5,
        time_limit: int = 3600,
        memory_limit: int | float = 3072,
        seed: int = 1,
        n_jobs: int = -1,
        metric: autosklearn.metrics.Scorer = None,
        smac_scenario_args: dict = None,
        dask_client = None,
        logging_config: dict = None
    ):
        # record target/feature column labels
        self.target = target
        self.features = features

        # initialize encoders
        self.target_encoder = TargetEncoder(classify=True)
        self.feature_encoder = FeatureEncoder()

        # parse ensemble_size
        if ensemble_size == 1:
            ensemble_class = autosklearn.ensembles.SingleBest
            ensemble_kwargs = {}
        else:
            ensemble_class = "default"
            ensemble_kwargs = {"ensemble_size": ensemble_size}

        # build include list
        include = [
            ("data_preprocessor", data_preprocessors),
            ("balancing", balancers),
            ("feature_preprocessor", feature_preprocessors),
            ("classifier", classifiers)
        ]

        # construct classifier
        super().__init__(
            time_left_for_this_task=time_limit,
            per_run_time_limit=time_limit // 10,
            ensemble_class=ensemble_class,
            ensemble_kwargs=ensemble_kwargs,
            seed=seed,
            memory_limit=memory_limit,
            include={k: v for k, v in include if v is not None},
            resampling_strategy=resampling,
            resampling_strategy_arguments={
                "train_size": 0.67,  # for holdout resampling, not overall
                "shuffle": True,
                "folds": folds,  # for cross-validation
            },
            n_jobs=n_jobs,
            dask_client=dask_client,
            smac_scenario_args=smac_scenario_args,
            logging_config=logging_config,
            metric=metric
        )


class SklearnRegressor(AutoModel, AutoSklearnRegressor):
    """An ``AutoSklearnRegressor`` that automatically encodes and decodes
    its input data and prediction results.
    """

    def __init__(
        self,
        target: list,
        features: list,
        regressors: list[str] = None,
        data_preprocessors: list[str] = None,
        balancers: list[str] = None,
        feature_preprocessors: list[str] = None,
        ensemble_size: int = 50,
        resampling: str = "cv",
        folds: int = 5,
        time_limit: int = 3600,
        memory_limit: int | float = 3072,
        seed: int = 1,
        n_jobs: int = -1,
        metric: autosklearn.metrics.Scorer = None,
        smac_scenario_args: dict = None,
        dask_client = None,
        logging_config: dict = None
    ):
        # record target/feature column labels
        self.target = target
        self.features = features

        # initialize encoders
        self.feature_encoder = FeatureEncoder()
        self.target_encoder = TargetEncoder(classify=False)

        # parse ensemble_size
        if ensemble_size == 1:
            ensemble_class = autosklearn.ensembles.SingleBest
            ensemble_kwargs = {}
        else:
            ensemble_class = "default"
            ensemble_kwargs = {"ensemble_size": ensemble_size}

        # build include list
        include = [
            ("data_preprocessor", data_preprocessors),
            ("balancing", balancers),
            ("feature_preprocessor", feature_preprocessors),
            ("regressor", regressors)
        ]

        # construct classifier
        super().__init__(
            time_left_for_this_task=time_limit,
            per_run_time_limit=time_limit // 10,
            ensemble_class=ensemble_class,
            ensemble_kwargs=ensemble_kwargs,
            seed=seed,
            memory_limit=memory_limit,
            include={k: v for k, v in include if v is not None},
            resampling_strategy=resampling,
            resampling_strategy_arguments={
                "train_size": 0.67,  # for holdout resampling, not overall
                "shuffle": True,
                "folds": folds,  # for cross-validation
            },
            n_jobs=n_jobs,
            dask_client=dask_client,
            smac_scenario_args=smac_scenario_args,
            logging_config=logging_config,
            metric=metric
        )
