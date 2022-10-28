import numpy as np
from sklearn.utils import safe_sqr
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted
import gpboost as gpb


def gpboost_model_train(
    X, y, group, feature_name, categorical_features, gpb_params, num_boost_round=50
):
    gpb_train = gpb.Dataset(X, y)
    gp_model = gpb.GPModel(group_data=group, likelihood="bernoulli_probit")

    model = gpb.train(
        params=gpb_params,
        train_set=gpb_train,
        categorical_feature=categorical_features,
        feature_name=feature_name,
        gp_model=gp_model,
        num_boost_round=num_boost_round,
        use_gp_model_for_validation=True,
        train_gp_model_cov_pars=True,
    )

    return model


def gpboost_feature_importance(booster, importance_type="gain"):
    return booster.feature_importance(importance_type=importance_type)


class RFE_Gpboost(SelectorMixin, BaseEstimator, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination.
    Customised from RFE implemented in sklearn version '0.22.1'
    """

    def __init__(
        self,
        categorical_features,
        gpb_params,
        num_boost_round,
        importance_type,
        n_features_to_select=None,
        step=1,
        verbose=0,
    ):
        self.categorical_features = categorical_features
        self.gpb_params = gpb_params
        self.num_boost_round = num_boost_round
        self.importance_type = importance_type
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    def fit(self, Xgroup, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.
        """
        self.classes_ = np.unique(y)
        X = Xgroup.drop("eid", axis="columns")
        group = Xgroup["eid"]
        return self._fit(X, y, group)

    def _fit(self, X, y, group, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        feature_name = X.columns
        tags = self._get_tags()

        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            feature_name_ = list(feature_name[support_])
            categorical_features_ = [
                f for f in feature_name_ if f in self.categorical_features
            ]
            estimator = gpboost_model_train(
                X[feature_name_],
                y,
                group,
                feature_name_,
                categorical_features_,
                self.gpb_params,
                self.num_boost_round,
            )

            # Get coefs
            coefs = gpboost_feature_importance(estimator, self.importance_type)
            if coefs is None:
                raise RuntimeError(
                    "The classifier does not expose "
                    '"coef_" or "feature_importances_" '
                    "attributes"
                )

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        feature_name_ = list(feature_name[support_])
        print(f"Selected features: {feature_name_}")
        categorical_features_ = [
            f for f in feature_name_ if f in self.categorical_features
        ]
        self.estimator_ = gpboost_model_train(
            X[feature_name_],
            y,
            group,
            feature_name_,
            categorical_features_,
            self.gpb_params,
            self.num_boost_round,
        )

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def predict(self, X):
        check_is_fitted(self)
        X_test = X.drop("eid", axis="columns")
        group_test = X["eid"]
        selected_features = list(X_test.columns[self.support_])

        pred = self.estimator_.predict(
            data=X_test[selected_features],
            group_data_pred=group_test,
            pred_latent=False,
        )
        y_pred = pred["response_mean"] > 0.5
        return y_pred.reshape(-1, 1)

    def predict_proba(self, X):
        check_is_fitted(self)
        X_test = X.drop("eid", axis="columns")
        group_test = X["eid"]
        selected_features = list(X_test.columns[self.support_])

        pred = self.estimator_.predict(
            data=X_test[selected_features],
            group_data_pred=group_test,
            pred_latent=False,
        )
        pos_probab = pred["response_mean"]
        neg_probab = 1 - pred["response_mean"]
        return np.vstack((neg_probab, pos_probab)).T
