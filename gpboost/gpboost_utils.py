import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as mtc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
import gpboost as gpb
from pdpbox import pdp


####################
# DATA PREPARATION #
####################
def stratified_group_kfold_split_undersampled(X, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    folds = list(cv.split(X, y=X["is_label"], groups=X["eid"]))

    folds_undersampled = []
    for fold_i, (train_index, test_index) in enumerate(folds):
        data_train = X.iloc[train_index, :]
        data_test = X.iloc[test_index, :]
        data_train_undersampled = undersampling_by_group(data_train, seed)
        data_test_undersampled = undersampling_by_group(data_test, seed)
        folds_undersampled.append(
            (
                np.array(data_train_undersampled.index),
                np.array(data_test_undersampled.index),
            )
        )
    return folds, folds_undersampled


def undersampling_by_group(X, seed):
    is_label_true = pd.Series(
        X.groupby("eid").filter(lambda g: g["is_label"].sum() > 0).eid.unique()
    )
    is_label_false = pd.Series(
        X.groupby("eid").filter(lambda g: g["is_label"].sum() == 0).eid.unique()
    )

    min_n_per_class = min(len(is_label_true), len(is_label_false))
    is_label_true_sampled = is_label_true.sample(
        n=min_n_per_class, replace=False, random_state=seed
    )
    is_label_false_sampled = is_label_false.sample(
        n=min_n_per_class, replace=False, random_state=seed
    )
    print(f"Sampled {min_n_per_class} from each label class")

    eids = pd.concat([is_label_true_sampled, is_label_false_sampled], axis=0).tolist()
    assert len(eids) == min_n_per_class * 2
    return X[X.eid.isin(eids)]


def split_dataset(X, seed, exclude_eids=None, balance=True, pct_train=0.8):
    is_label_true = pd.Series(
        X.groupby("eid").filter(lambda g: g["is_label"].sum() > 0).eid.unique()
    )
    is_label_false = pd.Series(
        X.groupby("eid").filter(lambda g: g["is_label"].sum() == 0).eid.unique()
    )

    if balance:
        min_n_per_class = min(len(is_label_true), len(is_label_false))
        is_label_true_sampled = is_label_true.sample(
            n=min_n_per_class, replace=False, random_state=seed
        )
        is_label_false_sampled = is_label_false.sample(
            n=min_n_per_class, replace=False, random_state=seed
        )
        print(f"Sampled {min_n_per_class} from each label class")

        eids = pd.concat(
            [is_label_true_sampled, is_label_false_sampled], axis=0
        ).tolist()
        assert len(eids) == min_n_per_class * 2
        train_eids_is_label_true = is_label_true_sampled.sample(
            frac=pct_train, replace=False, random_state=seed
        )
        train_eids_is_label_false = is_label_false_sampled.sample(
            frac=pct_train, replace=False, random_state=seed
        )
    else:
        eids = pd.concat([is_label_true, is_label_false], axis=0).tolist()
        assert len(eids) == len(X.eid.unique())
        train_eids_is_label_true = is_label_true.sample(
            frac=pct_train, replace=False, random_state=seed
        )
        train_eids_is_label_false = is_label_false.sample(
            frac=pct_train, replace=False, random_state=seed
        )

    train_eids = pd.concat(
        [train_eids_is_label_true, train_eids_is_label_false], axis=0
    ).tolist()
    test_eids = list(set(eids).difference(set(train_eids)))

    test_eids_is_label_true = list(
        set(test_eids).intersection(set(is_label_true.tolist()))
    )
    test_eids_is_label_false = list(
        set(test_eids).intersection(set(is_label_false.tolist()))
    )

    print(
        f"Number of cases in training data: {len(train_eids_is_label_true)} ({len(train_eids_is_label_true)/len(train_eids):.4f})"
    )
    print(
        f"Number of cases in test data: {len(test_eids_is_label_true)} ({len(test_eids_is_label_true)/len(test_eids):.4f})"
    )

    return train_eids, test_eids


####################
# HYP OPTIMISATION #
####################
def gpboost_cv_hyperparam_search(
    X,
    cv_folds,
    param_grid,
    params,
    num_boost_round,
    categorical_features,
    early_stopping_rounds=10,
):
    X_train = X.copy(deep=True)
    y_train = X_train.pop("is_label").to_numpy()
    group_train = X_train.pop("eid")

    data_train = gpb.Dataset(X_train, y_train)
    gp_model = gpb.GPModel(group_data=group_train, likelihood="bernoulli_probit")

    opt_params = gpb.grid_search_tune_parameters(
        param_grid=param_grid,
        params=params,
        folds=cv_folds,
        categorical_feature=categorical_features,
        gp_model=gp_model,
        use_gp_model_for_validation=True,
        train_set=data_train,
        verbose_eval=0,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        seed=params["seed"],
        metrics=params["metric"],
    )
    return opt_params


def gpboost_cv_niters_search(
    X, cv_folds, params, num_boost_round, categorical_features, early_stopping_rounds=10
):
    X_train = X.copy(deep=True)
    y_train = X_train.pop("is_label").to_numpy()
    group_train = X_train.pop("eid")

    data_train = gpb.Dataset(X_train, y_train)
    gp_model = gpb.GPModel(group_data=group_train, likelihood="bernoulli_probit")

    cvbst = gpb.cv(
        params=params,
        train_set=data_train,
        categorical_feature=categorical_features,
        folds=cv_folds,
        gp_model=gp_model,
        use_gp_model_for_validation=True,
        train_gp_model_cov_pars=True,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
        show_stdv=True,
        seed=params["seed"],
        metrics=params["metric"],
    )
    metric_mean = list(cvbst.keys())[0]
    metric_std = list(cvbst.keys())[1]
    best_iter = np.argmax(cvbst[metric_mean])
    print("Best number of iterations: " + str(best_iter))  # 87
    plt.plot(cvbst[metric_mean])
    plt.fill_between(
        range(len(cvbst[metric_mean])),
        np.subtract(cvbst[metric_mean], cvbst[metric_std]),
        np.add(cvbst[metric_mean], cvbst[metric_std]),
        color="lightblue",
    )
    plt.xlabel("Number of iterations")
    plt.ylabel("ROC-AUC")
    plt.show()

    return cvbst


#########
# MODEL #
#########
def run_gpboost(X_train_group, y_train, categorical_features, params, num_boost_round):
    # Define and train GPModel (random effects)
    group_train = X_train_group["eid"]
    X_train = X_train_group.drop("eid", axis="columns")

    gp_model = gpb.GPModel(group_data=group_train, likelihood="bernoulli_probit")
    gp_model.set_optim_params(params={"trace": False})

    # create dataset for gpb.train function
    data_train = gpb.Dataset(X_train, y_train)
    bst = gpb.train(
        params=params,
        train_set=data_train,
        categorical_feature=categorical_features,
        gp_model=gp_model,
        num_boost_round=num_boost_round,
        use_gp_model_for_validation=True,
        train_gp_model_cov_pars=True,
    )
    # Estimated random effects model (variances of random effects)
    # gp_model.summary()
    return bst


def get_model_results(
    bst,
    X_train_group,
    y_train,
    X_test_group,
    y_test,
    categorical_features,
    output_name,
    seed,
    fold_i=None,
    threshold=0.5,
):

    os.makedirs(output_name, exist_ok=True)
    bst.save_model(f"{output_name}/gpboost_model_{seed}_{fold_i}.json")

    group_train = X_train_group["eid"]
    X_train = X_train_group.drop("eid", axis="columns")

    group_test = X_test_group["eid"]
    X_test = X_test_group.drop("eid", axis="columns")

    # Predict response variable (pred_latent=False)
    y_pred_prob_train = bst.predict(
        data=X_train, group_data_pred=group_train, pred_latent=False
    )["response_mean"]
    y_pred_train = y_pred_prob_train > threshold

    y_pred_prob = bst.predict(
        data=X_test, group_data_pred=group_test, pred_latent=False
    )["response_mean"]
    y_pred = y_pred_prob > threshold

    predictions = pd.DataFrame(
        {
            "eid": list(group_test),
            "y_test": list(y_test),
            "y_pred_prob": list(y_pred_prob),
            "y_pred": list(y_pred),
        }
    )
    predictions.to_csv(
        f"{output_name}/gpboost_test_predictions_{seed}_{fold_i}.csv", index=False
    )

    importance_gain = get_feature_importance(
        bst, importance_type="gain", normalize=True, show_plot=False
    )
    importance_split = get_feature_importance(
        bst, importance_type="split", normalize=True, show_plot=False
    )
    importance = pd.merge(
        importance_gain, importance_split, how="outer", on="feature_name"
    )
    importance.to_csv(
        f"{output_name}/gpboost_importance_{seed}_{fold_i}.csv", index=False
    )

    gpboost_results = pd.DataFrame(
        [
            {
                "seed": seed,
                "fold_i": fold_i,
                "train_n_case": np.sum([y_train == 1]),
                "train_n_control": np.sum([y_train == 0]),
                "test_n_case": np.sum([y_test == 1]),
                "test_n_control": np.sum([y_test == 0]),
                "features": list(X_train.columns),
                "categorical_features": categorical_features,
                "prob_threshold": threshold,
                ### test set
                "confusion_matrix": np.array2string(
                    mtc.confusion_matrix(y_test, y_pred), separator=", "
                ),
                "macro_avg_precision": round(
                    mtc.precision_score(y_test, y_pred, average="macro"), 3
                ),
                "macro_avg_recall": round(
                    mtc.recall_score(y_test, y_pred, average="macro"), 3
                ),
                "macro_avg_f1score": round(
                    mtc.f1_score(y_test, y_pred, average="macro"), 3
                ),
                "micro_avg_precision": round(
                    mtc.precision_score(y_test, y_pred, average="micro"), 3
                ),
                "micro_avg_recall": round(
                    mtc.recall_score(y_test, y_pred, average="micro"), 3
                ),
                "micro_avg_f1score": round(
                    mtc.f1_score(y_test, y_pred, average="micro"), 3
                ),
                "accuracy": round(mtc.accuracy_score(y_test, y_pred), 3),
                "balanced_accuracy": round(
                    mtc.balanced_accuracy_score(y_test, y_pred), 3
                ),
                "roc_auc": round(
                    mtc.roc_auc_score(y_test, y_pred_prob, average=None), 3
                ),
                "pc_auc": round(mtc.average_precision_score(y_test, y_pred_prob), 3),
                ### train set
                "confusion_matrix_train": np.array2string(
                    mtc.confusion_matrix(y_train, y_pred_train), separator=", "
                ),
                "macro_avg_precision_train": round(
                    mtc.precision_score(y_train, y_pred_train, average="macro"), 3
                ),
                "macro_avg_recall_train": round(
                    mtc.recall_score(y_train, y_pred_train, average="macro"), 3
                ),
                "macro_avg_f1score_train": round(
                    mtc.f1_score(y_train, y_pred_train, average="macro"), 3
                ),
                "micro_avg_precision_train": round(
                    mtc.precision_score(y_train, y_pred_train, average="micro"), 3
                ),
                "micro_avg_recall_train": round(
                    mtc.recall_score(y_train, y_pred_train, average="micro"), 3
                ),
                "micro_avg_f1score_train": round(
                    mtc.f1_score(y_train, y_pred_train, average="micro"), 3
                ),
                "accuracy_train": round(mtc.accuracy_score(y_train, y_pred_train), 3),
                "balanced_accuracy_train": round(
                    mtc.balanced_accuracy_score(y_train, y_pred_train), 3
                ),
                "roc_auc_train": round(
                    mtc.roc_auc_score(y_train, y_pred_prob_train, average=None), 3
                ),
                "pc_auc_train": round(
                    mtc.average_precision_score(y_train, y_pred_prob_train), 3
                ),
            }
        ]
    )
    fname = f"{output_name}/gpboost_test_results.csv"
    if os.path.exists(fname):
        gpboost_results.to_csv(fname, mode="a", index=False, header=False)
    else:
        gpboost_results.to_csv(fname, index=False, header=True)

    return importance, gpboost_results


######################
# FEATURE IMPORTANCE #
######################
def get_feature_importance(
    booster,
    importance_type="split",
    max_num_features=None,
    normalize=False,
    show_plot=False,
):
    """
    importance_type : string, optional (default="split")
        How the importance is calculated.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
        (i.e. how much it reduces impurity in a split)
        These measures are calculated using training data.
    max_num_features : int or None, optional (default=None)
        Max number of top features displayed on plot.
        If None or <1, all features will be displayed.
    """
    importance = booster.feature_importance(importance_type=importance_type)
    if normalize:
        importance = importance / importance.sum()
    feature_names = booster.feature_name()

    colname_imp = f"feature_importance_{importance_type}"
    feature_importance = pd.DataFrame(
        list(zip(feature_names, importance)), columns=["feature_name", colname_imp]
    )
    feature_importance = feature_importance.sort_values(
        by=[colname_imp], ascending=False
    )
    if max_num_features is not None and max_num_features > 0:
        feature_importance = feature_importance.head(max_num_features)

    if show_plot:
        gpb.plot_importance(booster)
        plt.show()

    return feature_importance


def summarize_importance_results(files):
    results_gain = pd.DataFrame()
    results_split = pd.DataFrame()

    for f in files:
        match = re.search("importance_(.*)\.csv", f)
        seed = match.group(1)

        df_imp = pd.read_csv(f, index_col="feature_name")
        df_imp_gain = df_imp.loc[:, ["feature_importance_gain"]]
        df_imp_split = df_imp.loc[:, ["feature_importance_split"]]
        df_imp_gain.rename(columns={"feature_importance_gain": seed}, inplace=True)
        df_imp_split.rename(columns={"feature_importance_split": seed}, inplace=True)

        if any(results_gain):
            results_gain = pd.merge(
                results_gain, df_imp_gain, left_index=True, right_index=True
            )
            results_split = pd.merge(
                results_split, df_imp_split, left_index=True, right_index=True
            )
        else:
            results_gain = df_imp_gain.copy(deep=True)
            results_split = df_imp_split.copy(deep=True)

    results_gain = pd.concat(
        [
            results_gain,
            results_gain.apply(
                lambda x: pd.Series(
                    {
                        "mean": x.mean(),
                        "std": x.std(),
                        "median": x.median(),
                        "min": x.min(),
                        "max": x.max(),
                    }
                ),
                axis=1,
            ),
        ],
        axis=1,
    )

    results_split = pd.concat(
        [
            results_split,
            results_split.apply(
                lambda x: pd.Series(
                    {
                        "mean": x.mean(),
                        "std": x.std(),
                        "median": x.median(),
                        "min": x.min(),
                        "max": x.max(),
                    }
                ),
                axis=1,
            ),
        ],
        axis=1,
    )

    results_gain.sort_values("mean", ascending=False, inplace=True)
    results_split.sort_values("mean", ascending=False, inplace=True)

    return results_gain, results_split


##################
# VISUALISATIONS #
##################
def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, threshold = mtc.roc_curve(y_test, y_pred)
    score = mtc.roc_auc_score(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.plot(
        false_positive_rate, true_positive_rate, label=f"ROC-AUC = {round(score,3)}"
    )
    plt.title("Receiver Operating Characteristic")
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7")
    plt.plot([1, 1], c=".7")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.show()

    optimal_idx = np.argmax(true_positive_rate - false_positive_rate)
    optimal_threshold = threshold[optimal_idx]
    print("Optimal threshold value is:", optimal_threshold)


def plot_precision_recall_curve(y_test, y_pred):
    precision, recall, thresholds = mtc.precision_recall_curve(y_test, y_pred)
    # average_precision = mtc.average_precision_score(y_test, y_pred)
    auc_pr = mtc.auc(recall, precision)
    print(f"AUC for Precision-Recall: {auc_pr}")

    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    plt.plot(recall, precision, marker=".", label="Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


def plot_calibration_curve(y_test, y_pred):
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=30)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Not calibrated")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")


def plot_pdp(
    X,
    bst,
    feature_to_plot,
    feature_description,
    plot_params={},
    figsize=(6, 5),
    save_plot=False,
    output_name="",
    dpi=300,
):
    features = bst.feature_name()
    pdp_dist = pdp.pdp_isolate(
        model=bst,
        dataset=X.dropna(subset=[feature_to_plot]),
        model_features=features,
        feature=feature_to_plot,
        num_grid_points=10,
        grid_type="percentile",
        predict_kwds={"ignore_gp_model": True},
    )
    fig, axes = pdp.pdp_plot(
        pdp_dist,
        feature_description,
        figsize=figsize,
        plot_params=plot_params,
    )

    if save_plot:
        plt.savefig(
            f"{output_name}/plots/pdp_{feature_to_plot}.png",
            dpi=dpi,
            bbox_inches="tight",
        )


def plot_pdp_interact(
    X,
    bst,
    feature_to_plot,
    feature_description,
    plot_params={},
    figsize=(6, 5),
    save_plot=False,
    output_name="",
    dpi=300,
):
    features = bst.feature_name()
    inter_rf = pdp.pdp_interact(
        model=bst,
        dataset=X.dropna(subset=feature_to_plot),
        model_features=features,
        features=feature_to_plot,
        predict_kwds={"ignore_gp_model": True},
    )

    pdp.pdp_interact_plot(
        inter_rf,
        feature_description,
        figsize=figsize,
        x_quantile=True,
        plot_type="contour",
        plot_pdp=True,
        plot_params=plot_params,
    )

    if save_plot:
        feat_names_joined = "_".join(feature_to_plot)
        plt.savefig(
            f"{output_name}/plots/pdp_inter_{feat_names_joined}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
