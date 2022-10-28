import logging
import warnings
import pandas as pd
import numpy as np


def read_csv(file_path):
    df = pd.read_csv(file_path, header=0, low_memory=False)
    return df


def remove_not_consented_participants(df):
    col_consent_date = "200-0.0"

    idx = df.index[pd.isnull(df[col_consent_date])]
    logging.info(
        f"{len(idx)} people did not provide consent or consent date is not available"
    )
    df_consented = df.drop(idx, inplace=False)
    logging.info(f"{len(df_consented)} rows left after removing non-consenters")
    return df_consented


def remove_consent_withdrawals(df):
    col_reason_lost = "190-0.0"  # reason_lost
    withdrawal_val = 5  # Participant has withdrawn consent for future linkage

    idx = df.index[df[col_reason_lost].isin([withdrawal_val])]
    logging.info(f"{len(idx)} people withdrew consent")
    df_nonwithdrawals = df.drop(idx, inplace=False)
    logging.info(f"{len(df_nonwithdrawals)} rows left after removing withdrawals")
    return df_nonwithdrawals


def get_subset_of_rows(df, arg, random_state=0):
    if isinstance(arg, int):
        df_subset = df.sample(n=arg, replace=False, random_state=random_state)
    elif isinstance(arg, float):
        df_subset = df.sample(frac=arg, replace=False, random_state=random_state)
    logging.info(f"{len(df_subset)} rows left after sampling")
    return df_subset


def get_subset_of_rows_stratified(df, frac, stratify_key="label_class", random_state=0):
    if not isinstance(frac, float):
        raise TypeError("The frac argument {} should be of float type.")

    df_subset = df.groupby(stratify_key, group_keys=False).apply(
        lambda x: x.sample(frac=frac, replace=False, random_state=random_state)
    )
    logging.info(f"{len(df_subset)} rows left after sampling")
    return df_subset


def get_subset_of_rows_balanced(df, stratify_key="label_class", random_state=0):
    count_dict = df[stratify_key].value_counts().to_dict()
    nrows_to_sample = min(count_dict.values())

    df_subset = df.groupby(stratify_key, group_keys=False).apply(
        lambda x: x.sample(n=nrows_to_sample, replace=False, random_state=random_state)
    )
    logging.info(f"{len(df_subset)} rows left after sampling")
    return df_subset


def aggregate_repeat_measurements(df, field_ids, visit_id=0, drop_repeats_cols=False):
    """
    Average multiple measurements of the vital signs within the same visit.
    Ignores null values.
    Name of the aggregated column follows <field_id>-<visit_id> structure.
    """
    df_agg = df.copy(deep=True)
    colnames = df_agg.columns

    for field_id in field_ids:
        repeat_cols = [
            col for col in colnames if col.startswith(f"{field_id}-{visit_id}")
        ]
        if not repeat_cols:
            raise ValueError(
                f"There are no columns starting with '{field_id}-{visit_id}'"
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", message="Mean of empty slice")
                df_agg[f"{field_id}-{visit_id}"] = np.nanmean(
                    df_agg[repeat_cols], axis=1
                )

            if drop_repeats_cols:
                df_agg.drop(repeat_cols, axis="columns", inplace=True)

            logging.info(
                f"Aggregated repeat measurements {repeat_cols} in a new column named '{field_id}-{visit_id}'"
            )
    return df_agg


def aggregate_visits_for_static_col(
    df, field_id, method="most_common", drop_cols=False
):
    df_agg = df.copy(deep=True)
    field_cols = [col for col in df_agg.columns if col.startswith(f"{field_id}-")]

    if method == "most_common":
        df_agg[str(field_id)] = (
            df_agg.filter(like=f"{field_id}-").mode(axis=1, dropna=True).iloc[:, 0]
        )
    elif method == "min":
        df_agg[str(field_id)] = df_agg.filter(like=f"{field_id}-").min(
            axis=1, skipna=True
        )
    elif method == "max":
        df_agg[str(field_id)] = df_agg.filter(like=f"{field_id}-").max(
            axis=1, skipna=True
        )
    else:
        raise ValueError(f"Method {method} not available")

    if drop_cols:
        df_agg.drop(field_cols, axis="columns", inplace=True)

    logging.info(f"Aggregated columns {field_cols} in a new column named '{field_id}'")
    return df_agg


def aggregate_columns(df, field_ids, method="sum", drop_cols=False):
    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    field_ids = [str(field_id) for field_id in field_ids]
    visit_ids = [
        col.split("-")[1] for col in colnames if col.startswith(f"{field_ids[0]}-")
    ]

    for visit_id in visit_ids:
        cols_to_agg = ["-".join([field_id, visit_id]) for field_id in field_ids]
        new_field_name = "_".join(["_".join(field_ids), method])
        if method == "sum":
            df_agg[f"{new_field_name}-{visit_id}"] = df_agg[cols_to_agg].sum(
                min_count=len(cols_to_agg), axis=1
            )
        elif method == "mean":
            df_agg[f"{new_field_name}-{visit_id}"] = df_agg[cols_to_agg].mean(
                skipna=True, axis=1
            )
        elif method == "max":
            df_agg[f"{new_field_name}-{visit_id}"] = df_agg[cols_to_agg].max(
                skipna=True, axis=1
            )
        else:
            raise ValueError(f"Method {method} is not available.")

        if drop_cols:
            df_agg.drop(cols_to_agg, axis="columns", inplace=True)

        logging.info(
            f"Aggregated columns {cols_to_agg} in a new column named '{new_field_name}-{visit_id}'"
        )
    return df_agg


def mask_vals_as_nan_in_repeat_col(df, field_id, val):
    df_agg = df.copy(deep=True)
    field_cols = [col for col in df_agg.columns if col.startswith(f"{field_id}-")]
    for field_col in field_cols:
        df_agg[str(field_col)].replace(val, np.NaN, inplace=True)
    return df_agg


def set_type_as_datetime(df, field_id):
    df_agg = df.copy(deep=True)
    field_cols = [col for col in df_agg.columns if col.startswith(f"{str(field_id)}-")]
    df_agg[field_cols] = df_agg[field_cols].apply(pd.to_datetime, errors="coerce")
    return df_agg


def group_ethnicity_codes(df):
    return df.replace(
        {
            "21000": {
                (-1): np.NaN,
                (-3): np.NaN,
                1: 1,
                1001: 1,
                1002: 1,
                1003: 1,  # White
                2: 2,
                2001: 2,
                2002: 2,
                2003: 2,
                2004: 2,  # Mixed
                3: 3,
                3001: 3,
                3002: 3,
                3003: 3,
                3004: 3,
                5: 3,  # Asian
                4: 4,
                4001: 4,
                4002: 4,
                4003: 4,  # Black
                6: 6,  # Other
            }
        }
    )


def compute_age_at_visit(df, drop_col=True):
    """
    Computes age at each visit
    """
    df_age = df.copy(deep=True)
    col_age_at_recr = "21022-0.0"
    cols_visit_date = [col for col in df_age.columns if col.startswith("53-")]
    df_age[cols_visit_date] = df_age[cols_visit_date].apply(pd.to_datetime)

    for col_visit_date in cols_visit_date:
        visit_id = col_visit_date.split("-")[1]
        col_name = f"age-{visit_id}"

        if visit_id == "0.0":
            df_age[col_name] = df_age.loc[:, col_age_at_recr]
        else:
            df_age[col_name] = df_age.apply(
                lambda x: x[col_age_at_recr]
                + int((x[col_visit_date] - x["53-0.0"]).days / 365)
                if (
                    (pd.isnull(x[col_visit_date]) == False)
                    and (pd.isnull(x["53-0.0"]) == False)
                )
                else np.NaN,
                axis=1,
            )
        logging.info(
            f"Computed age at visit {visit_id} in a new column named '{col_name}'"
        )

    if drop_col:
        df_age.drop(col_age_at_recr, axis="columns", inplace=True)

    return df_age


def has_family_history(df, disease="cancer", drop_cols=True):
    if disease == "cancer":
        disease_codes = [3, 4, 5, 13]
    elif disease == "colorectal_cancer":
        disease_codes = [4]
    elif disease == "diabetes":
        disease_codes = [9]
    elif disease == "hypertension":
        disease_codes = [8]
    else:
        raise ValueError(f"Disease {disease} not available.")

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    field_ids = ["20107", "20110", "20111"]
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_ids[0]}-")
        ]
    )

    for visit_id in visit_ids:
        family_hist_visit_cols = [
            col
            for col in colnames
            if (col.startswith(tuple(field_ids)))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"famhist_{disease}-{visit_id}"

        df_agg[new_field_name] = (
            df_agg[family_hist_visit_cols].isin(disease_codes).any(axis=1)
        )
        df_agg[family_hist_visit_cols] = df_agg[family_hist_visit_cols].replace(
            [-11, -13, -21, -23], np.NaN
        )  # do not know and prefer not to answer --> np.NaN
        df_agg[new_field_name].mask(
            df_agg[family_hist_visit_cols].isna().all(axis=1), np.NaN, inplace=True
        )  # if all values are nan, convert it to NaN (instead of False)

        if drop_cols:
            df_agg.drop(family_hist_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated family history columns for disease {disease} - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def has_disease_history_touchscreen(df, disease, list_format=True, drop_cols=True):
    def check_disease_codes_from_list(x, disease_codes, field_id):
        if np.isnan(x).all():
            return np.nan
        else:
            if ((-1) in x) or ((-3) in x):
                return np.nan
            elif (-7) in x:
                return False
            elif any(code in x for code in disease_codes[field_id]):
                return True

    if disease == "cardiovascular":
        disease_codes = {"6150": [1, 2, 3, 4]}
    elif disease == "diabetes":
        disease_codes = {"2443": [1]}
    else:
        raise ValueError(f"Disease {disease} not available.")
    field_ids = list(disease_codes.keys())

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_ids[0]}-")
        ]
    )

    for visit_id in visit_ids:
        disease_visit_cols = [
            col
            for col in colnames
            if (col.startswith(tuple(field_ids)))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"diseasehist_{disease}-{visit_id}"
        if list_format:
            assert (
                len(field_ids) == 1
            ), "List_format option currently only supports single entry for field id"
            assert (
                len(disease_visit_cols) == 1
            ), "If list_format is True, field id should have only one array_id"
            df_agg[new_field_name] = df_agg[disease_visit_cols[0]].apply(
                check_disease_codes_from_list,
                disease_codes=disease_codes,
                field_id=field_ids[0],
            )
        else:
            disease_codes_v = {
                col: disease_codes[col.split("-")[0]] for col in disease_visit_cols
            }
            df_agg[new_field_name] = (
                df_agg[disease_visit_cols].isin(disease_codes_v).any(axis=1)
            )
            df_agg[disease_visit_cols] = df_agg[disease_visit_cols].replace(
                [-1, -3], np.NaN
            )  # do not know and prefer not to answer --> np.NaN
            df_agg[new_field_name].mask(
                df_agg[disease_visit_cols].isna().all(axis=1), np.NaN, inplace=True
            )  # if all values are nan, convert it to NaN (instead of False)

        if drop_cols:
            df_agg.drop(disease_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated disease columns for {disease} - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def has_disease_history_verbalinterview(df, disease, list_format=True, drop_cols=True):
    def check_disease_codes_from_list(x, disease_codes):
        if np.isnan(x).all():
            return np.nan
        else:
            return any(code in x for code in disease_codes)

    if disease == "ibd":
        disease_codes = [1461, 1462, 1463]
    elif disease == "hepatitis":
        disease_codes = [1155, 1156, 1157, 1578, 1579, 1580, 1581, 1582]
    elif disease == "anyliverbiliary":
        disease_codes = [
            1155,
            1156,
            1157,
            1578,
            1579,
            1580,
            1581,
            1582,
            1158,
            1506,
            1604,  # cirrhosis
            1159,
            1160,
            1161,
            1162,
            1163,
            1475,  # bile duct and gallbladder
        ]
    else:
        raise ValueError(f"Disease {disease} not available.")
    field_id = "20002"

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_id}-")
        ]
    )

    for visit_id in visit_ids:
        disease_visit_cols = [
            col
            for col in colnames
            if (col.startswith(field_id))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"diseasehist_{disease}-{visit_id}"
        if list_format:
            assert (
                len(disease_visit_cols) == 1
            ), "If list_format is True, field id should have only one array_id"
            df_agg[new_field_name] = df_agg[disease_visit_cols[0]].apply(
                check_disease_codes_from_list, disease_codes=disease_codes
            )
        else:
            raise ValueError("This has not been tested yet.")

        if drop_cols:
            df_agg.drop(disease_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated disease columns for {disease} - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def using_regular_medication(df, medication, drop_cols=True):
    if medication == "aspirin":
        medication_codes = {"6154": [1], "10004": [1]}  # second one pilot study
        field_ids = list(medication_codes.keys())
    else:
        raise ValueError(f"Medication {medication} not available.")

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_ids[0]}-")
        ]
    )

    for visit_id in visit_ids:
        med_visit_cols = [
            col
            for col in colnames
            if (col.startswith(tuple(field_ids)))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"regular_{medication}-{visit_id}"
        medication_codes_v = {
            col: medication_codes[col.split("-")[0]] for col in med_visit_cols
        }

        df_agg[new_field_name] = (
            df_agg[med_visit_cols].isin(medication_codes_v).any(axis=1)
        )
        df_agg[med_visit_cols] = df_agg[med_visit_cols].replace(
            [-1, -3], np.NaN
        )  # do not know and prefer not to answer --> np.NaN
        df_agg[new_field_name].mask(
            df_agg[med_visit_cols].isna().all(axis=1), np.NaN, inplace=True
        )  # if all values are nan, convert it to NaN (instead of False)

        if drop_cols:
            df_agg.drop(med_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated medication columns for {medication} - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def using_regular_medication_specific(df, medication, drop_cols=True):
    if medication == "aspirin":
        medication_codes = [
            1140861806,
            1140864860,
            1140868226,
            1140868282,
            1140872040,
            1140882108,
            1140882190,
            1140882268,
            1140882392,
            1141163138,
            1141164044,
            1141167844,
        ]
    elif medication == "statin":
        # simvastatin, fluvastatin, pravastatin, eptastatin, velastatin, atorvastatin, rosuvastatin
        # ecostatin, sandostatin, cilastatin, nystatin not included
        medication_codes = [
            1140861958,
            1140888594,
            1140888648,
            1140910632,
            1140910654,
            1141146234,
            141192410,
        ]
    else:
        raise ValueError(f"Medication {medication} not available.")

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    field_id = "20003"
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_id}-")
        ]
    )

    for visit_id in visit_ids:
        med_visit_cols = [
            col
            for col in colnames
            if (col.startswith(field_id))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"regular_{medication}-{visit_id}"
        # TODO: check the touchscreen medication use questions to see if they answered this question, if -1 or -3 or na, then mask the value as nan
        df_agg[new_field_name] = (
            df_agg[med_visit_cols].isin(medication_codes).any(axis=1)
        )

        if drop_cols:
            df_agg.drop(med_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated medication columns for {medication} - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def get_education_level(df, level="university", drop_cols=True):
    if level == "university":
        edu_codes = [1]
    else:
        raise ValueError(f"Education level {level} not available.")

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    field_ids = ["6138", "10722"]  # second one pilot study
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_ids[0]}-")
        ]
    )

    for visit_id in visit_ids:
        edu_visit_cols = [
            col
            for col in colnames
            if (col.startswith(tuple(field_ids)))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"edu_{level}-{visit_id}"

        df_agg[new_field_name] = df_agg[edu_visit_cols].isin(edu_codes).any(axis=1)
        df_agg[edu_visit_cols] = df_agg[edu_visit_cols].replace(
            [-3], np.NaN
        )  #  prefer not to answer --> np.NaN
        df_agg[new_field_name].mask(
            df_agg[edu_visit_cols].isna().all(axis=1), np.NaN, inplace=True
        )  # if all values are nan, convert it to NaN (instead of False)

        if drop_cols:
            df_agg.drop(edu_visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated education columns for {level} level - visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def calculate_redmeat_intake(df, drop_cols=True):
    cat_to_num = {0: 0, 1: 0.5, 2: 1, 3: 3, 4: 5.5, 5: 7}

    def num_to_cat(x):
        if x == 0:  # never
            return 0
        elif x <= 0.5:  # less than once a week
            return 1
        elif x > 0.5 and x < 1.5:  # once a week
            return 2
        elif x >= 1.5 and x < 4.5:  # 2-4 times a week
            return 3
        elif x >= 4.5 and x < 6.5:  # 5-6 times a week
            return 4
        elif x >= 6.5:  # once or more daily
            return 5
        else:
            return np.nan

    df_agg = df.copy(deep=True)
    colnames = df_agg.columns
    field_ids = ["1349", "1369", "1379", "1389"]
    visit_ids = np.unique(
        [
            col.split("-")[1].split(".")[0]
            for col in colnames
            if col.startswith(f"{field_ids[0]}-")
        ]
    )

    for visit_id in visit_ids:
        visit_cols = [
            col
            for col in colnames
            if (col.startswith(tuple(field_ids)))
            and (col.split("-")[1].split(".")[0] == visit_id)
        ]
        new_field_name = f"redmeat_intake-{visit_id}"

        df_agg[visit_cols] = df_agg[visit_cols].replace(
            [-1, -3], np.NaN
        )  # do not know and prefer not to answer --> np.NaN
        df_agg[new_field_name] = df_agg[visit_cols].apply(
            lambda x: np.nan
            if np.isnan(x.tolist()).all()
            else np.sum([cat_to_num[l] for l in x.tolist() if pd.isnull(l) == False]),
            axis=1,
        )
        df_agg[new_field_name] = df_agg[new_field_name].apply(num_to_cat)

        if drop_cols:
            df_agg.drop(visit_cols, axis="columns", inplace=True)

        logging.info(
            f"Aggregated processed and red meat columns for visit {visit_id} in a new column named '{new_field_name}'"
        )
    return df_agg


def is_fasted_before_assessment(df, fast_min_hour=8, drop_cols=True):
    df_agg = df.copy(deep=True)
    cols_fast_time = [col for col in df_agg.columns if col.startswith("74-")]

    for col_fast_time in cols_fast_time:
        visit_id = col_fast_time.split("-")[1]
        col_name = f"fasted-{visit_id}"
        df_agg[col_name] = df_agg[col_fast_time].apply(
            lambda x: np.nan if pd.isnull(x) else x >= fast_min_hour
        )
        logging.info(
            f"Computed fasting status for min {fast_min_hour} hours at visit {visit_id} in a new column named '{col_name}'"
        )

    if drop_cols:
        df_agg.drop(cols_fast_time, axis="columns", inplace=True)
    return df_agg


def compute_waist_to_hip_ratio(df, drop_cols=True):
    df_agg = df.copy(deep=True)
    col_waist = "48"
    col_hip = "49"
    visit_ids = np.unique(
        [col.split("-")[1] for col in df_agg.columns if col.startswith(f"{col_waist}-")]
    )

    for visit_id in visit_ids:
        visit_waist_col = f"{col_waist}-{visit_id}"
        visit_hip_col = f"{col_hip}-{visit_id}"

        col_name = "-".join(["whr", visit_id])
        df_agg[col_name] = df_agg[[visit_waist_col, visit_hip_col]].apply(
            lambda x: np.nan
            if (pd.isnull(x[visit_waist_col]))
            or (pd.isnull(x[visit_hip_col]))
            or x[visit_hip_col] == 0
            else round(x[visit_waist_col] / x[visit_hip_col], 3),
            axis=1,
        )
        logging.info(
            f"Computed waist to hip ratio at visit {visit_id} in a new column named '{col_name}'"
        )

        if drop_cols:
            df_agg.drop([visit_waist_col, visit_hip_col], axis="columns", inplace=True)
    return df_agg


def compute_trunk_to_leg_fat_ratio(df, drop_cols=True):
    df_agg = df.copy(deep=True)
    col_trunkfat = "23127"
    col_legfat = "23111_23115_sum"
    visit_ids = np.unique(
        [
            col.split("-")[1]
            for col in df_agg.columns
            if col.startswith(f"{col_trunkfat}-")
        ]
    )

    for visit_id in visit_ids:
        visit_trunkfat_col = f"{col_trunkfat}-{visit_id}"
        visit_legfat_col = f"{col_legfat}-{visit_id}"

        col_name = "-".join(["tlr", visit_id])
        df_agg[col_name] = df_agg[[visit_trunkfat_col, visit_legfat_col]].apply(
            lambda x: np.nan
            if (pd.isnull(x[visit_trunkfat_col]))
            or (pd.isnull(x[visit_legfat_col]))
            or x[visit_legfat_col] == 0
            else round(x[visit_trunkfat_col] / x[visit_legfat_col], 3),
            axis=1,
        )
        logging.info(
            f"Computed trunk to leg fat ratio at visit {visit_id} in a new column named '{col_name}'"
        )

        if drop_cols:
            df_agg.drop(
                [visit_trunkfat_col, visit_legfat_col], axis="columns", inplace=True
            )
    return df_agg


def alcohol_consumption_categorize(x, col_alco_status, col_alco_freq):
    if pd.isnull(x[col_alco_status]):
        return np.nan
    elif x[col_alco_status] == 0:
        return 0  # "never"
    elif x[col_alco_status] == 1:
        return 1  # "former"
    elif x[col_alco_status] == 2:
        if x[col_alco_freq] == 5:
            return 2  # "current_occasional"
        elif x[col_alco_freq] == 4:
            return 3  # "current_13permo"
        elif x[col_alco_freq] == 3:
            return 4  # "current_12perweek"
        elif x[col_alco_freq] == 2:
            return 5  # "current_34perweek"
        elif x[col_alco_freq] == 1:
            return 6  # "current_57perweek"
        else:
            raise ValueError("Unknown value for alcohol consumption frequency")
    else:
        raise ValueError("Unknown value for alcohol consumption status")


def smoking_categorize(x, col_smoke_status, col_smoke_freq):
    if pd.isnull(x[col_smoke_status]):
        return np.nan
    elif x[col_smoke_status] == 0:
        return 0  # "never"
    elif x[col_smoke_status] == 1:
        return 1  # "former"
    elif x[col_smoke_status] == 2:
        if pd.isnull(x[col_smoke_freq]):
            return 4  # "current_unk"
        elif x[col_smoke_freq] < 15:
            return 2  # "current_lt15"
        elif x[col_smoke_freq] >= 15:
            return 3  # "current_gt15"
        else:
            raise ValueError("Unknown value for smoking frequency")
    else:
        raise ValueError("Unknown value for smoking status")


def combine_alcohol_status_and_frequency(df):
    df_agg = df.copy(deep=True)
    col_alco_status = "20117"
    col_alco_freq = "1558"

    visit_ids = np.unique(
        [
            col.split("-")[1]
            for col in df_agg.columns
            if col.startswith(f"{col_alco_status}-")
        ]
    )

    for visit_id in visit_ids:
        visit_alco_status_col = f"{col_alco_status}-{visit_id}"
        visit_alco_freq_col = f"{col_alco_freq}-{visit_id}"

        col_name = "-".join(["alcohol", visit_id])
        df_agg[col_name] = df_agg.apply(
            alcohol_consumption_categorize,
            col_alco_status=visit_alco_status_col,
            col_alco_freq=visit_alco_freq_col,
            axis=1,
        )

        logging.info(
            f"Combined alcohol status and frequency for visit {visit_id} in a new column named '{col_name}'"
        )
    return df_agg


def combine_smoking_status_and_frequency(df):
    df_agg = df.copy(deep=True)
    col_smoke_status = "20116"
    col_smoke_freq = "3456"  # n cigarettes daily

    visit_ids = np.unique(
        [
            col.split("-")[1]
            for col in df_agg.columns
            if col.startswith(f"{col_smoke_status}-")
        ]
    )

    for visit_id in visit_ids:
        visit_smoke_status_col = f"{col_smoke_status}-{visit_id}"
        visit_smoke_freq_col = f"{col_smoke_freq}-{visit_id}"

        col_name = "-".join(["smoke", visit_id])
        df_agg[col_name] = df_agg.apply(
            smoking_categorize,
            col_smoke_status=visit_smoke_status_col,
            col_smoke_freq=visit_smoke_freq_col,
            axis=1,
        )

        logging.info(
            f"Combined smoking status and frequency for visit {visit_id} in a new column named '{col_name}'"
        )
    return df_agg
