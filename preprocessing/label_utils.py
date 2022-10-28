import logging
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


def get_icd10_codes(df, cols_cancer_registry_icd10_code):
    return pd.unique(
        [
            v
            for v in df[cols_cancer_registry_icd10_code].values.ravel("K")
            if pd.isnull(v) == False
        ]
    )


def aggregate_cancer_code_idx_info(df, cols_cancer_registry_icd10_code):
    """
    For each participant, aggregates registry id and diagnosis date per cancer code
    """
    code_date_maps = []
    for ind, row in df.iterrows():
        code_date_map = {}
        for col_code in cols_cancer_registry_icd10_code:
            if pd.isnull(row[col_code]) == False:
                code = row[col_code]

                idx = col_code.split("-")[1]
                col_date = "-".join(["40005", idx])
                col_cancer_age = "-".join(["40008", idx])
                col_behaviour = "-".join(["40012", idx])
                col_histology = "-".join(["40011", idx])
                d_info = {
                    float(idx): {
                        "date": row[col_date],
                        "age": row[col_cancer_age],
                        "behaviour": row[col_behaviour],
                        "histology": row[col_histology],
                    }
                }

                if code_date_map.get(code, None):
                    code_date_map[code].update(d_info)
                else:
                    code_date_map[code] = d_info
        code_date_maps.append(code_date_map)
    return code_date_maps


def compute_n_cancer_occurred_until_last_visit(x, cols_visit_date):
    last_visit_date = (
        np.NaN if pd.isnull(x[cols_visit_date]).all() else max(x[cols_visit_date])
    )
    if pd.isnull(last_visit_date):
        return np.NaN
    else:
        icd_codes_before_last_visit = [
            code
            for code, vals in x["icd_code_map"].items()
            if min([vvals["date"] for _, vvals in vals.items()]) < last_visit_date
        ]
        return len(icd_codes_before_last_visit)


def compute_label_first_occurred_date(x, icd10_codes_selected):
    dates = []
    for code, vals in x.items():
        if code in icd10_codes_selected:
            code_date = min([vvals["date"] for _, vvals in vals.items()])
            dates.append(code_date)
    return min(dates) if dates else np.nan


def compute_label_first_occurred_code(x, icd10_codes_selected):
    code_f = np.nan
    for code, vals in x["icd_code_map"].items():
        if code in icd10_codes_selected:
            dates = [vvals["date"] for _, vvals in vals.items()]
            if x["label_first_occurred_date"] in dates:
                code_f = code
    return code_f


def compute_label_first_occurred_age(x):
    age = np.nan
    for code, vals in x["icd_code_map"].items():
        if code == x["label_first_occurred_code"]:
            for idx, vvals in vals.items():
                if vvals["date"] == x["label_first_occurred_date"]:
                    age = vvals["age"]
                    return age
    return age


def compute_first_occurrence_of_selected_cancer(
    df, cols_cancer_registry_icd10_code, icd10_codes_selected
):
    # Filter to cancer cases only before we aggregate information, otherwise it takes too long
    df_cases = df[
        df["icd_codes"].apply(lambda x: len(x) > 0)
    ]  # get those who had a cancer diagnosis at least one

    df_cases["icd_code_map"] = aggregate_cancer_code_idx_info(
        df_cases, cols_cancer_registry_icd10_code
    )

    df_cases["label_first_occurred_date"] = df_cases["icd_code_map"].apply(
        compute_label_first_occurred_date, icd10_codes_selected=icd10_codes_selected
    )

    df_cases["label_first_occurred_code"] = df_cases.apply(
        compute_label_first_occurred_code,
        icd10_codes_selected=icd10_codes_selected,
        axis=1,
    )

    df_cases["label_first_occurred_age"] = df_cases.apply(
        compute_label_first_occurred_age, axis=1
    )

    df_full = pd.merge(
        df,
        df_cases[
            [
                "eid",
                "icd_code_map",
                "label_first_occurred_code",
                "label_first_occurred_date",
                "label_first_occurred_age",
            ]
        ],
        how="left",
        on="eid",
    )
    df_full.replace({"icd_code_map": {np.NaN: {}}}, inplace=True)
    return df_full


def compute_is_label_at_visit(x, col_visit_date):
    # if the visit date is NA, then is_label should be NA
    if pd.isnull(x[col_visit_date]):
        return np.NaN
    else:
        # visit date is not null, but label first occurred date is NA, then is_label should be False
        if pd.isnull(x["label_first_occurred_date"]):
            return False
        # if visit date is not null and there is a first occurrence date of label, then compare dates
        else:
            return x["label_first_occurred_date"] < x[col_visit_date]


def compute_label_tumour_behaviour_at_visit(x, col_visit_date, icd10_codes_selected):
    """
    For a given cancer type, extracts reported cancers that occurred before the visit date and corresponding tumour behaviour.
    If different tumour behaviour is reported across different timepoints before the visit, it takes the closest one to the visit date.
    If different tumour behaviour is reported across different ICD codes, it takes the worst behaviour.
    """
    if pd.isnull(x[col_visit_date]):
        return np.NaN
    elif pd.isnull(x["label_first_occurred_date"]):
        return np.NaN
    else:
        tbehaviours = []
        for code, vals in x["icd_code_map"].items():
            if code in icd10_codes_selected:
                tbehaviour = {
                    vvals["behaviour"]: vvals["date"]
                    for _, vvals in vals.items()
                    if vvals["date"] < x[col_visit_date]
                }
                if tbehaviour:
                    tbehaviour = max(
                        tbehaviour, key=tbehaviour.get
                    )  # closest one to visit date (before visit date)
                    tbehaviours.append(tbehaviour)
        return max(tbehaviours) if tbehaviours else np.NaN


def assign_label_class(x):
    is_label_at_recruitment_col = [
        col
        for col in x.index
        if col == "is_label_at_recruitment" or col.startswith("is_label-0")
    ][0]
    # If did not receive diagnosis for any cancer type throughout data collection period
    if len(x["icd_codes"]) == 0:
        return 0
    # If diagnosed with selected cancer type prior to recruitment
    elif x[is_label_at_recruitment_col] == True:
        return 1
    # If diagnosed with selected cancer but later in the study
    elif (
        pd.isnull(x["label_first_occurred_date"]) == False
        and x[is_label_at_recruitment_col] == False
    ):
        return 2
    # If diagnosed with cancer other than selected cancer type
    else:
        return 3


def compute_first_occurrence_of_other_cancer(x, icd10_codes_selected):
    dates = []
    codes = []
    ages = []
    tbehaviours = []
    for code, vals in x.items():
        if code not in icd10_codes_selected:
            code_date = min([vvals["date"] for _, vvals in vals.items()])
            tbehaviour = [
                vvals["behaviour"]
                for _, vvals in vals.items()
                if vvals["date"] == code_date
            ][0]
            age = [
                vvals["age"] for _, vvals in vals.items() if vvals["date"] == code_date
            ][0]
            dates.append(code_date)
            codes.append(code)
            ages.append(age)
            tbehaviours.append(tbehaviour)
    if dates:
        first_date_ind = np.argmin(dates)
        return (
            dates[first_date_ind],
            codes[first_date_ind],
            ages[first_date_ind],
            tbehaviours[first_date_ind],
        )
    else:
        return np.nan, np.nan, np.nan, np.nan


def generate_labels_from_cancer_registry(df, icd10_codes_selected, mode="recruitment"):
    """
    Given a list of ICD-10 codes (can be the first few letters), generates label columns:
     "label_first_occurred_code", "label_first_occurred_date", "is_label_at_recruitment", "label_class"
    """
    df_registry = df.copy(deep=True)
    colnames = df_registry.columns

    cols_visit_date = [col for col in colnames if col.startswith("53-")]
    cols_cancer_registry_diagnosis_date = [
        col for col in colnames if col.startswith("40005-")
    ]
    cols_cancer_registry_icd10_code = [
        col for col in colnames if col.startswith("40006-")
    ]
    cols_cancer_registry_diagnosis_age = [
        col for col in colnames if col.startswith("40008-")
    ]
    cols_cancer_registry_behaviour = [
        col for col in colnames if col.startswith("40012-")
    ]
    cols_cancer_registry_histology = [
        col for col in colnames if col.startswith("40011-")
    ]

    df_registry[cols_visit_date + cols_cancer_registry_diagnosis_date] = df_registry[
        cols_visit_date + cols_cancer_registry_diagnosis_date
    ].apply(pd.to_datetime, errors="coerce")
    df_registry[cols_cancer_registry_icd10_code] = (
        df_registry[cols_cancer_registry_icd10_code].astype(str).replace("nan", np.nan)
    )
    df_registry = df_registry[
        ["eid"]
        + cols_visit_date
        + cols_cancer_registry_diagnosis_date
        + cols_cancer_registry_icd10_code
        + cols_cancer_registry_diagnosis_age
        + cols_cancer_registry_behaviour
        + cols_cancer_registry_histology
    ]

    icd10_codes = get_icd10_codes(df_registry, cols_cancer_registry_icd10_code)
    icd10_codes_selected = [
        code for code in icd10_codes if code.startswith(tuple(icd10_codes_selected))
    ]

    df_registry["icd_codes"] = df_registry.apply(
        lambda x: list(
            pd.unique(
                [
                    x[col]
                    for col in cols_cancer_registry_icd10_code
                    if pd.isnull(x[col]) == False
                ]
            )
        ),
        axis=1,
    )
    df_label = compute_first_occurrence_of_selected_cancer(
        df_registry, cols_cancer_registry_icd10_code, icd10_codes_selected
    )

    df_label["n_cancer_occurred"] = df_label["icd_codes"].apply(lambda x: len(x))
    df_label["n_cancer_occurred_lastvisit"] = df_label.apply(
        compute_n_cancer_occurred_until_last_visit,
        cols_visit_date=cols_visit_date,
        axis=1,
    )
    df_label["cancer_first_occurred_age"] = df_label.apply(
        lambda x: min(x[cols_cancer_registry_diagnosis_age]), axis=1
    )
    df_label["cancer_first_occurred_date"] = df_label.apply(
        lambda x: min(x[cols_cancer_registry_diagnosis_date]), axis=1
    )

    (
        df_label["othercancer_first_occurred_date"],
        df_label["othercancer_first_occurred_code"],
        df_label["othercancer_first_occurred_age"],
        df_label["othercancer_first_occurred_behaviour"],
    ) = zip(
        *df_label.icd_code_map.apply(
            compute_first_occurrence_of_other_cancer,
            icd10_codes_selected=icd10_codes_selected,
        )
    )

    cols_to_include = [
        "eid",
        "icd_codes",
        "icd_code_map",
        "n_cancer_occurred",
        "n_cancer_occurred_lastvisit",
        "cancer_first_occurred_age",
        "cancer_first_occurred_date",
        "othercancer_first_occurred_code",
        "othercancer_first_occurred_date",
        "othercancer_first_occurred_age",
        "othercancer_first_occurred_behaviour",
        "label_first_occurred_code",
        "label_first_occurred_date",
        "label_first_occurred_age",
    ]

    if mode == "all_visits":
        # If diagnosed with target cancer prior to selected visit: True
        # If diagnosed with target cancer later in the study: False
        # If diagnosed with other cancers but not the target one: NaN
        cols_visit_label = []
        cols_visit_tbehaviour = []
        for col_visit_date in cols_visit_date:
            visit_id = col_visit_date.split("-")[1]
            col_name = f"is_label-{visit_id}"
            col_name_tbehaviour = f"label_tumour_behaviour-{visit_id}"
            df_label[col_name] = df_label.apply(
                compute_is_label_at_visit, col_visit_date=col_visit_date, axis=1
            )
            df_label[col_name_tbehaviour] = df_label.apply(
                compute_label_tumour_behaviour_at_visit,
                col_visit_date=col_visit_date,
                icd10_codes_selected=icd10_codes_selected,
                axis=1,
            )
            cols_visit_label.append(col_name)
            cols_visit_tbehaviour.append(col_name_tbehaviour)
        cols_to_include = cols_to_include + cols_visit_tbehaviour + cols_visit_label
    else:
        # If diagnosed with target cancer prior to recruitment: True
        # If diagnosed with target cancer later in the study: False
        # If diagnosed with other cancers but not the target one: NaN
        df_label["is_label_at_recruitment"] = df_label.apply(
            compute_is_label_at_visit, col_visit_date="53-0.0", axis=1
        )
        df_label["label_tumour_behaviour_at_recruitment"] = df_label.apply(
            compute_label_tumour_behaviour_at_visit,
            col_visit_date="53-0.0",
            icd10_codes_selected=icd10_codes_selected,
            axis=1,
        )
        cols_to_include = cols_to_include + [
            "is_label_at_recruitment",
            "label_tumour_behaviour_at_recruitment",
        ]

    df_label["label_class"] = df_label.apply(assign_label_class, axis=1)
    logging.info(f"Labels generated for ICD-10 codes: {icd10_codes_selected}")

    return df_label[cols_to_include + ["label_class"]]


def filter_controls_to_nocancer(df):
    df_filtered = df.loc[df["label_class"].isin([0, 1]), :]
    logging.info(
        f"{len(df_filtered)} rows left after restricting control group to only label_class=0"
    )
    return df_filtered
