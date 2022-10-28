import pandas as pd
import numpy as np


def compute_is_cancer_at_recruitment(x):
    if pd.isnull(x["visit_date-0"]):
        return np.NaN
    else:
        if pd.isnull(x["cancer_first_occurred_date"]):
            return False
        else:
            return x["cancer_first_occurred_date"] <= x["visit_date-0"]


def compute_survival_time_with_age_for_label(x, censoring_date):
    col_age_at_recr = "age-0"
    col_date_recr = "visit_date-0"
    obs_end_date = min(
        [censoring_date, x["label_first_occurred_date"], x["date_lfu"], x["date_death"]]
    )

    # event occurred before death / LFU / EOS
    if x["label_first_occurred_date"] == obs_end_date:
        event = 1
    # censored
    else:
        event = 0
    age = (
        x[col_age_at_recr] + round((obs_end_date - x[col_date_recr]).days / 365, 2)
        if (pd.isnull(x[col_date_recr]) == False)
        else np.NaN
    )
    return [event, age, obs_end_date]
