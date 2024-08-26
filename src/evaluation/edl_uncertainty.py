from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame


def compute_df_edl_uncertainty(
        df: DataFrame,
        emotion_labels: List,
):
    if not emotion_labels:
        return
    print(emotion_labels)
    uncertainties = [compute_edl_uncertainty(row) for row in df[emotion_labels].to_numpy()]
    return uncertainties


def compute_edl_uncertainty(row):
    # don't use rows with only nans
    if not pd.isnull(row).all():
        evidence = np.maximum(0, row)  # ReLU
        alpha = evidence + 1
        # print(alpha)
        edl_uncertainty = len(row) / np.sum(alpha)

        return edl_uncertainty

    return None
