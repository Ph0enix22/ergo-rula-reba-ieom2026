"""Subject-Wise Data Partitioning Engine to Prevent Inter-Frame Data Leakage.

Guarantees that frames/videos from the same subject never overlap across train, val, and test splits.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def split_by_subject(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits grouped strictly by subject.

    Parameters:
    - df: Input dataframe with posture samples
    - subject_col: Column identifying distinct subjects or worker sessions
    - train_ratio: Fraction of subjects allocated to training
    - val_ratio: Fraction of subjects allocated to validation
    - test_ratio: Fraction of subjects allocated to testing
    - seed: Random seed for deterministic reproducibility

    Returns:
    DataFrame with an added 'split' column containing 'train', 'val', or 'test'.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    if subject_col not in df.columns:
        raise KeyError(f"Subject column '{subject_col}' not found in dataframe.")

    out_df = df.copy()
    unique_subjects = np.array(sorted(out_df[subject_col].dropna().unique()))

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_train = int(np.round(n_subjects * train_ratio))
    n_val = int(np.round(n_subjects * val_ratio))

    # Ensure at least 1 subject in test if multiple subjects exist
    if n_subjects >= 3:
        n_train = max(1, min(n_train, n_subjects - 2))
        n_val = max(1, min(n_val, n_subjects - n_train - 1))

    train_subs = set(unique_subjects[:n_train])
    val_subs = set(unique_subjects[n_train:n_train + n_val])
    test_subs = set(unique_subjects[n_train + n_val:])

    # Edge case: if test_subs ended up empty due to rounding with few subjects
    if not test_subs and len(unique_subjects) >= 2:
        test_subs.add(unique_subjects[-1])
        if unique_subjects[-1] in val_subs:
            val_subs.remove(unique_subjects[-1])
        elif unique_subjects[-1] in train_subs:
            train_subs.remove(unique_subjects[-1])

    # Assert no subject leakage
    assert train_subs.isdisjoint(val_subs), "Train and Val subjects overlap!"
    assert train_subs.isdisjoint(test_subs), "Train and Test subjects overlap!"
    assert val_subs.isdisjoint(test_subs), "Val and Test subjects overlap!"

    def assign_split(s):
        if s in train_subs:
            return "train"
        elif s in val_subs:
            return "val"
        elif s in test_subs:
            return "test"
        return "unassigned"

    out_df["split"] = out_df[subject_col].apply(assign_split)
    return out_df
