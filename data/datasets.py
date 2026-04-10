"""
Dataset loading and preprocessing.

Each dataset is returned as a DatasetInfo with:
  - X, y arrays (ready for sklearn)
  - metadata: feature names, column indices, protected group info
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class DatasetInfo:
    """Preprocessed arrays and metadata for one loaded dataset."""
    X:                    np.ndarray   # float (n_samples, n_features), NaN preserved
    y:                    np.ndarray   # int   (n_samples,), binary 0/1
    feature_names:        List[str]
    num_col_indices:      List[int]    # originally-numerical columns
    cat_col_indices:      List[int]    # originally-categorical columns (factorized to int)
    outlier_col_idx:      int          # column used for feature-corruption experiments
    protected_col_name:   str
    protected_col_idx:    int
    protected_group_mask: np.ndarray   # bool (n_samples,), True = protected subgroup


def load_dataset(name: str, datasets_dir) -> DatasetInfo:
    """
    Load and preprocess a dataset by name.

    Categorical columns are factorised to integer codes; NaN is preserved
    so that pipeline-level imputers can handle it downstream.

    Supported names
    ---------------
    adult   — UCI Adult Income         (protected: sex = Female)
    german  — Statlog German Credit    (protected: personal_status female codes)
    titanic — Titanic survival         (protected: Sex = female)
    compas  — ProPublica COMPAS        (protected: race = African-American)
    """
    datasets_dir = Path(datasets_dir)

    if name == "adult":
        cols = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
        ]
        df = pd.read_csv(
            datasets_dir / "adult.data",
            header=None, names=cols, na_values="?", skipinitialspace=True,
        )
        y                    = (df["income"] == ">50K").astype(int).values
        protected_group_mask = (df["sex"] == "Female").values
        X_df                 = df.drop(columns="income").copy()
        cat_col_names        = list(X_df.select_dtypes(include="object").columns)
        outlier_col, protected_col = "hours_per_week", "sex"

    elif name == "german":
        cols = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class",
        ]
        df = pd.read_csv(datasets_dir / "german.data", sep=" ", header=None, names=cols)
        y                    = (df["class"] == 1).astype(int).values
        protected_group_mask = df["personal_status"].isin(["A92", "A95"]).values
        X_df                 = df.drop(columns="class").copy()
        cat_col_names        = list(X_df.select_dtypes(include="object").columns)
        outlier_col, protected_col = "duration", "personal_status"

    elif name == "titanic":
        df = pd.read_csv(datasets_dir / "titanic.csv")
        y                    = df["Survived"].astype(int).values
        protected_group_mask = (df["Sex"] == "female").values
        X_df                 = df.drop(columns=["Survived", "PassengerId", "Name",
                                                 "Ticket", "Cabin"]).copy()
        cat_col_names        = list(X_df.select_dtypes(include="object").columns)
        outlier_col, protected_col = "Age", "Sex"

    elif name == "compas":
        compas_path = datasets_dir / "compas.csv"
        if not compas_path.exists():
            raise FileNotFoundError(
                f"COMPAS dataset not found at {compas_path}.\n"
                "Download 'compas-scores-two-years.csv' from "
                "https://github.com/propublica/compas-analysis "
                "and save it as datasets/compas.csv"
            )
        df = pd.read_csv(compas_path)
        df = df[
            df["days_b_screening_arrest"].between(-30, 30) &
            (df["is_recid"] != -1) &
            (df["c_charge_degree"] != "O") &
            (df["score_text"] != "N/A")
        ].copy()
        y                    = df["two_year_recid"].astype(int).values
        protected_group_mask = (df["race"] == "African-American").values
        keep_cols = [
            "age", "priors_count", "days_b_screening_arrest",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "c_charge_degree", "race", "sex",
        ]
        X_df          = df[keep_cols].copy()
        cat_col_names = list(X_df.select_dtypes(include="object").columns)
        outlier_col, protected_col = "priors_count", "race"

    else:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from: adult, german, titanic, compas"
        )

    for col in cat_col_names:
        X_df[col], _ = pd.factorize(X_df[col])
    X_df = X_df.astype(float)

    feature_names   = list(X_df.columns)
    X               = X_df.values
    num_col_indices = [i for i, c in enumerate(feature_names) if c not in cat_col_names]
    cat_col_indices = [i for i, c in enumerate(feature_names) if c in cat_col_names]

    return DatasetInfo(
        X=X, y=y,
        feature_names=feature_names,
        num_col_indices=num_col_indices,
        cat_col_indices=cat_col_indices,
        outlier_col_idx=feature_names.index(outlier_col),
        protected_col_name=protected_col,
        protected_col_idx=feature_names.index(protected_col),
        protected_group_mask=protected_group_mask,
    )
