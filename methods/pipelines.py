"""
Pipeline factory functions — one per pipeline variant.

Each factory returns a fresh, unfitted sklearn Pipeline.
Call the factory inside loops to avoid state leaking between runs.

Pipelines
---------
make_pipeline_a — ColumnTransformer (MedianImpute → PowerTransform → Scale for num,
                  MostFreqImpute for cat) → LogisticRegression
make_pipeline_b — MedianImpute → Scale → PCA → SelectKBest → RandomForest
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def make_pipeline_a(num_col_indices: list, cat_col_indices: list,
                    classifier=None) -> Pipeline:
    """
    Pipeline A — two-path ColumnTransformer (benchmark paper [3]).

    Numerical path  : MedianImputer → PowerTransformer (Yeo-Johnson) → StandardScaler
    Categorical path: MostFrequentImputer
    → ColumnTransformer union → LogisticRegression (lbfgs, max_iter=1000)

    Parameters
    ----------
    num_col_indices : indices of numerical columns in X
    cat_col_indices : indices of categorical columns in X (already int-encoded)
    classifier      : optional custom classifier (default: LogisticRegression)
    """
    if classifier is None:
        classifier = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power",   PowerTransformer(method="yeo-johnson")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipe,   num_col_indices),
            ("cat", categorical_pipe, cat_col_indices),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


def make_pipeline_b(n_pca_components: int = 8, k_best: int = 5,
                    n_estimators: int = 100) -> Pipeline:
    """
    Pipeline B — automated feature engineering (benchmark paper [3]).

    MedianImputer → StandardScaler → PCA → SelectKBest (f_classif) → RandomForest

    Parameters
    ----------
    n_pca_components : PCA output dimensions (clip to n_features before calling)
    k_best           : number of top features kept by SelectKBest
    n_estimators     : number of trees in the RandomForest
    """
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("pca",        PCA(n_components=n_pca_components, random_state=42)),
        ("selector",   SelectKBest(f_classif, k=k_best)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )),
    ])
