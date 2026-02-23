from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _get_cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(df[c].dtype) == "category"]

def _build_train_category_schema(X_train: pd.DataFrame, other_token: str = "OTHER") -> Dict[str, list]:
    schema: Dict[str, list] = {}
    for c in _get_cat_cols(X_train):
        cats = list(X_train[c].cat.categories)
        if other_token not in cats:
            cats.append(other_token)
        schema[c] = cats
    return schema

def _apply_category_schema(X: pd.DataFrame, schema: Dict[str, list], other_token: str = "OTHER") -> pd.DataFrame:
    X_out = X.copy()
    for c, train_cats in schema.items():
        if c not in X_out.columns:
            X_out[c] = pd.Series([np.nan] * len(X_out), index=X_out.index, dtype="object")
        if str(X_out[c].dtype) != "category":
            X_out[c] = X_out[c].astype("category")
        known_or_nan = X_out[c].isna() | X_out[c].isin(train_cats)
        X_out[c] = X_out[c].where(known_or_nan, other_token).astype("category")
        X_out[c] = X_out[c].cat.set_categories(train_cats)
    return X_out

def _split_xy(df: pd.DataFrame, id_col: str = "id", label_col: str = "target1"):
    ids = df[id_col].astype(str)
    y = np.asarray(df[label_col]).astype(int) if (label_col and label_col in df.columns) else None
    drop_cols = [c for c in [id_col, label_col, "target2"] if c and c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    return ids, y, X

@dataclass
class XGBModelArtifact:
    booster: xgb.Booster
    feature_cols: List[str]
    cat_schema: Dict[str, list]
    best_iteration: Optional[int] = None

def fit_xgb(
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    xgb_params: Optional[dict] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> XGBModelArtifact:
    """부도확률 XGBoost 학습 (binary:logistic).

    원본 코드의 핵심 수정사항:
    - scale_pos_weight를 설정하지 않음 (기본값 1.0).
    """
    _, y_train, X_train = _split_xy(train_df)
    feature_cols = list(X_train.columns)

    cat_schema = _build_train_category_schema(X_train)
    X_train = _apply_category_schema(X_train, cat_schema)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

    evals = [(dtrain, "train")]
    if eval_df is not None:
        _, y_eval, X_eval = _split_xy(eval_df)
        X_eval = X_eval.reindex(columns=feature_cols)
        X_eval = _apply_category_schema(X_eval, cat_schema)
        dvalid = xgb.DMatrix(X_eval, label=y_eval, enable_categorical=True)
        evals.append((dvalid, "valid"))

    params = dict(xgb_params) if xgb_params else {}
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "aucpr")
    params.setdefault("tree_method", "hist")

    bst = xgb.train(
        params=params, dtrain=dtrain,
        num_boost_round=num_boost_round, evals=evals,
        early_stopping_rounds=early_stopping_rounds if eval_df is not None else None,
        verbose_eval=False,
    )
    return XGBModelArtifact(
        booster=bst, feature_cols=feature_cols, cat_schema=cat_schema,
        best_iteration=getattr(bst, "best_iteration", None),
    )

def predict_xgb(model: XGBModelArtifact, df: pd.DataFrame) -> pd.Series:
    ids, _, X = _split_xy(df, label_col=None)
    X = X.reindex(columns=model.feature_cols)
    X = _apply_category_schema(X, model.cat_schema)
    dmat = xgb.DMatrix(X, enable_categorical=True)
    if model.best_iteration is not None:
        proba = model.booster.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    else:
        proba = model.booster.predict(dmat)
    return pd.Series(proba, index=ids.values, name="default_proba")

def eval_auc(model: XGBModelArtifact, df: pd.DataFrame) -> float:
    _, y, X = _split_xy(df)
    X = X.reindex(columns=model.feature_cols)
    X = _apply_category_schema(X, model.cat_schema)
    dmat = xgb.DMatrix(X, enable_categorical=True)
    if model.best_iteration is not None:
        proba = model.booster.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    else:
        proba = model.booster.predict(dmat)
    return roc_auc_score(y, proba)

def optimize_hyperparams(train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 25) -> dict:
    """Optuna로 AUC를 최대화하는 하이퍼파라미터 탐색 (원본: 첫 번째 분할에서만 수행)."""
    def objective(trial):
        params = {
            'booster': 'gbtree', 'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        mdl = fit_xgb(train_df, eval_df=val_df, xgb_params=params)
        return eval_auc(mdl, val_df)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['booster'] = 'gbtree'
    best_params['tree_method'] = 'hist'
    return best_params
