"""Эксперимент с ансамблем или полносвязной нейронной сетью.

Выбранная модель: мягкое голосование (soft voting) из трёх разнородных моделей —
Random Forest (бэггинг), Gradient Boosting (бустинг) и небольшая MLP.
Все параметры зафиксированы по кросс-валидации на train (см. REPORT.md);
test_target используется только для вычисления итоговых метрик.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# Порядок признаков в german.csv (без целевой Creditability).
FEATURE_NAMES = [
    "Account_Balance", "Duration_of_Credit_monthly", "Payment_Status_of_Previous_Credit",
    "Purpose", "Credit_Amount", "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors", "Duration_in_Current_address",
    "Most_valuable_available_asset", "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents", "Telephone",
    "Foreign_Worker",
]
# Номинальные признаки (коды без порядка) -> one-hot.
CATEGORICAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Value_Savings_Stocks", "Length_of_current_employment", "Sex_Marital_Status",
    "Guarantors", "Most_valuable_available_asset", "Concurrent_Credits",
    "Type_of_apartment", "Occupation",
]
ENGINEERED = ["log_amount", "amount_per_month", "amount_per_instalment"]
POSITIVE_CLASS = 1  # Creditability = 1 — «хороший» заёмщик (700 из 1000)


def _to_frame(features) -> pd.DataFrame:
    """Приводит вход к DataFrame с известными именами столбцов."""
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    else:
        frame = pd.DataFrame(np.asarray(features))
    if not set(FEATURE_NAMES).issubset(frame.columns):
        if frame.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Ожидалось {len(FEATURE_NAMES)} признаков, получено {frame.shape[1]}")
        frame.columns = FEATURE_NAMES
    return frame[FEATURE_NAMES].astype(float)


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Собственные признаки: размер кредита в логарифме и нагрузка по платежам."""
    frame = frame.copy()
    amount = frame["Credit_Amount"]
    duration = frame["Duration_of_Credit_monthly"].clip(lower=1)
    frame["log_amount"] = np.log1p(amount)
    frame["amount_per_month"] = amount / duration
    frame["amount_per_instalment"] = amount / (duration * frame["Instalment_per_cent"].clip(lower=1))
    return frame


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = [c for c in FEATURE_NAMES if c not in CATEGORICAL] + ENGINEERED
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
    ])


def build_model(seed: int) -> VotingClassifier:
    """Soft voting: Random Forest + Gradient Boosting + MLP."""
    forest = make_pipeline(
        _preprocessor(scale_numeric=False),
        RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed,
        ),
    )
    boosting = make_pipeline(
        _preprocessor(scale_numeric=False),
        GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=3,
            subsample=0.7, max_features=0.5, random_state=seed,
        ),
    )
    mlp = make_pipeline(
        _preprocessor(scale_numeric=True),  # MLP чувствительна к масштабу
        MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0, max_iter=2000, random_state=seed),
    )
    return VotingClassifier(
        [("random_forest", forest), ("gradient_boosting", boosting), ("mlp", mlp)],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    x_train = _add_features(_to_frame(data.train_features))
    x_test = _add_features(_to_frame(data.test_features))
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    model = build_model(data.random_seed)
    model.fit(x_train, y_train)  # только train

    positive_index = list(model.classes_).index(POSITIVE_CLASS)
    probabilities = model.predict_proba(x_test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: Random Forest + Gradient Boosting + MLP(16)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
