"""Métrica opcional e importável para serialização pelos workers do AutoGluon."""

from autogluon.core.metrics import make_scorer

from .metrics import logistic_brier


logistic_brier_scorer = make_scorer(
    name="logistic_brier", score_func=logistic_brier,
    optimum=0, greater_is_better=False,
)
