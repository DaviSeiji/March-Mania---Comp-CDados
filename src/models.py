from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def criar_regressao_linear(*, fit_intercept=False, **params):
    return LinearRegression(fit_intercept=fit_intercept, **params)


def criar_random_forest(*, random_state=42, **params):

    config = {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "n_jobs": -1,
    }
    config.update(params)
    return RandomForestRegressor(random_state=random_state, **config)


def criar_xgboost(*, random_state=42, **params):

    config = {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    config.update(params)
    return XGBRegressor(random_state=random_state, **config)
