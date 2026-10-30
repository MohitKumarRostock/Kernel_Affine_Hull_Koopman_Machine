"""Standalone practical controls with explicitly historical information sets.

The Holt control estimates smoothing parameters once per fit. At a forecast
origin it assimilates only observations strictly before that origin. Seasonal
naive uses the last observed season; it is allowed its declared seasonal lag
even if this exceeds the geometric methods' finite context.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import Holt


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be an integer at least {minimum}')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be an integer at least {minimum}') from error
    if integer != value or integer < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return integer


def _history(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values):
        raise ValueError('History must be a nonempty one-dimensional array')
    return values


def _forecast_indices(values, origins, horizon):
    """Origins are FIRST FUTURE indices; len(values) is a valid origin."""
    values = _history(values)
    horizon = _integer(horizon, 'horizon')
    raw = np.atleast_1d(np.asarray(origins))
    if raw.ndim != 1:
        raise ValueError('Forecast origins must be a scalar or one-dimensional sequence')
    origins = np.asarray([_integer(value, 'forecast origin') for value in raw], dtype=int)
    if len(origins) and origins.max() > len(values):
        raise ValueError('Forecast origin exceeds the supplied historical observations')
    return values, origins, horizon


class TabularBaseline:
    def __init__(self, kind, *, seed=17, width=32, regularization=0.01,
                 iterations=150):
        self.kind = kind
        self.seed = int(seed)
        self.width = int(width)
        self.regularization = float(regularization)
        self.iterations = int(iterations)

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(Y)
        Z, T = self.x_scaler_.transform(X), self.y_scaler_.transform(Y)
        if self.kind == 'hist_gradient':
            # Independent direct heads, one per horizon; no internal random
            # validation split and no access to validation/test targets.
            self.estimator_ = MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=self.iterations, max_leaf_nodes=self.width,
                min_samples_leaf=max(2, min(20, len(X)//8)),
                l2_regularization=self.regularization,
                early_stopping=False, random_state=self.seed), n_jobs=1)
        elif self.kind == 'mlp':
            self.estimator_ = MLPRegressor(
                hidden_layer_sizes=(self.width,), alpha=self.regularization,
                solver='adam', max_iter=self.iterations, random_state=self.seed,
                shuffle=False, early_stopping=False, tol=1e-5,
                n_iter_no_change=25, batch_size=min(64, len(X)))
        else:
            raise ValueError(self.kind)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            target = T.ravel() if self.kind == 'mlp' and T.shape[1] == 1 else T
            self.estimator_.fit(Z, target)
        self.warnings_ = [str(w.message) for w in caught]
        self.n_outputs_ = Y.shape[1]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(self.x_scaler_.transform(X)))
        if pred.ndim == 1:
            pred = pred[:, None]
        return self.y_scaler_.inverse_transform(pred)

    def metadata(self):
        out = {'kind': self.kind, 'input_dimension': self.n_features_in_,
               'target_dimension': self.n_outputs_, 'width': self.width,
               'regularization': self.regularization,
               'max_iterations': self.iterations, 'warnings': self.warnings_}
        if self.kind == 'mlp':
            out['iterations_realized'] = int(self.estimator_.n_iter_)
            out['coefficient_count'] = int(sum(a.size for a in self.estimator_.coefs_)
                                          + sum(a.size for a in self.estimator_.intercepts_))
        return out


class SeasonalNaive:
    def __init__(self, seasonal_period):
        self.seasonal_period = _integer(seasonal_period, 'seasonal_period')

    def fit_history(self, values, cutoff, max_history=None):
        values = _history(values)
        self.cutoff_ = _integer(cutoff, 'cutoff')
        if self.cutoff_ > len(values):
            raise ValueError('Training cutoff exceeds the supplied history')
        self.requested_max_history_ = (None if max_history is None
                                       else _integer(max_history, 'max_history'))
        return self

    def predict_history(self, values, origins, horizon):
        values, origins, horizon = _forecast_indices(values, origins, horizon)
        forecasts = []
        for origin in origins:
            past = values[:origin]
            finite = np.flatnonzero(np.isfinite(past))
            if not len(finite):
                raise ValueError('No historical observation at forecast origin')
            last = float(past[finite[-1]])
            period = self.seasonal_period if origin >= self.seasonal_period else 1
            forecast = np.empty(horizon)
            for step in range(horizon):
                index = origin-period+(step % period)
                assert index < origin
                forecast[step] = values[index] if np.isfinite(values[index]) else last
            forecasts.append(forecast)
        return np.asarray(forecasts).reshape(len(origins), horizon)

    def metadata(self):
        return {'kind': 'seasonal_naive', 'period': self.seasonal_period,
                'missing_lag_fallback': 'last finite observation before origin',
                'short_history_fallback': 'period one until a complete seasonal lag exists',
                'origin_convention': 'first future index; observations at that index are excluded',
                'fit_cutoff': getattr(self, 'cutoff_', None),
                'requested_max_history': getattr(self, 'requested_max_history_', None),
                'history': 'full available past, not finite-context limited; max_history does not truncate seasonal lags'}


class HoltBaseline:
    def __init__(self, damped=True):
        self.damped = bool(damped)

    def fit_history(self, values, cutoff, max_history=None):
        values = _history(values)
        cutoff = _integer(cutoff, 'cutoff')
        if cutoff > len(values):
            raise ValueError('Training cutoff exceeds the supplied history')
        max_history = None if max_history is None else _integer(max_history, 'max_history')
        start = 0 if max_history is None else max(0, cutoff-max_history)
        # Forward filling is restricted to the historical training segment.
        history = values[start:cutoff].copy()
        finite = np.flatnonzero(np.isfinite(history))
        if len(finite) < 4:
            raise ValueError('Holt needs at least four historical observations')
        self.discarded_leading_missing_ = int(finite[0])
        self.fit_start_index_ = start + self.discarded_leading_missing_
        self.finite_training_observations_ = len(finite)
        history = history[finite[0]:]
        self.forward_filled_training_values_ = int(np.count_nonzero(~np.isfinite(history)))
        for i in range(1, len(history)):
            if not np.isfinite(history[i]):
                history[i] = history[i-1]
        self.mean_, self.scale_ = float(np.mean(history)), float(np.std(history))
        if self.scale_ < 1e-12:
            self.scale_ = 1.0
        standardized = (history-self.mean_)/self.scale_
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = Holt(standardized, damped_trend=self.damped,
                       initialization_method='estimated').fit(optimized=True)
        self.warnings_ = [str(w.message) for w in caught]
        self.alpha_ = float(fit.params['smoothing_level'])
        self.beta_ = float(fit.params['smoothing_trend'])
        self.phi_ = float(fit.params['damping_trend']) if self.damped else 1.0
        self.level_, self.trend_ = float(fit.level[-1]), float(fit.trend[-1])
        if not np.isfinite([self.alpha_, self.beta_, self.phi_, self.level_, self.trend_]).all():
            raise FloatingPointError('Holt fitting returned a nonfinite parameter or state')
        self.optimization_success_ = bool(fit.mle_retvals.get('success', False))
        self.cutoff_ = int(cutoff)
        self.max_history_ = max_history
        self.training_observations_ = len(history)
        return self

    def predict_history(self, values, origins, horizon):
        if not hasattr(self, 'cutoff_'):
            raise RuntimeError('Fit Holt before prediction')
        values, origins, horizon = _forecast_indices(values, origins, horizon)
        if len(origins) and origins.min() < self.cutoff_:
            raise ValueError('Forecast origin predates fitted history cutoff')
        order = np.argsort(origins, kind='stable')
        out = np.empty((len(origins), horizon))
        level, trend, consumed = self.level_, self.trend_, self.cutoff_
        multipliers = np.cumsum(self.phi_ ** np.arange(1, horizon+1))
        for index in order:
            origin = int(origins[index])
            while consumed < origin:
                observation = float(values[consumed])
                predicted_level = level+self.phi_*trend
                if np.isfinite(observation):
                    observation = (observation-self.mean_)/self.scale_
                    new_level = self.alpha_*observation+(1-self.alpha_)*predicted_level
                    new_trend = self.beta_*(new_level-level)+(1-self.beta_)*self.phi_*trend
                else:
                    new_level, new_trend = predicted_level, self.phi_*trend
                level, trend = new_level, new_trend
                consumed += 1
            out[index] = self.mean_+self.scale_*(level+multipliers*trend)
        return out

    def metadata(self):
        return {'kind': 'damped_holt' if self.damped else 'holt',
                'smoothing_level': self.alpha_, 'smoothing_trend': self.beta_,
                'damping': self.phi_, 'training_observations': self.training_observations_,
                'finite_training_observations': self.finite_training_observations_,
                'fit_cutoff': self.cutoff_, 'fit_start_index': self.fit_start_index_,
                'max_history': self.max_history_,
                'discarded_leading_missing': self.discarded_leading_missing_,
                'forward_filled_training_values': self.forward_filled_training_values_,
                'origin_convention': 'first future index; observations at that index are excluded',
                'missing_training': 'discard leading missing values, then historical forward fill',
                'missing_online': 'state prediction without measurement update',
                'optimization_success': self.optimization_success_,
                'warnings': self.warnings_}
