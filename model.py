"""
model.py — Multi-output gradient boosting with:

  Priority 1: Bayesian hyperparameter optimisation (GP-based, replaces random search)
              - Gaussian Process surrogate models the objective landscape
              - Expected Improvement acquisition guides next trial
              - Finds better params in fewer trials than random search
  Priority 2: Extended feature set (26 features via updated core.py)
              Wider parameter search space
  Model versioning, evaluation, permutation importance unchanged.
"""

import os, json, pickle, hashlib, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from core import (
    ALL_FEATS, TARGET_VOLS, PCU_FACTORS, LAG_COLS,
    LOS_LABELS, LOS_ENCODE, MODEL_DIR, LOG_DIR,
    make_weights, compute_los, los_array, smape,
)


# ── Default & search space ─────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "estimator__max_iter":         300,
    "estimator__learning_rate":    0.05,
    "estimator__max_depth":        6,
    "estimator__min_samples_leaf": 20,
    "estimator__l2_regularization":0.1,
}

# Bayesian search space: (low, high, type)
# type: "int", "float", "log" (log-uniform float), "choice"
BAYES_SPACE = {
    "estimator__learning_rate":    (0.005, 0.2,  "log"), # Allow slower, more precise learning
    "estimator__max_depth":        (8,    20,     "int"), # INCREASED: Allows deeper trees for complex peaks
    "estimator__min_samples_leaf": (5,    40,     "int"), # Slightly smaller leaves for finer detail
    "estimator__l2_regularization":(1e-3, 0.5,    "log"), # DECREASED: High regularization causes under-prediction
    "estimator__max_iter":         (400,  1200,   "int"), # INCREASED: More boosting rounds to correct residuals
}


def _build_model(params=None):
    p = params or {}
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter          = int(p.get("estimator__max_iter",         DEFAULT_PARAMS["estimator__max_iter"])),
            learning_rate     = float(p.get("estimator__learning_rate",  DEFAULT_PARAMS["estimator__learning_rate"])),
            max_depth         = int(p.get("estimator__max_depth",        DEFAULT_PARAMS["estimator__max_depth"])),
            min_samples_leaf  = int(p.get("estimator__min_samples_leaf", DEFAULT_PARAMS["estimator__min_samples_leaf"])),
            l2_regularization = float(p.get("estimator__l2_regularization", DEFAULT_PARAMS["estimator__l2_regularization"])),
            random_state      = 42,
            early_stopping    = True,
            n_iter_no_change  = 20,
            validation_fraction = 0.1,
            interaction_cst = None, # You can define specific feature pairs here later
            categorical_features = [True if f in ['hour', 'day_of_week', 'road_idx'] else False for f in ALL_FEATS],
        ),
        n_jobs=1,
    )


# ── Bayesian Optimisation (Gaussian Process surrogate) ────────────────────────
class BayesianOptimiser:
    """
    Lightweight GP-based Bayesian optimisation.
    Uses RBF kernel GP as surrogate + Expected Improvement acquisition.
    No external dependencies beyond numpy/scipy.
    """

    def __init__(self, space: dict, n_initial: int = 5, random_state: int = 42):
        self.space       = space
        self.n_initial   = n_initial
        self.rng         = np.random.RandomState(random_state)
        self.X_observed  = []   # list of param vectors (normalised [0,1])
        self.y_observed  = []   # list of objective scores
        self._keys       = list(space.keys())

    # -- Sampling helpers -------------------------------------------------------
    def _sample_random(self):
        params = {}
        for k, (lo, hi, typ) in self.space.items():
            if typ == "int":
                params[k] = int(self.rng.randint(lo, hi + 1))
            elif typ == "log":
                params[k] = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            else:
                params[k] = float(self.rng.uniform(lo, hi))
        return params

    def _normalise(self, params: dict) -> np.ndarray:
        vec = []
        for k, (lo, hi, typ) in self.space.items():
            v = params[k]
            if typ == "log":
                vec.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
            else:
                vec.append((v - lo) / (hi - lo))
        return np.array(vec)

    # -- GP surrogate -----------------------------------------------------------
    def _rbf_kernel(self, X1, X2, length_scale=0.5, sigma_f=1.0):
        """Squared exponential (RBF) kernel."""
        diff = X1[:, None, :] - X2[None, :, :]          # (n1, n2, d)
        sq   = np.sum(diff ** 2, axis=-1)                # (n1, n2)
        return sigma_f ** 2 * np.exp(-0.5 * sq / length_scale ** 2)

    def _gp_predict(self, X_new):
        """Returns GP posterior mean and std at X_new points."""
        if len(self.X_observed) < 2:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        noise = 1e-4

        K    = self._rbf_kernel(X_obs, X_obs) + noise * np.eye(len(X_obs))
        K_s  = self._rbf_kernel(X_obs, X_new)
        K_ss = self._rbf_kernel(X_new, X_new) + noise * np.eye(len(X_new))

        try:
            L     = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu    = K_s.T @ alpha
            v     = np.linalg.solve(L, K_s)
            sigma = np.sqrt(np.maximum(np.diag(K_ss - v.T @ v), 1e-9))
        except np.linalg.LinAlgError:
            mu    = np.full(len(X_new), np.mean(y_obs))
            sigma = np.ones(len(X_new))

        return mu, sigma

    # -- Acquisition: Expected Improvement -------------------------------------
    def _expected_improvement(self, X_cand, xi=0.01):
        from scipy.stats import norm
        mu, sigma = self._gp_predict(X_cand)
        y_best    = np.min(self.y_observed) if self.y_observed else 0.0
        z         = (y_best - mu - xi) / (sigma + 1e-9)
        ei        = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    # -- Propose next candidate ------------------------------------------------
    def suggest(self) -> dict:
        """Return next hyperparameter candidate to evaluate."""
        if len(self.X_observed) < self.n_initial:
            return self._sample_random()

        # Sample a pool of random candidates, pick max EI
        n_pool   = 200
        pool     = [self._sample_random() for _ in range(n_pool)]
        X_pool   = np.array([self._normalise(p) for p in pool])
        ei       = self._expected_improvement(X_pool)
        best_idx = int(np.argmax(ei))
        return pool[best_idx]

    def register(self, params: dict, score: float):
        """Record an observed (params, score) pair."""
        self.X_observed.append(self._normalise(params))
        self.y_observed.append(float(score))

    def best(self) -> dict:
        if not self.y_observed:
            return DEFAULT_PARAMS.copy()
        return self._params_at(int(np.argmin(self.y_observed)))

    def _params_at(self, idx: int) -> dict:
        """Reconstruct param dict from normalised observation at index."""
        vec = self.X_observed[idx]
        params = {}
        for i, (k, (lo, hi, typ)) in enumerate(self.space.items()):
            v_norm = vec[i]
            if typ == "int":
                params[k] = int(round(lo + v_norm * (hi - lo)))
            elif typ == "log":
                params[k] = float(np.exp(np.log(lo) + v_norm * (np.log(hi) - np.log(lo))))
            else:
                params[k] = float(lo + v_norm * (hi - lo))
        return params


# ── CV objective ───────────────────────────────────────────────────────────────
def _cv_score(params, train_df, n_splits=5):
    """
    Evaluate params via TimeSeriesSplit CV.
    Returns scalar: 0.7 * V/C_MAE - 0.3 * LOS_ACC  (lower = better).
    """
    X      = train_df[ALL_FEATS].values
    Ys     = train_df[TARGET_VOLS].values
    caps   = train_df["computed_capacity"].values
    t_los  = train_df["los_grade"].values
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Ys[tr_idx], Ys[va_idx]
        w_tr = make_weights(train_df.iloc[tr_idx])
        m    = _build_model(params)
        try:
            m.fit(X_tr, Y_tr, **{"sample_weight": w_tr})
        except TypeError:
            m.fit(X_tr, Y_tr)

        pred   = np.clip(m.predict(X_va), 0, None)
        pred_pcu = pred  @ np.array(PCU_FACTORS)
        true_pcu = Y_va  @ np.array(PCU_FACTORS)
        caps_va  = caps[va_idx]
        pred_vc  = np.where(caps_va > 0, pred_pcu / caps_va, 0)
        true_vc  = np.where(caps_va > 0, true_pcu / caps_va, 0)
        diff = true_vc - pred_vc
        # Penalty multiplier: 2.5x penalty if Actual > Predicted (under-prediction)
        weighted_diff = np.where(diff > 0, diff * 2.5, np.abs(diff))
        vc_mae = np.mean(weighted_diff)
        los_acc  = np.mean(los_array(pred_vc) == t_los[va_idx])


        # Increase LOS weight to 50% to prioritize the Grade (LOS F vs B)
        scores.append(0.5 * vc_mae - 0.5 * los_acc)

    return float(np.mean(scores))


# ── Public tuning API ──────────────────────────────────────────────────────────
def tune_model(train_df, n_iter=20, n_splits=5, verbose=True):
    """
    Bayesian hyperparameter search with GP surrogate + Expected Improvement.
    Replaces random search. Same interface as before.
    Returns best params dict.
    """
    optimiser = BayesianOptimiser(BAYES_SPACE, n_initial=15) # Force more exploration

    if verbose:
        print(f"  Bayesian tuning: {n_iter} trials × {n_splits}-fold time-series CV")
        print(f"  Objective: 0.7 × V/C MAE − 0.3 × LOS Accuracy  (lower = better)")
        print(f"  First {optimiser.n_initial} trials: random exploration")
        print(f"  Remaining {n_iter - optimiser.n_initial} trials: GP-guided (Expected Improvement)\n")

    best_score = np.inf

    for i in range(n_iter):
        params = optimiser.suggest()
        score  = _cv_score(params, train_df, n_splits)
        optimiser.register(params, score)

        if score < best_score:
            best_score = score
            tag = "★ new best"
        else:
            tag = ""

        if verbose:
            phase = "explore" if i < optimiser.n_initial else "exploit"
            lr  = params.get("estimator__learning_rate", "?")
            dep = params.get("estimator__max_depth", "?")
            it  = params.get("estimator__max_iter", "?")
            print(f"  [{i+1:>2}/{n_iter}] [{phase}] "
                  f"lr={lr:.4f}  depth={dep}  iter={it}  "
                  f"score={score:.5f}  {tag}")

    best = optimiser.best()
    if verbose:
        print(f"\n  Best score: {best_score:.5f}")
        print(f"  Best params: {best}")
    return best


# ── Train ──────────────────────────────────────────────────────────────────────
def train(train_df, params=None, verbose=True):
    X  = train_df[ALL_FEATS].values
    Ys = train_df[TARGET_VOLS].values
    w  = make_weights(train_df)
    model = _build_model(params)
    try:
        model.fit(X, Ys, **{"sample_weight": w})
    except TypeError:
        model.fit(X, Ys)
    if verbose:
        print(f"  Multi-output HistGBR trained  |  "
              f"targets={len(TARGET_VOLS)}  features={len(ALL_FEATS)}  rows={len(train_df):,}")
    return model


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(model, test_df):
    X_te  = test_df[ALL_FEATS].values
    Ys_te = test_df[TARGET_VOLS].values
    pred  = np.clip(model.predict(X_te), 0, None)

    per_target = {}
    for j, target in enumerate(TARGET_VOLS):
        y_t = Ys_te[:, j]; p_t = pred[:, j]
        per_target[target] = {
            "mae":   round(float(mean_absolute_error(y_t, p_t)), 3),
            "rmse":  round(float(np.sqrt(mean_squared_error(y_t, p_t))), 3),
            "r2":    round(float(r2_score(y_t, p_t)), 4),
            "smape": round(smape(y_t, p_t), 2),
        }

    pred_pcu = pred  @ np.array(PCU_FACTORS)
    true_pcu = Ys_te @ np.array(PCU_FACTORS)
    caps     = test_df["computed_capacity"].values
    pred_vc  = np.where(caps > 0, pred_pcu / caps, 0.0)
    true_vc  = test_df["vc_ratio"].values
    true_los = test_df["los_grade"].values
    pred_los = los_array(pred_vc)

    vc_mae  = float(mean_absolute_error(true_vc, pred_vc))
    vc_rmse = float(np.sqrt(mean_squared_error(true_vc, pred_vc)))
    vc_r2   = float(r2_score(true_vc, pred_vc))
    los_acc = float(np.mean(pred_los == true_los))
    los_w1  = float(np.mean(
        np.abs(np.array([LOS_ENCODE[x] for x in pred_los]) -
               np.array([LOS_ENCODE[x] for x in true_los])) <= 1))

    per_road = {}
    for road in test_df["road"].unique():
        mask = test_df["road"].values == road
        if mask.sum() < 2: continue
        per_road[road] = {
            "mae":     round(float(mean_absolute_error(true_vc[mask], pred_vc[mask])), 4),
            "r2":      round(float(r2_score(true_vc[mask], pred_vc[mask])), 4),
            "los_acc": round(float(np.mean(pred_los[mask] == true_los[mask])), 4),
        }

    per_hour = {}
    for h in range(24):
        mask = test_df["hour"].values == h
        if mask.sum() == 0: continue
        per_hour[str(h)] = {
            "actual_vc": round(float(np.mean(true_vc[mask])), 4),
            "pred_vc":   round(float(np.mean(pred_vc[mask])), 4),
            "mae":       round(float(mean_absolute_error(true_vc[mask], pred_vc[mask])), 4),
            "los_acc":   round(float(np.mean(pred_los[mask] == true_los[mask])), 4),
        }

    residuals = true_vc - pred_vc
    confusion = {}
    for g_t in LOS_LABELS:
        confusion[f"actual_{g_t}"] = {
            f"pred_{g_p}": int(np.sum((true_los==g_t)&(pred_los==g_p)))
            for g_p in LOS_LABELS
        }

    return {
        "per_target":   per_target,
        "vc_mae":       round(vc_mae, 5),
        "vc_rmse":      round(vc_rmse, 5),
        "vc_r2":        round(vc_r2, 4),
        "los_accuracy": round(los_acc, 4),
        "los_within1":  round(los_w1, 4),
        "per_road":     per_road,
        "per_hour":     per_hour,
        "residuals":    {
            "mean": round(float(residuals.mean()), 5),
            "std":  round(float(residuals.std()),  5),
            "p5":   round(float(np.percentile(residuals, 5)),  5),
            "p95":  round(float(np.percentile(residuals, 95)), 5),
            "extreme_count": int(np.sum(np.abs(residuals) > 0.5)),
        },
        "confusion": confusion,
    }


# ── Explainability ─────────────────────────────────────────────────────────────
def explain(model, df, n_repeats=10):
    sample = df.sample(min(500, len(df)), random_state=42)
    X  = sample[ALL_FEATS].values
    Ys = sample[TARGET_VOLS].values

    def scorer(est, X_s, Y_s):
        p    = np.clip(est.predict(X_s), 0, None)
        caps = sample["computed_capacity"].values[:len(X_s)]
        vc_p = np.where(caps > 0, p  @ np.array(PCU_FACTORS) / caps, 0)
        vc_t = np.where(caps > 0, Y_s @ np.array(PCU_FACTORS) / caps, 0)
        return -float(mean_absolute_error(vc_t, vc_p))

    result = permutation_importance(
        model, X, Ys, n_repeats=n_repeats, random_state=42, n_jobs=-1,
        scoring=lambda est, X_s, Y_s: scorer(est, X_s, Y_s),
    )
    ranked = sorted(zip(ALL_FEATS, result.importances_mean, result.importances_std),
                    key=lambda x: -abs(x[1]))
    return [{"feature": f, "importance": round(float(i), 6), "std": round(float(s), 6)}
            for f, i, s in ranked]


# ── Versioning ─────────────────────────────────────────────────────────────────
def save_model(model, params, metrics, road_params, road_order, csv_path=None):
    ts  = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tag = f"v_{ts}"
    pkl_path  = os.path.join(MODEL_DIR, f"model_{tag}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"meta_{tag}.json")

    with open(pkl_path, "wb") as f:
        pickle.dump({"model": model, "road_params": road_params,
                     "road_order": road_order}, f)

    meta = {
        "version":     tag,
        "trained_at":  ts,
        "params":      {k: str(v) for k, v in (params or {}).items()},
        "vc_mae":      metrics.get("vc_mae"),
        "los_accuracy":metrics.get("los_accuracy"),
        "features":    ALL_FEATS,
        "n_features":  len(ALL_FEATS),
        "targets":     TARGET_VOLS,
        "tuner":       "BayesianGP",
    }
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            meta["csv_sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(MODEL_DIR, "latest.txt"), "w") as f:
        f.write(tag)

    return tag


def load_model(version="latest"):
    if version == "latest":
        ptr = os.path.join(MODEL_DIR, "latest.txt")
        if not os.path.exists(ptr):
            raise FileNotFoundError("No trained model found. Run /train first.")
        with open(ptr) as f:
            tag = f.read().strip()
    else:
        tag = version

    pkl  = os.path.join(MODEL_DIR, f"model_{tag}.pkl")
    meta = os.path.join(MODEL_DIR, f"meta_{tag}.json")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Model not found: {pkl}")
    with open(pkl,  "rb") as f: bundle = pickle.load(f)
    with open(meta)       as f: meta_d = json.load(f)
    return bundle["model"], bundle["road_params"], bundle["road_order"], meta_d


def list_versions():
    versions = []
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.startswith("meta_v") and fname.endswith(".json"):
            with open(os.path.join(MODEL_DIR, fname)) as f:
                versions.append(json.load(f))
    return versions
