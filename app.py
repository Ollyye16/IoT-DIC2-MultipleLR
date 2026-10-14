"""DIC2 Multiple Linear Regression — Flask backend (scikit-learn + statsmodels).

POST /api/analyze  {"url": csv_url, "target": column_index, "test_idx": [row indices]}
  -> feature names + every (method, k) model with test metrics and OLS inference.
The frontend (index.html) calls this when the backend is reachable and otherwise
falls back to the same math in browser JavaScript (e.g. on GitHub Pages).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, enet_path, lasso_path
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).parent
app = Flask(__name__)


@app.after_request
def cors(resp):  # GitHub Pages frontend may call a Vercel-hosted API
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


# ---------- data ----------
def load_features(url, target):
    """Same encoding rules as the JS frontend: numeric columns as-is, text columns
    (2..20 levels) one-hot with the first sorted level dropped as reference."""
    df = pd.read_csv(url, dtype=str, keep_default_na=False, skipinitialspace=True)
    df = df.apply(lambda s: s.str.strip())
    y = pd.to_numeric(df.iloc[:, target]).to_numpy(float)
    names, cols = [], []
    for j, h in enumerate(df.columns):
        if j == target:
            continue
        num = pd.to_numeric(df[h], errors="coerce")
        if (df[h] != "").all() and num.notna().all():
            names.append(h)
            cols.append(num.to_numpy(float))
            continue
        levels = sorted(df[h].unique())
        if not 2 <= len(levels) <= 20:
            continue  # ID-like or constant text column
        for lv in levels[1:]:
            names.append(f"{h}_{lv}")
            cols.append((df[h] == lv).to_numpy(float))
    return names, np.column_stack(cols), y


# ---------- feature selection (all fitted on the training set only) ----------
def sse(X, y, F):
    m = LinearRegression().fit(X[:, F], y)
    return float(((y - m.predict(X[:, F])) ** 2).sum())


def path_order(Z, yz, l1_ratio):
    """Rank features by when they enter the Lasso / ElasticNet regularization path."""
    if l1_ratio == 1:
        _, coefs, _ = lasso_path(Z, yz, eps=1e-4, alphas=100)
    else:
        _, coefs, _ = enet_path(Z, yz, l1_ratio=l1_ratio, eps=1e-4, alphas=100)
    nz = coefs != 0  # shape (p, n_alphas), alphas descending
    enter = np.where(nz.any(1), nz.argmax(1), np.inf)
    mag = np.abs(coefs[np.arange(len(enter)), np.minimum(enter, coefs.shape[1] - 1).astype(int)])
    return sorted(range(len(enter)), key=lambda j: (enter[j], -mag[j]))


def forward_order(X, y):
    out, left = [], list(range(X.shape[1]))
    while left:
        best = min(left, key=lambda j: sse(X, y, out + [j]))
        out.append(best)
        left.remove(best)
    return out


def rfe_order(Z, yz):
    rank = RFE(LinearRegression(), n_features_to_select=1).fit(Z, yz).ranking_
    return list(np.argsort(rank))


def exhaustive(X, y):
    p = X.shape[1]
    if p > 15:  # ponytail: 2^p blowup; falls back to forward selection above 15 features
        return None
    return [list(min(combinations(range(p), k), key=lambda F: sse(X, y, list(F)))) for k in range(1, p + 1)]


# ---------- evaluation ----------
def evaluate(F, Xtr, ytr, Xte, yte):
    ols = sm.OLS(ytr, sm.add_constant(Xtr[:, F], has_constant="add")).fit()
    yh = sm.add_constant(Xte[:, F], has_constant="add") @ ols.params
    res = yte - yh
    nt, k = len(yte), len(F)
    r2 = 1 - (res ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    Xc = sm.add_constant(Xtr[:, F], has_constant="add")
    vif = [1.0] if k == 1 else [float(variance_inflation_factor(Xc, i)) for i in range(1, k + 1)]
    ci = ols.conf_int(0.05)
    return {
        "k": k, "F": [int(j) for j in F],
        "b": ols.params.tolist(), "se": ols.bse.tolist(), "t": ols.tvalues.tolist(),
        "p": ols.pvalues.tolist(), "lo": ci[:, 0].tolist(), "hi": ci[:, 1].tolist(), "vif": vif,
        "fstat": float(ols.fvalue), "fp": float(ols.f_pvalue), "df": int(ols.df_resid),
        "rmse": float(np.sqrt((res ** 2).mean())), "r2": float(r2), "mae": float(np.abs(res).mean()),
        "adj": float(1 - (1 - r2) * (nt - 1) / (nt - k - 1)) if nt - k - 1 > 0 else None,
        "yhTe": yh.tolist(),
    }


def analyze(url, target, test_idx):
    names, X, y = load_features(url, target)
    test = np.zeros(len(y), bool)
    test[test_idx] = True
    Xtr, ytr, Xte, yte = X[~test], y[~test], X[test], y[test]
    Z = StandardScaler().fit_transform(Xtr)
    yz = (ytr - ytr.mean()) / (ytr.std() or 1)
    prefixes = lambda order: [list(order[: k + 1]) for k in range(len(order))]
    fwd = forward_order(Xtr, ytr)
    subsets = {
        "lasso": prefixes(path_order(Z, yz, 1.0)),
        "enet": prefixes(path_order(Z, yz, 0.5)),
        "forward": prefixes(fwd),
        "rfe": prefixes(rfe_order(Z, yz)),
        "exhaustive": exhaustive(Xtr, ytr) or prefixes(fwd),
    }
    results = {m: [evaluate([int(j) for j in F], Xtr, ytr, Xte, yte) for F in S] for m, S in subsets.items()}
    return {"engine": "python", "features": names, "results": results}


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def api_analyze():
    if request.method == "OPTIONS":
        return "", 204
    body = request.get_json(force=True)
    try:
        return jsonify(analyze(body["url"], int(body["target"]), [int(i) for i in body["test_idx"]]))
    except Exception as e:  # bad URL / unparsable CSV / singular matrix -> frontend falls back to JS
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
