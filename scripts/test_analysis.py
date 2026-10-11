"""Checks on the statistical machinery in analysis.py (no model, no data needed): the hat-matrix leave-one-out
residuals equal explicit refits; lambda = inf is model A; the permutation null is centred near zero; the topic
override rules and the vote-fragment rule behave on known sentences."""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A, common as C


def test_loo_identity():
    rng = np.random.default_rng(0)
    n, k = 40, 6
    dh = rng.normal(size=n)
    G = rng.normal(size=(n, k))
    y = 2 * dh + G[:, 0] + rng.normal(size=n)
    for lam in [0.5, 10.0, np.inf]:
        m = A.Model(lam).fit(dh, G, y)
        r = m.loo_resid(y, m.predict(dh, G))
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            mi = A.Model(lam).fit(dh[keep], G[keep], y[keep])
            # explicit refit standardises G on n-1 rows; the hat-matrix identity holds for a fixed design, so compare
            # against a refit with the full-sample standardisation
            mi.mu, mi.sd = m.mu, m.sd
            X = mi._design(dh[keep], G[keep])
            P = np.zeros(X.shape[1])
            P[2:] = 1
            if np.isinf(lam):
                X = X[:, :2]
                P = P[:2]
            w = np.linalg.solve(X.T @ X + np.diag(P) * (0 if np.isinf(lam) else lam), X.T @ y[keep])
            xi = mi._design(dh[i : i + 1], G[i : i + 1])
            xi = xi[:, :2] if np.isinf(lam) else xi
            assert abs((y[i] - xi @ w)[0] - r[i]) < 1e-8, (lam, i)


def test_inf_is_model_A():
    rng = np.random.default_rng(1)
    n, k = 30, 5
    dh = rng.normal(size=n)
    G = rng.normal(size=(n, k))
    y = dh + rng.normal(size=n)
    m = A.Model(np.inf).fit(dh, G, y)
    b = np.polyfit(dh, y, 1)
    assert np.allclose(m.w, [b[1], b[0]])
    m2 = A.Model(1e9).fit(dh, G, y)
    assert np.allclose(m2.w[:2], [b[1], b[0]], atol=1e-4)


def test_permutation_null_centred():
    rng = np.random.default_rng(2)
    n, k = 90, 10
    dh = rng.normal(size=n)
    G = rng.normal(size=(n, k))
    y = 3 * dh + rng.normal(size=n) * 5
    tr = np.arange(n) < 68
    te = ~tr
    res, _ = A.ab_test(dh[tr], G[tr], y[tr], dh[te], G[te], y[te])
    assert abs(res["delta_oos_r2"]) < 0.3


def test_overrides():
    assert C.override_topic("Voting for the monetary policy action were Jerome H.") == "voting"
    assert C.override_topic("Bostic; Lael Brainard; Loretta J.") == "voting"
    assert (
        C.override_topic(
            "Waller, who preferred to lower the target range for the federal funds rate by 1/4 percentage point at this meeting."
        )
        == "voting"
    )
    assert (
        C.override_topic(
            "The Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities."
        )
        == "balance_sheet"
    )
    assert (
        C.override_topic(
            "Inflation is elevated, in part reflecting the recent increase in global energy prices."
        )
        == "external"
    )
    assert (
        C.override_topic(
            "On a 12-month basis, inflation has declined recently and, like the measure excluding food and energy prices, is running somewhat below 2 percent."
        )
        is None
    )
    assert (
        C.override_topic(
            "The Committee's assessments will take into account a wide range of information, including readings on public health, labor market conditions, inflation pressures and inflation expectations, and financial and international developments."
        )
        == "mandate"
    )
    assert (
        C.override_topic(
            "In support of its goals, the Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent."
        )
        == "policy_path"
    )
    assert (
        C.override_topic(
            "Job gains have been strong, on average, in recent months, and the unemployment rate has stayed low."
        )
        is None
    )
    assert (
        C.override_topic(
            "Overall financial conditions remain accommodative, in part reflecting policy measures to support the economy."
        )
        is None
    )


if __name__ == "__main__":
    for f in [
        test_loo_identity,
        test_inf_is_model_A,
        test_permutation_null_centred,
        test_overrides,
    ]:
        f()
        print("ok", f.__name__)
