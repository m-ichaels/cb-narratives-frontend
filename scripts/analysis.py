"""Tests on the per-topic split of FOMC statements against the USMPD 2y statement-window move.

  (a) aggregate: corr(dh, 2y) on train / test, recomputed on the 95 policy statements, beside the rates-curve-nlp numbers
  (b) incremental information: model A = 2y ~ dh; model B = 2y ~ dh + sum_k g_k dc_k with ridge on g only
      (lambda -> inf recovers A); lambda by leave-one-out on the training statements (<= 2023-12-31); strict
      out-of-sample on 2024-26; permutation p-value for the OOS improvement (row-shuffle of the topic block)
  (c) stability: drop-one-topic and only-one-topic aggregates; drop-one-statement jackknife of the test correlation
  (d) what broke: undefined topic changes, n per topic, mix-vs-tone decomposition, vote fragments, the 2026 format
      break, the near-absence of the Fulcrum narratives from statements, RoBERTa's training cut-off
Writes results/topic_tests.json and results/topic_tests.md.
"""

import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DATA = os.path.join(C.ROOT, "data")
RES = os.path.join(C.ROOT, "results")
os.makedirs(RES, exist_ok=True)
TOPICS = C.TOPICS
LAMBDAS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, np.inf]
N_PERM = 2000
N_BOOT = 2000
SUCCESSORS_OF_NON_POLICY = [
    "2019-10-30",
    "2020-04-29",
    "2020-09-16",
    "2025-09-17",
]  # policy statements whose previous *file* is not a policy statement


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def boot_ci(a, b, B=N_BOOT, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        ix = rng.integers(0, len(a), len(a))
        if np.std(a[ix]) > 0 and np.std(b[ix]) > 0:
            v.append(np.corrcoef(a[ix], b[ix])[0, 1])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def perm_p(a, b, n=5000, seed=1):
    """two-sided permutation p-value of corr(a, b) (as in rates-curve-nlp's tools/nlp_features.py)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    rng = np.random.default_rng(seed)
    c = abs(np.corrcoef(a, b)[0, 1])
    return float(np.mean([abs(np.corrcoef(a, rng.permutation(b))[0, 1]) >= c for _ in range(n)]))


# ---- ridge with an unpenalised block (intercept + dh) and a penalised topic block ---------------------------------
class Model:
    """y = a + b*dh + G @ g; penalty lam * |g|^2 on the standardised topic block; lam = inf -> model A."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, dh, G, y):
        self.mu, self.sd = G.mean(0), G.std(0) + 1e-12
        X = self._design(dh, G)
        p = X.shape[1]
        P = np.zeros(p)
        P[2:] = 1.0
        if np.isinf(self.lam):
            X = X[:, :2]
            P = P[:2]
        A = X.T @ X + np.diag(P) * (0 if np.isinf(self.lam) else self.lam)
        self.w = np.linalg.solve(A, X.T @ y)
        self.H = X @ np.linalg.solve(A, X.T)
        return self

    def _design(self, dh, G):
        return np.column_stack([np.ones(len(dh)), dh, (G - self.mu) / self.sd])

    def predict(self, dh, G):
        X = self._design(dh, G)
        return X[:, :2] @ self.w if np.isinf(self.lam) else X @ self.w

    def loo_resid(self, y, yhat):  # leave-one-out residuals of a linear smoother: e_i / (1 - H_ii)
        return (y - yhat) / (1 - np.diag(self.H))


def fit_select(dh, G, y, lambdas=LAMBDAS):
    """lambda by LOO mean-squared error on the training set; returns (model, lam, loo_r2, loo_corr)"""
    best = None
    for lam in lambdas:
        m = Model(lam).fit(dh, G, y)
        r = m.loo_resid(y, m.predict(dh, G))
        mse = float(np.mean(r**2))
        if best is None or mse < best[0]:
            best = (mse, lam, m, r)
    mse, lam, m, r = best
    loo_pred = y - r
    return m, lam, float(1 - mse / y.var()), corr(loo_pred, y)


def r2_oos(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def ab_test(dh_tr, G_tr, y_tr, dh_te, G_te, y_te):
    mA, lamA, looA_r2, looA_c = fit_select(dh_tr, G_tr, y_tr, [np.inf])
    mB, lamB, looB_r2, looB_c = fit_select(dh_tr, G_tr, y_tr)
    pA, pB = mA.predict(dh_te, G_te), mB.predict(dh_te, G_te)
    return {
        "A": {
            "lambda": "inf",
            "loo_r2_train": looA_r2,
            "loo_corr_train": looA_c,
            "oos_corr": corr(pA, y_te),
            "oos_r2": r2_oos(y_te, pA),
        },
        "B": {
            "lambda": ("inf" if np.isinf(lamB) else lamB),
            "loo_r2_train": looB_r2,
            "loo_corr_train": looB_c,
            "oos_corr": corr(pB, y_te),
            "oos_r2": r2_oos(y_te, pB),
            "g_std_units": {
                t: float(w)
                for t, w in zip(TOPICS, (mB.w[2:] if not np.isinf(lamB) else np.zeros(len(TOPICS))))
            },
        },
        "delta_oos_r2": r2_oos(y_te, pB) - r2_oos(y_te, pA),
        "delta_oos_corr": corr(pB, y_te) - corr(pA, y_te),
        "delta_loo_r2_train": looB_r2 - looA_r2,
    }, (pA, pB)


def main():
    S = pd.read_csv(os.path.join(DATA, "statements.csv"))
    T = pd.read_csv(os.path.join(DATA, "statement_topics.csv"))
    S = S[S["dh"].notna() & S["usmpd_r2"].notna()].reset_index(
        drop=True
    )  # 94 statements with a predecessor and a window
    Wc = T.pivot(index="date", columns="topic", values="dc").reindex(S["date"])[TOPICS].fillna(0.0)
    Wdh = T.pivot(index="date", columns="topic", values="dh").reindex(S["date"])[TOPICS]
    Wsh = (
        T.pivot(index="date", columns="topic", values="dshare")
        .reindex(S["date"])[TOPICS]
        .fillna(0.0)
    )
    Wn = T.pivot(index="date", columns="topic", values="n").reindex(S["date"])[TOPICS]
    assert np.allclose(Wc.sum(1).values, S["dh"].values)
    tr, te = S["train"].values, ~S["train"].values
    y = S["usmpd_r2"].values
    dh = S["dh"].values
    out = {
        "n_statements": int(len(S)),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "train_end": C.TRAIN_END,
        "test_dates": S.loc[te, "date"].tolist(),
    }

    # ---- (a) aggregate, recomputed, and the rates-curve-nlp numbers ----
    own = pd.read_csv(C.RCN_FEATURES)
    own_te = own[(own.date > C.TRAIN_END)]
    own_fo = own_te[own_te.bank == "FOMC"]
    own_ecb = own_te[own_te.bank == "ECB"]
    sig = json.load(open(C.RCN_SIGNAL, encoding="utf-8"))["feature_sets"]["roberta_doc_only"]["r2"]
    out["aggregate"] = {
        "rcn_pooled_test": {
            "corr": sig["corr"],
            "ci": sig["corr_ci"],
            "n": sig["n"],
            "perm_p": sig["placebo_p_value"],
            "r2_oos": sig["r2_oos"],
            "source": "rates-curve-nlp/results/nlp_signal.json: feature_sets.roberta_doc_only.r2",
        },
        "rcn_fomc_test": {
            "corr": corr(own_fo.rob_dh, own_fo.r2),
            "n": int(len(own_fo)),
            "ci": boot_ci(own_fo.rob_dh, own_fo.r2),
            "source": "rates-curve-nlp/data/derived/nlp_features.csv, bank == FOMC, date > 2023-12-31 (includes the 2025-08-22 framework note, src = daily)",
        },
        "rcn_ecb_test": {
            "corr": corr(own_ecb.rob_dh, own_ecb.r2),
            "n": int(len(own_ecb)),
            "ci": boot_ci(own_ecb.rob_dh, own_ecb.r2),
            "source": "same file, bank == ECB (daily CMT 2y changes, not event windows)",
        },
        "policy_test": {
            "corr": corr(dh[te], y[te]),
            "n": int(te.sum()),
            "ci": boot_ci(dh[te], y[te]),
            "perm_p": perm_p(dh[te], y[te]),
            "note": "95 policy statements, dh vs the previous policy statement, USMPD statement window",
        },
        "policy_train": {
            "corr": corr(dh[tr], y[tr]),
            "n": int(tr.sum()),
            "ci": boot_ci(dh[tr], y[tr]),
            "perm_p": perm_p(dh[tr], y[tr]),
        },
        "policy_all": {
            "corr": corr(dh, y),
            "n": int(len(S)),
            "ci": boot_ci(dh, y),
            "perm_p": perm_p(dh, y),
        },
        "other_tenors_policy_test": {
            t: corr(dh[te], S.loc[te, "usmpd_" + t]) for t in ["r5", "r10", "r30"]
        },
        "crosscheck_rcn_rob_h_max_abs_diff": float(np.nanmax(np.abs(S["h"] - S["rcn_rob_h"]))),
        # rates-curve-nlp's dh is against the previous *file*; the four statements that follow a non-policy file differ by construction
        "crosscheck_rcn_rob_dh_max_abs_diff_excl_successors_of_non_policy_files": float(
            np.nanmax(np.abs((S["dh"] - S["rcn_rob_dh"])[~S.date.isin(SUCCESSORS_OF_NON_POLICY)]))
        ),
        "successors_of_non_policy_files": SUCCESSORS_OF_NON_POLICY,
    }

    # ---- per-topic correlations (raw and partial given dh), test and all ----
    def partial(x, z, yy):
        m = np.isfinite(x) & np.isfinite(z) & np.isfinite(yy)
        x, z, yy = x[m], z[m], yy[m]
        if m.sum() < 5:
            return np.nan
        Z = np.column_stack([np.ones(m.sum()), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = yy - Z @ np.linalg.lstsq(Z, yy, rcond=None)[0]
        return corr(rx, ry)

    per = {}
    for t in TOPICS:
        c = Wc[t].values
        d = Wdh[t].values
        per[t] = {
            "n_present_both_test": int(np.isfinite(d[te]).sum()),
            "n_present_both_all": int(np.isfinite(d).sum()),
            "median_n_sentences": float(np.nanmedian(Wn[t].values)),
            "share_mean": float((T[T.topic == t].groupby("date")["share"].mean()).mean()),
            "dc_corr_test": corr(c[te], y[te]),
            "dc_corr_test_ci": boot_ci(c[te], y[te]),
            "dc_corr_all": corr(c, y),
            "dc_corr_all_ci": boot_ci(c, y),
            "dc_partial_corr_given_dh_test": partial(c[te], dh[te], y[te]),
            "dc_partial_corr_given_dh_all": partial(c, dh, y),
            "dh_topic_corr_test": corr(d[te], y[te]),
            "dh_topic_corr_all": corr(d, y),
            "dh_topic_corr_all_ci": boot_ci(d, y),
            "dh_topic_partial_given_dh_all": partial(d, dh, y),
        }
    out["per_topic"] = per

    # ---- (b) incremental information ----
    G = Wc.values
    res, (pA, pB) = ab_test(dh[tr], G[tr], y[tr], dh[te], G[te], y[te])
    rng = np.random.default_rng(7)
    null_r2, null_c = [], []
    for _ in range(N_PERM):
        Gp = G.copy()
        Gp[tr] = G[tr][rng.permutation(tr.sum())]
        Gp[te] = G[te][rng.permutation(te.sum())]
        r, _ = ab_test(dh[tr], Gp[tr], y[tr], dh[te], Gp[te], y[te])
        null_r2.append(r["delta_oos_r2"])
        null_c.append(r["delta_oos_corr"])
    res["perm_p_delta_oos_r2"] = float(np.mean(np.array(null_r2) >= res["delta_oos_r2"]))
    res["perm_p_delta_oos_corr"] = float(np.mean(np.array(null_c) >= res["delta_oos_corr"]))
    res["perm_null_delta_oos_r2_p95"] = float(np.percentile(null_r2, 95))
    res["n_perm"] = N_PERM
    res["note"] = (
        "B nests A: lambda = inf gives A; the topic block is row-shuffled within train and within test for the null"
    )
    out["incremental_dc"] = res
    # variant: topic-level tone changes dh_k (undefined -> 0, plus an 'undefined' indicator per topic)
    D = Wdh.values
    und = (~np.isfinite(D)).astype(float)
    D0 = np.nan_to_num(D)
    G2 = np.column_stack([D0, und[:, und.std(0) > 0]])
    res2, _ = ab_test(dh[tr], G2[tr], y[tr], dh[te], G2[te], y[te])
    res2["note"] = (
        "topic block = dh_k with NaN -> 0 and an undefined indicator per intermittent topic"
    )
    res2["B"].pop("g_std_units", None)
    out["incremental_dh_topic"] = res2
    # variant: topic shares (prevalence) changes
    res3, _ = ab_test(dh[tr], Wsh.values[tr], y[tr], dh[te], Wsh.values[te], y[te])
    res3["note"] = "topic block = dshare_k (prevalence change)"
    res3["B"].pop("g_std_units", None)
    out["incremental_dshare"] = res3
    # the same incremental test, leave-one-out over all 94 statements (no train/test split), for the n-starved reader
    looA = np.full(len(S), np.nan)
    looB = np.full(len(S), np.nan)
    for i in range(len(S)):
        m = np.ones(len(S), bool)
        m[i] = False
        mA, *_ = fit_select(dh[m], G[m], y[m], [np.inf])
        mB, *_ = fit_select(dh[m], G[m], y[m])
        looA[i] = mA.predict(dh[i : i + 1], G[i : i + 1])[0]
        looB[i] = mB.predict(dh[i : i + 1], G[i : i + 1])[0]
    out["incremental_dc_loo_all"] = {
        "A_corr": corr(looA, y),
        "B_corr": corr(looB, y),
        "A_r2": r2_oos(y, looA),
        "B_r2": r2_oos(y, looB),
        "n": int(len(S)),
    }

    # ---- (c) stability ----
    sent = pd.read_csv(os.path.join(DATA, "sentences.csv"))
    sent = sent[sent.policy]
    dates = S["date"].tolist()
    prev = dict(zip(S["date"], S["prev_date"]))

    def agg_excl(excl=None, only=None):
        hh = {}
        for d, g in sent.groupby("date"):
            x = g[~g.topic.isin(excl)] if excl else (g[g.topic == only] if only else g)
            hh[d] = x.score.mean() if len(x) else np.nan
        return np.array([hh.get(d, np.nan) - hh.get(prev[d], np.nan) for d in dates])

    stab = {"drop_one_topic": {}, "only_one_topic": {}}
    for t in TOPICS:
        v = agg_excl(excl=[t])
        stab["drop_one_topic"][t] = {
            "corr_test": corr(v[te], y[te]),
            "corr_all": corr(v, y),
            "n_test": int(np.isfinite(v[te]).sum()),
        }
        v = agg_excl(only=t)
        stab["only_one_topic"][t] = {
            "corr_test": corr(v[te], y[te]),
            "corr_all": corr(v, y),
            "n_test": int(np.isfinite(v[te]).sum()),
            "n_all": int(np.isfinite(v).sum()),
        }
    v = agg_excl(excl=["voting", "other", "mandate"])
    stab["substantive_only"] = {
        "corr_test": corr(v[te], y[te]),
        "corr_all": corr(v, y),
        "note": "aggregate over the seven substantive topics only",
    }
    jack = []
    for i in np.where(te)[0]:
        m = te.copy()
        m[i] = False
        jack.append(
            {
                "dropped": S.date[i],
                "corr_test": corr(dh[m], y[m]),
                "dh": float(dh[i]),
                "r2_bp": float(y[i]),
            }
        )
    stab["jackknife_test"] = sorted(jack, key=lambda r: r["corr_test"])
    out["stability"] = stab

    # ---- (d) what broke ----
    broke = {}
    st = T[T.status != "first"]
    broke["undefined_topic_dh"] = {
        "pairs_total": int(len(st)),
        "pairs_undefined": int((st.status != "both").sum()),
        "by_topic": {
            t: {s: int(v) for s, v in st[st.topic == t].status.value_counts().items()}
            for t in TOPICS
        },
    }
    broke["median_sentences_per_topic"] = {t: float(np.nanmedian(Wn[t])) for t in TOPICS}
    # mix vs tone: dh = sum_k share_k dh_k (tone) + sum_k h_prev_k dshare_k (mix) + cross terms when a topic appears/vanishes
    tone = np.zeros(len(S))
    mix = np.zeros(len(S))
    for i, d in enumerate(dates):
        g = T[T.date == d]
        for r in g.itertuples():
            if r.status == "both":
                tone[i] += r.share * r.dh
                mix[i] += r.h_prev * r.dshare
            else:
                mix[i] += r.dc
    broke["mix_vs_tone"] = {
        "var_share_mix": float(np.var(mix) / np.var(dh)),
        "var_share_tone": float(np.var(tone) / np.var(dh)),
        "corr_mix_tone": corr(mix, tone),
        "corr_mix_r2_test": corr(mix[te], y[te]),
        "corr_tone_r2_test": corr(tone[te], y[te]),
        "corr_mix_r2_all": corr(mix, y),
        "corr_tone_r2_all": corr(tone, y),
        "corr_dh_dn_sent_all": corr(dh, S["n_sent"] - S["n_sent_prev"]),
    }
    vf = sent[sent.topic == "voting"]
    broke["vote_fragments"] = {
        "share_of_sentences": float(len(vf) / len(sent)),
        "mean_score": float(vf.score.mean()),
        "std_score": float(vf.score.std()),
        "corr_test_without_voting": stab["drop_one_topic"]["voting"]["corr_test"],
        "example": vf.sentence.iloc[0] if len(vf) else "",
    }
    fb = S[S.date.isin(["2026-04-29", "2026-06-17", "2026-07-29"])][
        ["date", "n_sent", "h", "dh", "usmpd_r2"]
    ]
    broke["format_break_2026"] = {
        "rows": fb.to_dict("records"),
        "jackknife_corr_without_2026-06-17": next(
            r["corr_test"] for r in jack if r["dropped"] == "2026-06-17"
        ),
    }
    allsent = pd.read_csv(os.path.join(DATA, "sentences.csv"))
    kw = {
        "tariff": r"tariff",
        "AI": r"artificial intelligence|\bAI\b",
        "Middle East / Hormuz": r"middle east|hormuz|iran",
        "energy": r"\benergy\b",
        "net exports": r"net exports",
        "Ukraine / war": r"ukraine|russia|\bwar\b",
        "pandemic": r"pandemic|coronavirus|covid|virus",
    }
    broke["narratives_in_statements"] = {
        k: {
            "n_sentences": int(allsent.sentence.str.contains(p, case=False, regex=True).sum()),
            "n_statements": int(
                allsent[allsent.sentence.str.contains(p, case=False, regex=True)].date.nunique()
            ),
        }
        for k, p in kw.items()
    }
    ext = sent[(sent.topic == "external") & (sent.date >= "2022-11-01")][
        ["date", "score", "sentence"]
    ]
    broke["external_after_roberta_cutoff"] = {
        "n_sentences": int(len(ext)),
        "n_statements": int(ext.date.nunique()),
        "rows": [
            {"date": r.date, "score": round(float(r.score), 3), "sentence": r.sentence[:200]}
            for r in ext.itertuples()
        ],
    }
    broke["roberta"] = {
        "model": "tim9510019/FOMC-RoBERTa",
        "commit": C.ROB_COMMIT,
        "training_text_to": "2022-10 (Shah, Paturi, Chava 2023)",
        "checkpoint": "2023-09-26",
        "n_policy_statements_after_cutoff": int((S.date > "2022-10-31").sum()),
    }
    out["what_broke"] = broke
    json.dump(
        out,
        open(os.path.join(RES, "topic_tests.json"), "w", encoding="utf-8"),
        indent=1,
        default=float,
    )
    write_md(out)
    print(
        json.dumps(
            {
                "aggregate_policy_test": out["aggregate"]["policy_test"],
                "incremental": {k: v for k, v in res.items() if k != "B"},
                "B": res["B"],
            },
            indent=1,
            default=float,
        )
    )


def write_md(o):
    a = o["aggregate"]
    r = o["incremental_dc"]
    L = []
    L.append("# Per-topic split of FOMC statements: results\n")
    L.append(
        f"Statements: {o['n_statements']} policy statements with a predecessor and a USMPD window; train n = {o['n_train']} (to {o['train_end']}), test n = {o['n_test']}.\n"
    )
    L.append("## (a) Aggregate dh vs the 2y statement-window move\n")
    L.append("| sample | corr | 95 % CI | n | perm p |\n|---|---|---|---|---|")
    L.append(
        f"| rates-curve-nlp, pooled FOMC + ECB test (nlp_signal.json) | {a['rcn_pooled_test']['corr']:.4f} | [{a['rcn_pooled_test']['ci'][0]:.4f}, {a['rcn_pooled_test']['ci'][1]:.4f}] | {a['rcn_pooled_test']['n']} | {a['rcn_pooled_test']['perm_p']:.4f} |"
    )
    L.append(
        f"| rates-curve-nlp, FOMC test only (nlp_features.csv) | {a['rcn_fomc_test']['corr']:.3f} | [{a['rcn_fomc_test']['ci'][0]:.2f}, {a['rcn_fomc_test']['ci'][1]:.2f}] | {a['rcn_fomc_test']['n']} | |"
    )
    L.append(
        f"| rates-curve-nlp, ECB test only, daily changes | {a['rcn_ecb_test']['corr']:.3f} | [{a['rcn_ecb_test']['ci'][0]:.2f}, {a['rcn_ecb_test']['ci'][1]:.2f}] | {a['rcn_ecb_test']['n']} | |"
    )
    L.append(
        f"| this repo, policy statements, test | {a['policy_test']['corr']:.3f} | [{a['policy_test']['ci'][0]:.2f}, {a['policy_test']['ci'][1]:.2f}] | {a['policy_test']['n']} | {a['policy_test']['perm_p']:.3f} |"
    )
    L.append(
        f"| this repo, policy statements, train | {a['policy_train']['corr']:.3f} | [{a['policy_train']['ci'][0]:.2f}, {a['policy_train']['ci'][1]:.2f}] | {a['policy_train']['n']} | {a['policy_train']['perm_p']:.3f} |"
    )
    L.append(
        f"| this repo, policy statements, all | {a['policy_all']['corr']:.3f} | [{a['policy_all']['ci'][0]:.2f}, {a['policy_all']['ci'][1]:.2f}] | {a['policy_all']['n']} | {a['policy_all']['perm_p']:.3f} |\n"
    )
    L.append(
        f"Cross-check: max |h - rates-curve-nlp's rob_h| = {a['crosscheck_rcn_rob_h_max_abs_diff']:.1e}; max |dh - rates-curve-nlp's rob_dh| = {a['crosscheck_rcn_rob_dh_max_abs_diff_excl_successors_of_non_policy_files']:.1e} excluding the four statements that follow a non-policy file ({', '.join(a['successors_of_non_policy_files'])}), whose 'previous statement' differs by construction. Other tenors (test): "
        + ", ".join(f"{k} {v:.3f}" for k, v in a["other_tenors_policy_test"].items())
        + ".\n"
    )
    L.append("## Per topic\n")
    ub = o["what_broke"]["undefined_topic_dh"]["by_topic"]
    n_pairs = o["what_broke"]["undefined_topic_dh"]["pairs_total"] // len(TOPICS)
    L.append(
        f"| topic | median n/stmt | mean share | undefined dh_k (of {n_pairs}) | corr(dc_k, 2y) test [CI] | partial | corr(dc_k, 2y) all [CI] | partial | corr(dh_k, 2y) all (n) |\n|---|---|---|---|---|---|---|---|---|"
    )
    for t, p in o["per_topic"].items():
        und = sum(v for s, v in ub[t].items() if s != "both")
        L.append(
            f"| {t} | {p['median_n_sentences']:.0f} | {p['share_mean']:.2f} | {und} | {p['dc_corr_test']:.2f} [{p['dc_corr_test_ci'][0]:.2f}, {p['dc_corr_test_ci'][1]:.2f}] | {p['dc_partial_corr_given_dh_test']:.2f} | {p['dc_corr_all']:.2f} [{p['dc_corr_all_ci'][0]:.2f}, {p['dc_corr_all_ci'][1]:.2f}] | {p['dc_partial_corr_given_dh_all']:.2f} | {p['dh_topic_corr_all']:.2f} ({p['n_present_both_all']}) |"
        )
    L.append(
        "\n## (b) Incremental information: A = dh only; B = dh + per-topic contributions dc_k (ridge on the topic block)\n"
    )
    L.append(
        "| model | lambda | LOO R2 train | LOO corr train | OOS corr | OOS R2 |\n|---|---|---|---|---|---|"
    )
    for k in ["A", "B"]:
        m = r[k]
        L.append(
            f"| {k} | {m['lambda']} | {m['loo_r2_train']:.3f} | {m['loo_corr_train']:.3f} | {m['oos_corr']:.3f} | {m['oos_r2']:.3f} |"
        )
    L.append(
        f"\nOOS delta R2 (B - A) = {r['delta_oos_r2']:+.3f}, delta corr = {r['delta_oos_corr']:+.3f}; permutation p (topic block row-shuffled, {r['n_perm']} draws) = {r['perm_p_delta_oos_r2']:.3f} for delta R2, {r['perm_p_delta_oos_corr']:.3f} for delta corr; null 95th percentile of delta R2 = {r['perm_null_delta_oos_r2_p95']:+.3f}."
    )
    L.append(
        "Topic weights in B (per standardised dc_k, bp): "
        + ", ".join(f"{t} {w:+.2f}" for t, w in r["B"]["g_std_units"].items())
        + "."
    )
    r2_ = o["incremental_dh_topic"]
    r3 = o["incremental_dshare"]
    r4 = o["incremental_dc_loo_all"]
    L.append(
        f"\nVariants: topic block = tone changes dh_k (NaN -> 0 + indicators): OOS corr A {r2_['A']['oos_corr']:.3f} -> B {r2_['B']['oos_corr']:.3f}, delta R2 {r2_['delta_oos_r2']:+.3f}; topic block = prevalence changes dshare_k: B {r3['B']['oos_corr']:.3f}, delta R2 {r3['delta_oos_r2']:+.3f}. Leave-one-out over all {r4['n']} statements (no split): corr A {r4['A_corr']:.3f} -> B {r4['B_corr']:.3f}, R2 {r4['A_r2']:.3f} -> {r4['B_r2']:.3f}.\n"
    )
    s = o["stability"]
    L.append("## (c) Stability\n")
    L.append(
        "| topic | drop it: corr test | drop it: corr all | only it: corr test (n) | only it: corr all (n) |\n|---|---|---|---|---|"
    )
    for t in TOPICS:
        d, oo = s["drop_one_topic"][t], s["only_one_topic"][t]
        L.append(
            f"| {t} | {d['corr_test']:.3f} | {d['corr_all']:.3f} | {oo['corr_test']:.2f} ({oo['n_test']}) | {oo['corr_all']:.2f} ({oo['n_all']}) |"
        )
    L.append(
        f"\nSubstantive topics only (no voting / mandate / other): corr test {s['substantive_only']['corr_test']:.3f}, all {s['substantive_only']['corr_all']:.3f}."
    )
    j = s["jackknife_test"]
    L.append(
        f"Jackknife of the test correlation ({o['n_test']} statements): min {j[0]['corr_test']:.3f} (dropping {j[0]['dropped']}), max {j[-1]['corr_test']:.3f} (dropping {j[-1]['dropped']}).\n"
    )
    b = o["what_broke"]
    L.append("## (d) What broke\n")
    u = b["undefined_topic_dh"]
    L.append(
        f"- Topic-level dh_k is undefined in {u['pairs_undefined']} of {u['pairs_total']} (statement, topic) pairs; per topic: "
        + "; ".join(
            f"{t}: " + ", ".join(f"{s} {v}" for s, v in d.items()) for t, d in u["by_topic"].items()
        )
        + "."
    )
    L.append(
        "- Median sentences per topic per statement: "
        + ", ".join(f"{t} {v:.0f}" for t, v in b["median_sentences_per_topic"].items())
        + "."
    )
    m = b["mix_vs_tone"]
    L.append(
        f"- Mix vs tone: dh = tone (share-weighted dh_k) + mix (topic entering / leaving / changing share). Variance share of dh: mix {m['var_share_mix']:.2f}, tone {m['var_share_tone']:.2f} (corr {m['corr_mix_tone']:.2f}). corr with the 2y move, test: mix {m['corr_mix_r2_test']:.2f}, tone {m['corr_tone_r2_test']:.2f}; all: mix {m['corr_mix_r2_all']:.2f}, tone {m['corr_tone_r2_all']:.2f}. corr(dh, change in sentence count) = {m['corr_dh_dn_sent_all']:.2f}."
    )
    v = b["vote_fragments"]
    L.append(
        f'- Vote roll-call fragments are {v["share_of_sentences"]:.0%} of all scored sentences (mean score {v["mean_score"]:+.2f}, sd {v["std_score"]:.2f}); e.g. "{v["example"]}". Test corr without them: {v["corr_test_without_voting"]:.3f}.'
    )
    f = b["format_break_2026"]
    L.append(
        "- 2026 format break: "
        + "; ".join(
            f"{r['date']}: {r['n_sent']} sentences, h {r['h']:+.3f}, dh {r['dh']:+.3f}, 2y {r['usmpd_r2']:+.1f} bp"
            for r in f["rows"]
        )
        + f". Test corr without 2026-06-17: {f['jackknife_corr_without_2026-06-17']:.3f}."
    )
    L.append(
        "- The Fulcrum narratives in the statements (sentences / statements over all 99 files): "
        + ", ".join(
            f"{k} {v['n_sentences']}/{v['n_statements']}"
            for k, v in b["narratives_in_statements"].items()
        )
        + "."
    )
    e = b["external_after_roberta_cutoff"]
    L.append(
        f"- External-shock sentences after FOMC-RoBERTa's training cut-off (Oct 2022): {e['n_sentences']} sentences in {e['n_statements']} statements; scores: "
        + "; ".join(f'{r["date"]} {r["score"]:+.2f} "{r["sentence"][:90]}..."' for r in e["rows"])
        + "."
    )
    open(os.path.join(RES, "topic_tests.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
