"""The one chart: left, the aggregate hawkishness change against the USMPD 2y statement-window move (train grey,
test blue); right, the same statements split by topic (correlation of each topic's contribution with the 2y move,
all 94 statements with bootstrap CIs, and the partial correlation given the aggregate). Reads results/topic_tests.json
and data/statements.csv; writes results/figures/topic_split.png (+ .svg)."""

import os, sys, json
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RES = os.path.join(C.ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
# one blue accent; greys for everything else
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
ACCENT, DEEMPH = "#2a78d6", "#b8b7b0"
LABELS = {
    "inflation": "inflation",
    "labour": "labour market",
    "activity": "activity / growth",
    "financial": "financial conditions",
    "balance_sheet": "balance sheet",
    "policy_path": "policy path / guidance",
    "external": "external shocks",
    "mandate": "mandate / risks boilerplate",
    "voting": "vote roll-call",
    "other": "other",
}


def main():
    fam = [f.name for f in font_manager.fontManager.ttflist]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [f for f in ["Segoe UI", "DejaVu Sans"] if f in fam]
            or ["DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    R = json.load(open(os.path.join(RES, "topic_tests.json"), encoding="utf-8"))
    S = pd.read_csv(os.path.join(C.ROOT, "data", "statements.csv"))
    S = S[S.dh.notna() & S.usmpd_r2.notna()].reset_index(drop=True)
    tr, te = S.train.values, ~S.train.values
    a = R["aggregate"]
    inc = R["incremental_dc"]
    per = R["per_topic"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42}
    )
    # ---- left: scatter ----
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.8, ls="-", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax1.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax1.scatter(
        S.dh[tr],
        S.usmpd_r2[tr],
        s=34,
        color=DEEMPH,
        edgecolor=SURFACE,
        lw=1.0,
        zorder=3,
        label=f"train 2015-2023 (n = {tr.sum()}), corr {a['policy_train']['corr']:.2f}",
    )
    ax1.scatter(
        S.dh[te],
        S.usmpd_r2[te],
        s=40,
        color=ACCENT,
        edgecolor=SURFACE,
        lw=1.0,
        zorder=4,
        label=f"test 2024-26 (n = {te.sum()}), corr {a['policy_test']['corr']:.2f} [{a['policy_test']['ci'][0]:.2f}, {a['policy_test']['ci'][1]:.2f}]",
    )
    for m, col, z in ((tr, DEEMPH, 2), (te, ACCENT, 3)):
        b = np.polyfit(S.dh[m], S.usmpd_r2[m], 1)
        xx = np.array([S.dh[m].min(), S.dh[m].max()])
        ax1.plot(xx, np.polyval(b, xx), color=col, lw=1.6, zorder=z)
    ylo, yhi = S.usmpd_r2.min(), S.usmpd_r2.max()
    ax1.set_ylim(ylo - 2.5, yhi + 6.5)  # headroom for the annotation
    i = S.index[S.date == "2026-06-17"]
    if len(i):
        i = i[0]
        ax1.annotate(
            "17 Jun 2026: statement cut from 21 to 9 sentences\n(largest Δh in the test set)",
            (S.dh[i], S.usmpd_r2[i]),
            xytext=(S.dh[i] - 0.01, yhi + 1.8),
            ha="right",
            va="bottom",
            fontsize=7.6,
            color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4),
        )
    ax1.set_xlabel(
        "Δh: change in FOMC-RoBERTa hawkishness vs the previous statement\n(mean over sentences of P(hawk) − P(dove))"
    )
    ax1.set_ylabel("2y yield change in the USMPD statement window, bp (−10 to +20 min)")
    ax1.legend(loc="lower right", frameon=False, fontsize=7.8, handletextpad=0.3)
    ax1.set_title(
        "Aggregate: hawkishness change vs the 2y window move, 94 FOMC statements",
        loc="left",
        fontsize=10,
        color=INK,
        pad=26,
    )
    pt, ptr, pa = a["policy_test"], a["policy_train"], a["policy_all"]
    ax1.text(
        0.0,
        1.01,
        f"Out of sample, 2024-26 (n = {pt['n']}): corr {pt['corr']:.2f} [{pt['ci'][0]:.2f}, {pt['ci'][1]:.2f}], permutation p = {pt['perm_p']:.3f}\n"
        f"Training years 2015-23 (n = {ptr['n']}): corr {ptr['corr']:.2f}; all {pa['n']} statements: corr {pa['corr']:.2f}",
        transform=ax1.transAxes,
        fontsize=7.2,
        color=MUTED,
        va="bottom",
        linespacing=1.25,
    )

    # ---- right: per topic ----
    order = [t for t in C.TOPICS]
    yv = np.arange(len(order))[::-1]
    raw = np.array([per[t]["dc_corr_all"] for t in order])
    ci = np.array([per[t]["dc_corr_all_ci"] for t in order])
    part = np.array([per[t]["dc_partial_corr_given_dh_all"] for t in order])
    test_raw = np.array([per[t]["dc_corr_test"] for t in order])
    ax2.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax2.axvline(a["policy_all"]["corr"], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax2.set_ylim(-0.6, len(order) - 0.3)
    ax2.text(
        a["policy_all"]["corr"] + 0.012,
        yv[0] + 0.42,
        f"aggregate Δh, all 94: {a['policy_all']['corr']:.2f}",
        fontsize=7.2,
        color=MUTED,
        va="center",
    )
    ax2.barh(
        yv,
        raw,
        height=0.42,
        color=ACCENT,
        zorder=3,
        label="topic contribution Δcₖ: corr with the 2y move, all 94 statements (95 % bootstrap CI)",
    )
    ax2.errorbar(
        raw,
        yv,
        xerr=[raw - ci[:, 0], ci[:, 1] - raw],
        fmt="none",
        ecolor=INK2,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=4,
    )
    ax2.scatter(
        part,
        yv,
        s=46,
        marker="D",
        color=SURFACE,
        edgecolor=INK,
        lw=1.2,
        zorder=5,
        label="partial corr given the aggregate Δh (all 94)",
    )
    ax2.scatter(
        test_raw,
        yv,
        s=30,
        marker="o",
        color=SURFACE,
        edgecolor=ACCENT,
        lw=1.2,
        zorder=5,
        label=f"corr on the test set alone (n = {R['n_test']})",
    )
    ax2.set_yticks(yv)
    ax2.set_yticklabels(
        [f"{LABELS[t]}  ({per[t]['median_n_sentences']:.0f}/stmt)" for t in order],
        fontsize=8.2,
        color=INK2,
    )
    ax2.set_xlim(-0.55, 0.75)
    ax2.set_xlabel(
        "correlation with the 2y statement-window move   (label: median sentences per statement)"
    )
    ax2.set_title(
        "The same statements split by topic: Δh = Σₖ Δcₖ, contribution per topic",
        loc="left",
        fontsize=10,
        color=INK,
        pad=26,
    )
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.17),
        frameon=False,
        fontsize=7.4,
        handletextpad=0.3,
        borderaxespad=0,
    )
    lamB = inc["B"]["lambda"]
    lam_txt = "∞ (= aggregate only)" if lamB is None else f"{lamB:g}"
    ax2.text(
        0.0,
        1.01,
        f"Incremental test, strict OOS 2024-26 (n = {R['n_test']}): Δh alone R² {inc['A']['oos_r2']:.2f} → with topics {inc['B']['oos_r2']:.2f}\n"
        f"(ΔR² {inc['delta_oos_r2']:+.2f}, permutation p {inc['perm_p_delta_oos_r2']:.2f}, {inc['n_perm']} draws; ridge λ by leave-one-out on 2015-23 = {lam_txt})",
        transform=ax2.transAxes,
        fontsize=7.2,
        color=MUTED,
        va="bottom",
        linespacing=1.25,
    )
    fig.text(
        0.01,
        0.012,
        "Sources: FOMC statements 2015-01 to 2026-07 (95 policy statements); FOMC-RoBERTa (tim9510019 re-upload, commit d6acba5e, trained on text to Oct 2022); SF Fed USMPD statement windows (LSEG tick data).\n"
        "Topics: MiniLM embeddings against prototype sentences plus a keyword override (hand-check on 40 sentences in the repo: 38 agree). Code: github.com/m-ichaels/cb-narratives-frontend.",
        fontsize=6.6,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.3,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.25)
    fig.suptitle(
        "FOMC statement hawkishness and the 2-year yield: the aggregate change and its split by topic",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
        y=0.975,
    )
    fig.savefig(os.path.join(FIG, "topic_split.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "topic_split.svg"))
    print("wrote", os.path.join(FIG, "topic_split.png"))


if __name__ == "__main__":
    main()
