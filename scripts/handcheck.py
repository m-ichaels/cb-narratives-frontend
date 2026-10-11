"""Draw the hand-check sample (40 sentences, 10 per era, seed 20261006) from the policy statements, without looking at
any model output, and write data/handcheck_sample.csv with an empty hand_label column to be filled by reading.
`python scripts/handcheck.py` draws; `python scripts/handcheck.py --score` compares the filled labels with the
assignments in data/sentences.csv and writes results/handcheck.json.
"""

import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.ROOT, "data", "handcheck_sample.csv")
ERAS = [
    ("2015-01-01", "2017-12-31"),
    ("2018-01-01", "2020-12-31"),
    ("2021-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]


def draw():
    rng = np.random.default_rng(20261006)
    rows = []
    for d, sents in C.load_statements(policy_only=True):
        for i, s in enumerate(sents):
            rows.append({"date": d, "idx": i, "sentence": s})
    df = pd.DataFrame(rows)
    picks = []
    for a, b in ERAS:
        x = df[(df.date >= a) & (df.date <= b)]
        picks.append(x.iloc[rng.choice(len(x), 10, replace=False)])
    s = pd.concat(picks).sort_values(["date", "idx"]).reset_index(drop=True)
    s.insert(0, "id", range(1, len(s) + 1))
    s["hand_label"] = ""
    s.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(s)} sentences)")


def score():
    s = pd.read_csv(OUT)
    a = pd.read_csv(os.path.join(C.ROOT, "data", "sentences.csv"))
    m = s.merge(
        a[["date", "idx", "topic", "topic_emb", "override", "sim_max"]],
        on=["date", "idx"],
        how="left",
    )
    assert m["hand_label"].notna().all() and (m["hand_label"] != "").all(), "fill hand_label first"
    assert set(m["hand_label"]) <= set(C.TOPICS), set(m["hand_label"]) - set(C.TOPICS)
    res = {
        "n": int(len(m)),
        "agree_final": float((m.topic == m.hand_label).mean()),
        "agree_embedding_only": float((m.topic_emb == m.hand_label).mean()),
        "n_override": int((m.override.fillna("") != "").sum()),
        "agree_on_overridden": float(
            (
                m[m.override.fillna("") != ""].topic == m[m.override.fillna("") != ""].hand_label
            ).mean()
        )
        if (m.override.fillna("") != "").any()
        else None,
        "agree_on_embedding_assigned": float(
            (
                m[m.override.fillna("") == ""].topic == m[m.override.fillna("") == ""].hand_label
            ).mean()
        ),
        "disagreements": [
            {
                "id": int(r.id),
                "date": r.date,
                "hand": r.hand_label,
                "final": r.topic,
                "embedding": r.topic_emb,
                "sentence": r.sentence[:160],
            }
            for r in m.itertuples()
            if r.topic != r.hand_label
        ],
    }
    cm = pd.crosstab(m["hand_label"], m["topic"])
    res["confusion"] = {h: cm.loc[h].to_dict() for h in cm.index}
    os.makedirs(os.path.join(C.ROOT, "results"), exist_ok=True)
    json.dump(
        res,
        open(os.path.join(C.ROOT, "results", "handcheck.json"), "w", encoding="utf-8"),
        indent=1,
    )
    m.to_csv(os.path.join(C.ROOT, "data", "handcheck_scored.csv"), index=False, encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("disagreements", "confusion")}, indent=1
        )
    )
    print(cm.to_string())
    for d in res["disagreements"]:
        print(d)


if __name__ == "__main__":
    score() if "--score" in sys.argv else draw()
