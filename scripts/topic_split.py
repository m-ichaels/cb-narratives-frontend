"""Score every sentence of every FOMC statement with FOMC-RoBERTa, assign each sentence a topic, aggregate per
statement and per topic, and write tidy CSVs.

    python scripts/topic_split.py [--limit N] [--bs 8]

Outputs (data/):
  cache/rob_scores.json      per-statement sentence scores (resumable; one entry per file)
  sentences.csv              one row per sentence: date, policy flag, sentence, RoBERTa score, embedding topic,
                             cosine to every centroid, keyword override, final topic
  statement_topics.csv       tidy: one row per (statement, topic): n, share, h (mean score), c (= share * h),
                             and the changes versus the previous policy statement (dh is NaN when the topic is absent
                             in either statement; dc and dshare are always defined; dc sums to the aggregate dh)
  statements.csv             one row per policy statement: n_sent, h, dh, plus rates-curve-nlp's rob_h / rob_dh for the
                             cross-check, and the USMPD 2y/5y/10y/30y statement-window moves (bp)
Heavy: roberta-large on CPU, about 25 minutes for the 2,065 sentences (cached, resumable per statement).
"""

import os, sys, json, time, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DATA = os.path.join(C.ROOT, "data")
CACHE = os.path.join(DATA, "cache")
os.makedirs(CACHE, exist_ok=True)


class Rob:
    """FOMC-RoBERTa from the pinned local snapshot; per-sentence score = P(hawk) - P(dove), as in rates-curve-nlp's HF class."""

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(C.ROB_SNAPSHOT)
        self.m = AutoModelForSequenceClassification.from_pretrained(C.ROB_SNAPSHOT).eval()
        labels = {v.lower(): k for k, v in self.m.config.id2label.items()}
        self.i_dove = labels.get("dovish", 0)
        self.i_hawk = labels.get("hawkish", 1)  # gtfintechlab: 0 dovish, 1 hawkish, 2 neutral

    def scores(self, sents, bs=8):
        out = []
        for k in range(0, len(sents), bs):
            x = self.tok(
                sents[k : k + bs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with self.torch.no_grad():
                p = self.torch.softmax(self.m(**x).logits, -1).numpy()
            out.extend((p[:, self.i_hawk] - p[:, self.i_dove]).tolist())
        return out


def _load_cache(f):
    """Per-statement checkpoint. A file truncated by a mid-write kill is treated as empty rather than fatal."""
    if not os.path.exists(f):
        return {}
    try:
        return json.load(open(f, encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  checkpoint {f} unreadable, rescoring everything", flush=True)
        return {}


def _save_cache(cache, f):
    """Atomic: write a sibling temp file, then rename over the checkpoint (os.replace is atomic on NTFS)."""
    tmp = f + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    os.replace(tmp, f)


def score_all(docs, bs):
    cache_f = os.path.join(CACHE, "rob_scores.json")
    cache = _load_cache(cache_f)
    todo = [(d, s) for d, s in docs if d not in cache or len(cache[d]) != len(s)]
    print(
        f"{len(docs)} statements, {sum(len(s) for _, s in docs)} sentences; {len(todo)} statements to score",
        flush=True,
    )
    if todo:
        rob = Rob()
        t0 = time.time()
        done = 0
        for d, s in todo:
            cache[d] = rob.scores(s, bs=bs)
            done += len(s)
            _save_cache(
                cache, cache_f
            )  # one checkpoint per statement: a killed run resumes from the next statement
            print(
                f"  {d}: {len(s)} sentences, h={np.mean(cache[d]):+.4f}  [{done} done, {time.time() - t0:.0f}s]",
                flush=True,
            )
    return cache


def embed_topics(sentences):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(C.MINILM_SNAPSHOT, device="cpu")
    E = st.encode(sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cent = {}
    for t, protos in C.PROTOTYPES.items():
        P = st.encode(
            protos, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        ).mean(0)
        cent[t] = P / np.linalg.norm(P)
    S = np.stack([E @ cent[t] for t in C.TOPICS], 1)  # cosine to each centroid
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    docs = C.load_statements(policy_only=False)
    if a.limit:
        docs = docs[: a.limit]
    cache = score_all(docs, a.bs)

    # ---- sentence table ----
    rows = []
    for d, sents in docs:
        for i, s in enumerate(sents):
            rows.append(
                {
                    "date": d,
                    "policy": d not in C.NON_POLICY,
                    "idx": i,
                    "sentence": s,
                    "score": cache[d][i],
                }
            )
    df = pd.DataFrame(rows)
    S = embed_topics(df["sentence"].tolist())
    for j, t in enumerate(C.TOPICS):
        df["sim_" + t] = S[:, j]
    df["topic_emb"] = [C.TOPICS[j] for j in S.argmax(1)]
    df["sim_max"] = S.max(1)
    df.loc[df["sim_max"] < C.SIM_FLOOR, "topic_emb"] = "other"
    df["override"] = [C.override_topic(s) or "" for s in df["sentence"]]
    df["topic"] = np.where(df["override"] != "", df["override"], df["topic_emb"])
    df.to_csv(os.path.join(DATA, "sentences.csv"), index=False, encoding="utf-8")
    print("topic counts (all sentences):")
    print(df["topic"].value_counts().to_string())
    print("override share: %.2f" % (df["override"] != "").mean())

    # ---- per-statement, per-topic aggregation (policy statements only; previous = previous policy statement) ----
    pol = df[df["policy"]].copy()
    dates = sorted(pol["date"].unique())
    agg = {}
    for d in dates:
        x = pol[pol["date"] == d]
        n = len(x)
        agg[d] = {
            "n_sent": n,
            "h": x["score"].mean(),
            "topics": {
                t: (
                    len(x[x.topic == t]),
                    x[x.topic == t]["score"].mean() if (x.topic == t).any() else np.nan,
                )
                for t in C.TOPICS
            },
        }
    trows, srows = [], []
    for k, d in enumerate(dates):
        prev = dates[k - 1] if k else None
        A = agg[d]
        P = agg[prev] if prev else None
        srows.append(
            {
                "date": d,
                "prev_date": prev,
                "n_sent": A["n_sent"],
                "n_sent_prev": P["n_sent"] if P else np.nan,
                "h": A["h"],
                "h_prev": P["h"] if P else np.nan,
                "dh": A["h"] - P["h"] if P else np.nan,
            }
        )
        for t in C.TOPICS:
            n_t, h_t = A["topics"][t]
            share = n_t / A["n_sent"]
            c = share * h_t if n_t else 0.0
            if P:
                n_p, h_p = P["topics"][t]
                share_p = n_p / P["n_sent"]
                c_p = share_p * h_p if n_p else 0.0
                dh = h_t - h_p if (n_t and n_p) else np.nan
                dc = c - c_p
                dshare = share - share_p
                status = (
                    "both"
                    if (n_t and n_p)
                    else ("appeared" if n_t else ("vanished" if n_p else "absent"))
                )
            else:
                h_p = share_p = c_p = dh = dc = dshare = np.nan
                status = "first"
            trows.append(
                {
                    "date": d,
                    "topic": t,
                    "n": n_t,
                    "share": share,
                    "h": h_t,
                    "c": c,
                    "n_prev": (n_p if P else np.nan),
                    "share_prev": share_p,
                    "h_prev": h_p,
                    "c_prev": c_p,
                    "dh": dh,
                    "dc": dc,
                    "dshare": dshare,
                    "status": status,
                }
            )
    T = pd.DataFrame(trows)
    Sd = pd.DataFrame(srows)
    # the decomposition identity: sum_k dc_k == dh (exact)
    chk = T.groupby("date")["dc"].sum().reindex(Sd["date"]).values - Sd["dh"].values
    assert np.nanmax(np.abs(chk)) < 1e-9, np.nanmax(np.abs(chk))

    # ---- cross-check with rates-curve-nlp's document scores and attach the USMPD windows ----
    own = pd.read_csv(C.RCN_FEATURES)
    own = own[own.bank == "FOMC"][["date", "rob_dh", "r2", "r5", "r10", "r30", "src"]]
    oc = {
        r["date"]: r["rob_h"]
        for r in json.load(open(C.RCN_CACHE, encoding="utf-8"))
        if r["bank"] == "FOMC"
    }
    Sd["rcn_rob_h"] = Sd["date"].map(oc)
    Sd = Sd.merge(own.rename(columns={"rob_dh": "rcn_rob_dh"}), on="date", how="left")
    us = pd.read_excel(C.USMPD, sheet_name="Statements")
    us["Date"] = pd.to_datetime(us["Date"]).dt.strftime("%Y-%m-%d")
    us = us.set_index("Date")
    for t, col in [
        ("r2", "UST2Y"),
        ("r5", "UST5Y"),
        ("r10", "UST10Y"),
        ("r30", "UST30Y"),
        ("mp1", "MP1"),
    ]:
        Sd["usmpd_" + t] = Sd["date"].map(us[col] * 100)  # percentage points -> bp
    Sd["train"] = Sd["date"] <= C.TRAIN_END
    T.to_csv(os.path.join(DATA, "statement_topics.csv"), index=False)
    Sd.to_csv(os.path.join(DATA, "statements.csv"), index=False)
    ok = Sd.dropna(subset=["rcn_rob_h"])
    err = np.abs(ok["h"] - ok["rcn_rob_h"]).max()
    print(f"cross-check vs rates-curve-nlp's rob_h on {len(ok)} statements: max |diff| = {err:.2e}")
    print(f"wrote {len(df)} sentences, {len(Sd)} policy statements, {len(T)} statement-topic rows")


if __name__ == "__main__":
    main()
