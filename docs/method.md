# Method

## Data and definitions

- **FOMC statements.** 99 files from 28 Jan 2015 to 29 Jul 2026, as downloaded by rates-curve-nlp (`tools/download.py`). Four are not monetary-policy statements and are left out: the 11 Oct 2019 reserve-management notation vote, the 31 Mar 2020 FIMA repo facility, and the framework statements of 27 Aug 2020 and 22 Aug 2025. That leaves 95 policy statements, 94 with a predecessor and a USMPD window. "Previous statement" means the previous policy statement. rates-curve-nlp uses the previous file instead, so its Δh differs for the four statements that follow a non-policy file (30 Oct 2019, 29 Apr 2020, 16 Sep 2020, 17 Sep 2025); h agrees on all 94 statements and Δh on the other 90 to within 1e-6.
- **Score.** The sentence split, the document reader and the per-sentence score P(hawk) − P(dove) are rates-curve-nlp's (`tools/text_features.py`), unchanged. h is the mean over a statement's sentences; Δh = h(this) − h(previous).
- **Models.** `tim9510019/FOMC-RoBERTa`, a re-upload of the Shah, Paturi and Chava (2023) model, at commit `d6acba5e`, trained on text up to October 2022; `sentence-transformers/all-MiniLM-L6-v2` at commit `1110a243` for the topic embeddings.
- **Target.** SF Fed USMPD, sheet *Statements*: the 2-year Treasury yield change from 10 minutes before to 20 minutes after the release, in basis points. rates-curve-nlp's pooled test also includes ECB statements, for which its target is the daily change in the US 2-year yield; that leg is quoted for comparison but not decomposed here.
- **Split.** rates-curve-nlp's: fitted on statements to 31 Dec 2023, tested on 2024-26.

## Topics

Ten labels fitted to statement language: inflation; labour market; activity and growth; financial conditions and credit; balance sheet; policy path and forward guidance; external shocks (trade, tariffs, energy, geopolitics, pandemic); mandate and risks boilerplate (the goals, assessment and balance-of-risks sentences that recur in every statement); vote roll-call; other.

Each sentence is embedded with MiniLM and assigned to the nearest topic centroid by cosine. A centroid is the mean embedding of three to seven prototype sentences written in statement register; below a cosine of 0.20 the sentence is "other". A short, ordered keyword override takes precedence for votes (including the name fragments the sentence split produces at initials such as "Loretta J."), balance-sheet instruments, named external shocks, verbatim boilerplate and the rate-decision sentence. Over the corpus the override assigns 64 % of sentences.

Per statement and topic: n_k, share_k = n_k / n, h_k = the mean score of the topic's sentences, and c_k = share_k · h_k. Against the previous policy statement: Δh_k, undefined when the topic is absent from either statement (reported, not imputed), and Δc_k and Δshare_k, always defined. Δh = Σ_k Δc_k holds exactly (asserted), so the topic contributions decompose the aggregate.

**Hand check.** 40 sentences drawn before any model output, 10 per era, labelled by hand: 38 agree with the final assignment, 32 with the embedding alone. The keyword override fired on 26 of the 40 and agreed on all 26. Both misses are "The Committee … monitor(s) inflation developments" sentences (hand: inflation; model: mandate).

## Tests

- **(a) Aggregate.** corr(Δh, 2y) on the training years, the test years and all statements, with bootstrap confidence intervals and permutation p-values, beside rates-curve-nlp's published figures.
- **(b) Incremental information.** Model A: 2y ~ Δh. Model B: 2y ~ Δh + Σ_k g_k Δc_k, with a ridge penalty on g only, so an infinite penalty recovers A exactly. The penalty is chosen by leave-one-out mean squared error on the 73 training statements; B is then evaluated strictly out of sample on the 21 policy statements of 2024-26 (rates-curve-nlp's FOMC test set has 22, because it includes the 2025 framework statement). The p-value for the out-of-sample improvement comes from 2,000 row-shuffles of the topic block within the training and test sets. Variants: topic-level Δh_k as the block (missing values set to 0, plus an indicator), Δshare_k (prevalence), and a leave-one-out version over all 94 statements.
- **(c) Stability.** The aggregate recomputed without each topic and from each topic alone; a jackknife of the test correlation.
- **(d) What broke.** Undefined topic changes, sentences per topic, the mix-versus-tone split of Δh, the vote fragments, the 2026 change in statement format, how often the Biosca and Wales narratives appear in statements at all, and the model's training cut-off.

## Further results

All from `results/topic_tests.md`.

**Aggregate, beside rates-curve-nlp.** Its pooled FOMC and ECB test gives corr 0.34 [0.09, 0.54] (n = 44, permutation p = 0.022); FOMC alone 0.52 [0.17, 0.78] (n = 22); ECB alone, on daily changes, 0.16 [−0.27, 0.49] (n = 22). Model A's leave-one-out R² on the training years is −0.03.

**Per topic.** The correlation of each topic's contribution Δc_k with the 2y move: over all 94 statements only the mandate boilerplate has an interval clear of zero, 0.23 [0.06, 0.39], partial 0.13 given Δh; every other topic is at most 0.15 in absolute value with an interval covering zero. On the 21 test statements the raw correlations are larger (mandate 0.52, policy path 0.45, external 0.41, activity 0.32) and the partials run from −0.40 (other) to +0.27 (activity): n = 21 with a median of one sentence per topic.

**Variants of (b).** Topic tone changes Δh_k as the block: infinite penalty again, ΔR² 0.000. Prevalence changes Δshare_k: a finite penalty on the training years, out-of-sample corr 0.256, ΔR² −0.203. Leave-one-out over all 94 statements: corr 0.211 with and without the topic block.

**Stability.** Without each topic in turn, the test correlation runs from 0.50 (without the vote) to 0.63 (without the balance sheet), and the all-statement correlation from 0.22 (without the mandate boilerplate) to 0.30 (without the vote). From one topic alone: mandate 0.54 on the test years and 0.22 overall; policy path 0.51 (n = 19) and 0.12; the vote 0.40 (n = 19) and 0.08; the seven substantive topics together 0.38 and 0.23. Jackknife of the test correlation: 0.45 (without 17 Jun 2026) to 0.69 (without 18 Dec 2024).

**What broke, in detail.** Δh_k is undefined in 206 of 940 statement-topic pairs; financial conditions is absent from both statements in 66 of its 94 pairs. Δh is 83 % tone (share-weighted Δh_k) and 20 % mix (topics entering, leaving or changing share) by variance. Vote fragments ("Voting for the FOMC monetary policy action were: Janet L.") are 22 % of all scored sentences, with mean score −0.00 and standard deviation 0.22; without them the test correlation is 0.50. The 17 Jun 2026 statement was cut from 21 to 9 sentences (Δh +0.28, the largest in the test set; 2y +7.1 bp); without it the test correlation is 0.45. The Biosca and Wales narratives in the 99 statement files: tariffs 0 sentences, AI 0, Middle East or Hormuz 9 (in 7 statements); related words: energy 71, pandemic 51, Ukraine or war 12, net exports 12. After FOMC-RoBERTa's training cut-off, the 19 external-shock sentences include "Although swings in net exports have affected the data, recent indicators suggest that economic activity has continued to expand", scored +1.00 (hawkish) in May and June 2025.

## Run sizes

2,065 sentences scored once (cached in `data/cache/rob_scores.json`, resumable per statement; the full run took about 25 minutes on CPU); 2,000 permutations and 2,000 bootstrap draws in the analysis; a 40-sentence hand check.
