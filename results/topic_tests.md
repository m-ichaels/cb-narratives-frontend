# Per-topic split of FOMC statements: results

Statements: 94 policy statements with a predecessor and a USMPD window; train n = 73 (to 2023-12-31), test n = 21.

## (a) Aggregate dh vs the 2y statement-window move

| sample | corr | 95 % CI | n | perm p |
|---|---|---|---|---|
| rates-curve-nlp, pooled FOMC + ECB test (nlp_signal.json) | 0.3437 | [0.0857, 0.5444] | 44 | 0.0216 |
| rates-curve-nlp, FOMC test only (nlp_features.csv) | 0.516 | [0.17, 0.78] | 22 | |
| rates-curve-nlp, ECB test only, daily changes | 0.161 | [-0.27, 0.49] | 22 | |
| this repo, policy statements, test | 0.573 | [0.21, 0.84] | 21 | 0.007 |
| this repo, policy statements, train | 0.194 | [-0.09, 0.44] | 73 | 0.097 |
| this repo, policy statements, all | 0.285 | [0.06, 0.48] | 94 | 0.006 |

Cross-check: max |h - rates-curve-nlp's rob_h| = 8.9e-07; max |dh - rates-curve-nlp's rob_dh| = 1.0e-06 excluding the four statements that follow a non-policy file (2019-10-30, 2020-04-29, 2020-09-16, 2025-09-17), whose 'previous statement' differs by construction. Other tenors (test): r5 0.553, r10 0.530, r30 0.352.

## Per topic

| topic | median n/stmt | mean share | undefined dh_k (of 94) | corr(dc_k, 2y) test [CI] | partial | corr(dc_k, 2y) all [CI] | partial | corr(dh_k, 2y) all (n) |
|---|---|---|---|---|---|---|---|---|
| inflation | 1 | 0.06 | 20 | -0.01 [-0.29, 0.31] | -0.29 | -0.01 [-0.19, 0.16] | -0.04 | 0.10 (74) |
| labour | 1 | 0.06 | 14 | 0.08 [-0.29, 0.37] | -0.14 | 0.15 [-0.05, 0.34] | 0.03 | 0.01 (80) |
| activity | 1 | 0.05 | 22 | 0.32 [0.10, 0.64] | 0.27 | 0.06 [-0.16, 0.25] | -0.00 | 0.00 (72) |
| financial | 0 | 0.01 | 72 | 0.11 [0.01, 0.34] | 0.06 | 0.05 [-0.21, 0.31] | -0.03 | 0.04 (22) |
| balance_sheet | 1 | 0.06 | 24 | -0.09 [-0.53, 0.28] | -0.33 | 0.00 [-0.16, 0.09] | -0.04 | -0.17 (70) |
| policy_path | 3 | 0.14 | 2 | 0.45 [0.01, 0.76] | 0.20 | 0.12 [-0.13, 0.34] | -0.03 | 0.12 (92) |
| external | 1 | 0.08 | 40 | 0.41 [-0.11, 0.74] | 0.09 | 0.12 [-0.08, 0.31] | -0.00 | -0.01 (54) |
| mandate | 5 | 0.24 | 2 | 0.52 [0.12, 0.83] | 0.11 | 0.23 [0.06, 0.39] | 0.13 | 0.22 (92) |
| voting | 5 | 0.22 | 2 | 0.24 [-0.28, 0.73] | -0.01 | 0.12 [-0.11, 0.33] | -0.01 | 0.08 (92) |
| other | 1 | 0.07 | 8 | -0.44 [-0.73, 0.08] | -0.40 | -0.04 [-0.25, 0.01] | 0.00 | -0.04 (86) |

## (b) Incremental information: A = dh only; B = dh + per-topic contributions dc_k (ridge on the topic block)

| model | lambda | LOO R2 train | LOO corr train | OOS corr | OOS R2 |
|---|---|---|---|---|---|
| A | inf | -0.027 | 0.017 | 0.573 | 0.260 |
| B | inf | -0.027 | 0.017 | 0.573 | 0.260 |

OOS delta R2 (B - A) = +0.000, delta corr = +0.000; permutation p (topic block row-shuffled, 2000 draws) = 0.715 for delta R2, 0.681 for delta corr; null 95th percentile of delta R2 = +0.024.
Topic weights in B (per standardised dc_k, bp): inflation +0.00, labour +0.00, activity +0.00, financial +0.00, balance_sheet +0.00, policy_path +0.00, external +0.00, mandate +0.00, voting +0.00, other +0.00.

Variants: topic block = tone changes dh_k (NaN -> 0 + indicators): OOS corr A 0.573 -> B 0.573, delta R2 +0.000; topic block = prevalence changes dshare_k: B 0.256, delta R2 -0.203. Leave-one-out over all 94 statements (no split): corr A 0.211 -> B 0.211, R2 0.040 -> 0.040.

## (c) Stability

| topic | drop it: corr test | drop it: corr all | only it: corr test (n) | only it: corr all (n) |
|---|---|---|---|---|
| inflation | 0.599 | 0.281 | 0.12 (18) | 0.10 (74) |
| labour | 0.591 | 0.253 | 0.06 (21) | 0.01 (80) |
| activity | 0.518 | 0.278 | 0.02 (17) | 0.00 (72) |
| financial | 0.567 | 0.279 | nan (0) | 0.04 (22) |
| balance_sheet | 0.628 | 0.272 | -0.27 (17) | -0.17 (70) |
| policy_path | 0.502 | 0.249 | 0.51 (19) | 0.12 (92) |
| external | 0.521 | 0.250 | 0.11 (6) | -0.01 (54) |
| mandate | 0.528 | 0.222 | 0.54 (21) | 0.22 (92) |
| voting | 0.501 | 0.303 | 0.40 (19) | 0.08 (92) |
| other | 0.576 | 0.293 | -0.32 (21) | -0.04 (86) |

Substantive topics only (no voting / mandate / other): corr test 0.383, all 0.230.
Jackknife of the test correlation (21 statements): min 0.454 (dropping 2026-06-17), max 0.692 (dropping 2024-12-18).

## (d) What broke

- Topic-level dh_k is undefined in 206 of 940 (statement, topic) pairs; per topic: inflation: both 74, absent 13, vanished 4, appeared 3; labour: both 80, absent 8, vanished 3, appeared 3; activity: both 72, absent 12, vanished 5, appeared 5; financial: absent 66, both 22, appeared 3, vanished 3; balance_sheet: both 70, absent 18, vanished 3, appeared 3; policy_path: both 92, vanished 1, absent 1; external: both 54, absent 26, vanished 7, appeared 7; mandate: both 92, vanished 1, appeared 1; voting: both 92, vanished 1, appeared 1; other: both 86, absent 7, appeared 1.
- Median sentences per topic per statement: inflation 1, labour 1, activity 1, financial 0, balance_sheet 1, policy_path 3, external 1, mandate 5, voting 5, other 1.
- Mix vs tone: dh = tone (share-weighted dh_k) + mix (topic entering / leaving / changing share). Variance share of dh: mix 0.20, tone 0.83 (corr -0.04). corr with the 2y move, test: mix 0.39, tone 0.55; all: mix 0.23, tone 0.20. corr(dh, change in sentence count) = -0.03.
- Vote roll-call fragments are 22% of all scored sentences (mean score -0.00, sd 0.22); e.g. "Voting for the FOMC monetary policy action were: Janet L.". Test corr without them: 0.501.
- 2026 format break: 2026-04-29: 21 sentences, h +0.075, dh +0.048, 2y +2.1 bp; 2026-06-17: 9 sentences, h +0.351, dh +0.276, 2y +7.1 bp; 2026-07-29: 12 sentences, h +0.352, dh +0.001, 2y -4.5 bp. Test corr without 2026-06-17: 0.454.
- The Fulcrum narratives in the statements (sentences / statements over all 99 files): tariff 0/0, AI 0/0, Middle East / Hormuz 9/7, energy 71/51, net exports 12/12, Ukraine / war 12/8, pandemic 51/24.
- External-shock sentences after FOMC-RoBERTa's training cut-off (Oct 2022): 19 sentences in 12 statements; scores: 2022-11-02 +1.00 "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemi..."; 2022-11-02 -0.75 "Russia's war against Ukraine is causing tremendous human and economic hardship...."; 2022-11-02 +1.00 "The war and related events are creating additional upward pressure on inflation and are we..."; 2022-12-14 +1.00 "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemi..."; 2022-12-14 -0.75 "Russia's war against Ukraine is causing tremendous human and economic hardship...."; 2022-12-14 +1.00 "The war and related events are contributing to upward pressure on inflation and are weighi..."; 2023-02-01 -0.96 "Russia's war against Ukraine is causing tremendous human and economic hardship and is cont..."; 2023-03-22 -0.00 "The extent of these effects is uncertain...."; 2025-03-19 -0.33 "Uncertainty around the economic outlook has increased...."; 2025-05-07 +1.00 "Although swings in net exports have affected the data, recent indicators suggest that econ..."; 2025-06-18 +1.00 "Although swings in net exports have affected the data, recent indicators suggest that econ..."; 2025-07-30 -0.17 "Although swings in net exports continue to affect the data, recent indicators suggest that..."; 2026-03-18 -0.00 "The implications of developments in the Middle East for the U.S. economy are uncertain...."; 2026-04-29 +1.00 "Inflation is elevated, in part reflecting the recent increase in global energy prices...."; 2026-04-29 -0.00 "Developments in the Middle East are contributing to a high level of uncertainty about the ..."; 2026-06-17 +0.96 "Economic activity is expanding at a solid pace despite elevated uncertainty that owes, in ..."; 2026-06-17 +1.00 "Inflation remains elevated relative to the Committee's 2 percent goal, in part reflecting ..."; 2026-07-29 +0.96 "Economic activity is expanding at a solid pace despite elevated uncertainty that owes, in ..."; 2026-07-29 +1.00 "Inflation remains elevated relative to the Committee's 2 percent goal, in part reflecting ...".
