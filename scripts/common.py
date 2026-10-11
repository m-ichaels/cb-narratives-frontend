"""Shared definitions for the per-topic split of FOMC statements.

Everything that must agree with rates-curve-nlp (github.com/m-ichaels/rates-curve-nlp) is copied verbatim
from its tools/text_features.py: the sentence split, the document reader and the per-sentence score
P(hawk) - P(dove) under FOMC-RoBERTa (tim9510019/FOMC-RoBERTa, commit d6acba5e...).
Inputs are read from a rates-curve-nlp checkout (RCN_DIR) and the Hugging Face cache (HF_HOME).
"""

import os, re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The two models are read from the local Hugging Face cache at the pinned commits below.
HF_HOME = os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# A rates-curve-nlp checkout with its raw data downloaded (python tools/download.py data/raw, run there).
RCN = os.environ.get("RCN_DIR", os.path.join(os.path.dirname(ROOT), "rates-curve-nlp"))
FOMC_DIR = os.path.join(RCN, "data", "raw", "fomc")
USMPD = os.path.join(RCN, "data", "raw", "USMPD.xlsx")
RCN_FEATURES = os.path.join(RCN, "data", "derived", "nlp_features.csv")
RCN_CACHE = os.path.join(RCN, "data", "derived", "nlp_cache.json")
RCN_SIGNAL = os.path.join(RCN, "results", "nlp_signal.json")

ROB_COMMIT = "d6acba5e8c62eb35f79527a099af789cd778387d"
ROB_SNAPSHOT = os.path.join(
    HF_HOME, "hub", "models--tim9510019--FOMC-RoBERTa", "snapshots", ROB_COMMIT
)
MINILM_COMMIT = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_SNAPSHOT = os.path.join(
    HF_HOME, "hub", "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots", MINILM_COMMIT
)

TRAIN_END = (
    "2023-12-31"  # rates-curve-nlp's split: models fitted on statements to 2023-12-31, tested after
)
FIRST, LAST = (
    "2015-01-28",
    "2026-07-29",
)  # the statements studied here; later downloads add meetings

# The four FOMC releases among the statement files that are not monetary-policy statements (no decision on the
# stance of policy). Count: 99 files, 95 policy statements. Rule: a policy statement is a Committee decision on the
# federal funds rate or on asset purchases; these four are a reserve-management notation vote, a repo facility, and
# two framework ("Statement on Longer-Run Goals") announcements. Asserted in load_statements().
NON_POLICY = {
    "2019-10-11": "reserve management purchases / notation vote (balance sheet implementation)",
    "2020-03-31": "FIMA repo facility announcement",
    "2020-08-27": "Statement on Longer-Run Goals and Monetary Policy Strategy (framework)",
    "2025-08-22": "Statement on Longer-Run Goals and Monetary Policy Strategy (framework update)",
}


# ---- copied verbatim from rates-curve-nlp/tools/text_features.py --------------------------------------------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [x.strip() for x in s if 25 <= len(x.strip()) <= 600]


def fomc_text(path):
    return open(path, encoding="utf-8", errors="ignore").read()


# -------------------------------------------------------------------------------------------------------------------


def load_statements(policy_only=True):
    """[(date 'YYYY-MM-DD', [sentences])] for the FOMC files, in date order."""
    out = []
    for f in sorted(glob.glob(os.path.join(FOMC_DIR, "*.txt"))):
        d = os.path.basename(f)[:8]
        date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        if not FIRST <= date <= LAST:
            continue
        text = fomc_text(f)
        if policy_only and date in NON_POLICY:
            continue
        if policy_only:
            assert re.search(r"federal funds rate|asset purchases|purchase", text, re.I), (
                f"{date}: not a policy statement?"
            )
        out.append((date, split_sentences(text)))
    if policy_only:
        assert len(out) == 95, len(out)
    return out


# ---- topic taxonomy ---------------------------------------------------------------------------------------------
# Ten labels fitted to FOMC-statement language. The seven substantive ones cover the usual statement themes (inflation, labour,
# activity, financial conditions, balance sheet, policy path, external shocks); "mandate" holds the goals /
# balance-of-risks boilerplate that recurs in every statement, "voting" the vote sentence, "other" the rest.
TOPICS = [
    "inflation",
    "labour",
    "activity",
    "financial",
    "balance_sheet",
    "policy_path",
    "external",
    "mandate",
    "voting",
    "other",
]

# Prototype sentences per topic, written in statement register (not copied from any one statement). Each topic's
# centroid is the mean of its normalised MiniLM embeddings; a sentence is assigned to the nearest centroid by cosine.
PROTOTYPES = {
    "inflation": [
        "Inflation has declined but remains elevated.",
        "Inflation is running below the Committee's 2 percent longer-run objective.",
        "Market-based measures of inflation compensation remain low; survey-based measures of longer-term inflation expectations are little changed.",
        "Inflation on a 12-month basis is expected to remain near its recent low level in the near term.",
        "Consumer prices have risen more slowly and core inflation has moderated.",
        "Inflation remains somewhat elevated.",
    ],
    "labour": [
        "Job gains have been solid, and the unemployment rate has remained low.",
        "Labor market conditions have improved further, with strong job gains and a lower unemployment rate.",
        "The unemployment rate has moved up but remains low.",
        "Job gains have slowed in recent months, and the unemployment rate has edged up.",
        "A range of labor market indicators suggests that underutilization of labor resources continues to diminish.",
        "Employment has been growing in recent months.",
    ],
    "activity": [
        "Economic activity has been expanding at a solid pace.",
        "Household spending has been rising moderately and business fixed investment has remained soft.",
        "Growth of economic activity has slowed from its strong pace in the third quarter.",
        "Recent indicators point to modest growth in spending and production.",
        "The housing sector has shown some improvement.",
        "Economic activity is expanding at a moderate pace.",
    ],
    "financial": [
        "Financial conditions have tightened significantly in recent months.",
        "The U.S. banking system is sound and resilient; tighter credit conditions for households and businesses are likely to weigh on economic activity.",
        "Strains in financial markets have eased and credit is flowing to households and businesses.",
        "The Committee is closely monitoring financial developments.",
        "Equity prices have fallen and mortgage rates have risen.",
        "Tighter financial conditions could weigh on hiring and investment.",
    ],
    "balance_sheet": [
        "The Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities.",
        "The Committee is maintaining its existing policy of reinvesting principal payments from its holdings of agency debt and agency mortgage-backed securities.",
        "The Federal Reserve will continue to purchase Treasury securities and agency mortgage-backed securities in the amounts needed.",
        "The Committee will slow the pace of decline of its securities holdings by reducing the monthly redemption cap on Treasury securities.",
        "The Committee reaffirmed its policy of maintaining ample reserves in the banking system.",
        "The Committee will reduce the monthly pace of its net asset purchases.",
    ],
    "policy_path": [
        "The Committee decided to maintain the target range for the federal funds rate.",
        "The Committee decided to raise the target range for the federal funds rate by 1/4 percentage point.",
        "In considering the extent and timing of additional adjustments to the target range, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks.",
        "The Committee expects that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive.",
        "The Committee anticipates that it will be appropriate to maintain the current target range until inflation has moved down sustainably.",
        "The stance of monetary policy remains accommodative, thereby supporting further improvement in labor market conditions.",
        "The Committee expects to maintain this target range until it is confident that the economy has weathered recent events.",
    ],
    "external": [
        "The Committee is closely monitoring global economic and financial developments.",
        "The invasion of Ukraine by Russia is causing tremendous human and economic hardship and is creating additional upward pressure on inflation.",
        "The coronavirus outbreak is causing tremendous human and economic hardship across the United States and around the world.",
        "Swings in net exports have affected the data.",
        "Developments in the Middle East are contributing to a high level of uncertainty about the economic outlook.",
        "Inflation is elevated, in part reflecting the recent increase in global energy prices.",
        "Higher tariffs are likely to raise prices and weigh on economic activity.",
    ],
    "mandate": [
        "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.",
        "The Committee judges that the risks to achieving its employment and inflation goals are roughly balanced.",
        "Uncertainty about the economic outlook has increased.",
        "The Committee is attentive to the risks to both sides of its dual mandate.",
        "The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, indicators of inflation pressures and inflation expectations, and readings on financial and international developments.",
        "In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook.",
        "The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective.",
    ],
    "voting": [
        "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; and Michelle W. Bowman.",
        "Voting against the action was James Bullard, who preferred at this meeting to raise the target range for the federal funds rate by 3/4 percentage point.",
        "Voting against this action was Esther L. George, who preferred to lower the target range by 1/4 percentage point.",
    ],
    "other": [
        "For media inquiries, please email or call the press office.",
        "Implementation Note issued on this date.",
        "The Federal Open Market Committee approved the following statement for release.",
        "Information received since the Federal Open Market Committee met in December suggests the following.",
    ],
}

# Keyword override: short, high-precision, ordered (first match wins). Everything not matched here is assigned by the
# embedding. The patterns name instruments, shocks, votes and verbatim boilerplate - not generic words such as
# "inflation" or "growth", which the embedding handles. Verbatim boilerplate (mandate) is checked before the generic
# policy phrases because "stance of monetary policy" occurs inside the boilerplate too.
OVERRIDES = [
    (
        "voting",
        r"^voting (for|against)\b|\bvoting against\b|\bvoting for the\b|preferred (to|at this meeting|no change)|\bdissent",
    ),
    (
        "balance_sheet",
        r"holdings of (treasury|agency|longer-term|securities)|purchases and holdings|agency mortgage-backed|agency debt|mortgage-backed securities|reinvest|balance sheet|"
        r"reserve balances|ample reserves|asset purchases|securities holdings|principal payments|redemption cap|"
        r"system open market account|reserve management|purchase(s|d)? (of )?(treasury|agency)|net asset purchases|"
        r"treasury securities|treasury bills",
    ),
    # verbatim goals / assessment boilerplate that itself names shocks ("readings on public health", "international
    # developments") is caught before the shock patterns
    (
        "mandate",
        r"seeks to (achieve|foster) maximum employment|maximum employment and inflation at the rate of 2 percent|"
        r"dual mandate|statutory mandate|longer-run goals|wide range of information|monitor the implications of incoming information",
    ),
    # "energy prices" as a shock, not the core-inflation phrases "excluding food and energy prices" / "excluding energy
    # and food"; "net exports" only in the 2025 tariff-front-running phrasing "swings in net exports" (the 2015-16
    # "net exports have been soft" is a demand component and goes to the embedding)
    (
        "external",
        r"tariff|swings in net exports|\bglobal\b|geopolit|middle east|(?<!food and )energy price|oil price|"
        r"coronavirus|pandemic|public health|\bvirus\b|vaccin|ukraine|russia|\bwar\b|conflict|supply shock|"
        r"supply chain|commodity price|including energy",
    ),
    (
        "mandate",
        r"would be prepared to adjust|risks to (achieving|both sides)|roughly balanced|attentive to (the )?(inflation )?risks|"
        r"uncertainty about the economic outlook|strongly committed to|committed to using its full range",
    ),
    (
        "policy_path",
        r"decided to (maintain|raise|lower|keep|reduce|increase)|target range for the federal funds rate (at|by|to)|"
        r"stance of monetary policy|extent and timing|additional firming|pace of future|future adjustments|"
        r"ongoing increases in the target range|full range of tools|policy accommodation|accommodative stance",
    ),
]
OVERRIDE_RE = [(t, re.compile(p, re.I)) for t, p in OVERRIDES]
SIM_FLOOR = 0.20  # below this cosine to every centroid the sentence is "other"
_NAME_TOKEN = re.compile(r"^[A-Z][A-Za-z.'\-]*[,;]?$")


def is_vote_fragment(sentence):
    """The rates-curve-nlp sentence split breaks the vote roll-call at initials ("Loretta J. Mester" -> "...; Loretta J." /
    "Mester; ..."), leaving fragments of semicolon-separated names. Rule: contains ';', under 200 characters, and at
    least 60 % of its tokens are capitalised name tokens."""
    s = sentence.strip()
    if ";" not in s or len(s) > 200:
        return False
    toks = [t for t in s.replace(";", " ;").split() if t != ";" and t != "and"]
    if not toks:
        return False
    return sum(bool(_NAME_TOKEN.match(t)) for t in toks) / len(toks) >= 0.6


def override_topic(sentence):
    if is_vote_fragment(sentence):
        return "voting"
    s = sentence.strip().lower()
    for t, rx in OVERRIDE_RE:
        if rx.search(s):
            return t
    return None
