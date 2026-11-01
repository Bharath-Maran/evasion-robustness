"""Shared pieces of the evasion study: data loading, evasion transforms, the normalizer, and metrics.

Tasks (all binary, 1 = hateful):
  hatexplain : HateXplain majority label, hatespeech vs. normal, using the published split.
  dyna_exp   : DynaHate derogation/dehumanization/threatening vs. none (explicit hate).
  dyna_imp   : DynaHate animosity vs. none (implicit hate). Same benign texts as dyna_exp.
"""

import json
import math
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
TASKS = ("hatexplain", "dyna_exp", "dyna_imp")
BUDGETS = (0.1, 0.2, 0.3)
MAX_HATEFUL_EVAL = 400


def load_task(task):
    """Return {'train': (texts, labels), 'val': ..., 'test': ...}."""
    if task == "hatexplain":
        posts = json.load(open(os.path.join(DATA, "hatexplain.json")))
        splits = json.load(open(os.path.join(DATA, "hatexplain_splits.json")))
        out = {}
        for name in ("train", "val", "test"):
            texts, labels = [], []
            for pid in splits[name]:
                votes = Counter(a["label"] for a in posts[pid]["annotators"]).most_common()
                # Ties and the separate 'offensive' class are left out of the binary task.
                if len(votes) > 1 and votes[0][1] == votes[1][1]:
                    continue
                if votes[0][0] in ("hatespeech", "normal"):
                    texts.append(" ".join(posts[pid]["post_tokens"]))
                    labels.append(int(votes[0][0] == "hatespeech"))
            out[name] = (texts, np.array(labels))
        return out
    frame = pair_aware_split(pd.read_csv(os.path.join(DATA, "dynahate.csv")))
    hateful = {"dyna_exp": {"derogation", "dehumanization", "threatening"}, "dyna_imp": {"animosity"}}[task]
    out = {}
    for name, split in (("train", "train"), ("val", "dev"), ("test", "test")):
        # Round 1 hateful texts have no type label, so keeping its benign rows would leave them with no counterpart.
        part = frame[(frame["split"] == split) & (frame["round.base"] >= 2) & (frame["type"].isin(hateful | {"none"}))]
        out[name] = (part["text"].astype(str).tolist(), part["type"].isin(hateful).astype(int).to_numpy())
    return out


def pair_aware_split(frame):
    """Keep each DynaHate contrast pair in one split.

    A perturbation is a near copy of an original with the opposite label, and the published split separates
    about a third of the pairs, which makes models learn the reverse of the truth. Each perturbation is moved
    to the split of its original.
    """
    split_of = dict(zip(frame["acl.id"], frame["split"]))
    frame = frame.copy()
    moved = (frame["level"] == "perturbation") & frame["acl.id.matched"].notna()
    frame.loc[moved, "split"] = frame.loc[moved, "acl.id.matched"].map(split_of)
    return frame


def eval_set(split, seed=0):
    """All benign test texts plus up to MAX_HATEFUL_EVAL hateful ones; every comparison uses this same set."""
    texts, labels = split
    rng = np.random.RandomState(seed)
    hateful = np.where(labels == 1)[0]
    rng.shuffle(hateful)
    keep = np.concatenate([np.where(labels == 0)[0], hateful[:MAX_HATEFUL_EVAL]])
    return [texts[i] for i in keep], labels[keep]


def metrics(y, prob):
    """Hateful-class F1 (primary), macro F1, precision and recall at a 0.5 threshold."""
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "acc": float((pred == y).mean())}


# ---------------------------------------------------------------------------------------------
# Organic evasion. The attacker never queries the model: the words to disguise come from a list of
# hate-associated words learned from training text, standing in for what users learn is moderated.
# ---------------------------------------------------------------------------------------------

WORD = re.compile(r"^[a-zA-Z]{3,}$")
LEET_LIGHT = {"a": "4", "e": "3", "i": "1", "o": "0"}
LEET_HEAVY = {**LEET_LIGHT, "s": "$", "t": "7", "l": "1", "g": "9", "b": "8"}
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і", "s": "ѕ",
              "x": "х", "y": "у"}
# Documented algospeak for ordinary vocabulary. Identity terms and slurs are deliberately not listed here;
# those are only ever disguised by the character-level transforms.
ALGOSPEAK = {"kill": "unalive", "killed": "unalived", "killing": "unaliving", "kills": "unalives",
             "die": "unalive", "dead": "unalived", "death": "unaliving", "murder": "unalive",
             "suicide": "sewerslide", "gun": "pew pew", "guns": "pew pews", "shoot": "pew", "sex": "seggs",
             "rape": "grape", "raped": "graped", "porn": "corn", "nazi": "nahtzee", "nazis": "nahtzees",
             "racist": "raycist", "racism": "raycism", "lesbian": "le dollar bean", "lesbians": "le dollar beans",
             "hate": "h8", "white": "yt", "whites": "yts"}
REVERSE_ALGOSPEAK = {v: k for k, v in ALGOSPEAK.items() if " " not in v}


def leet(word, table, rng):
    """Replace about two thirds of the replaceable letters so the word stays readable."""
    return "".join(table[c] if c in table and rng.random() < 0.67 else c for c in word.lower())


def homoglyph(word, rng):
    """Swap Latin letters for look-alike Cyrillic ones."""
    return "".join(HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < 0.67 else c for c in word.lower())


def spacing(word, rng):
    """Insert one separator mid-word, e.g. wo.rd."""
    i = rng.randint(1, len(word) - 1)
    return word[:i] + rng.choice([" ", ".", "-", "*"]) + word[i:]


def repeat(word, rng):
    """Stretch one vowel, or the middle letter if there is none."""
    spots = [i for i, c in enumerate(word) if c in "aeiou"] or [len(word) // 2]
    i = rng.choice(spots)
    return word[:i] + word[i] * 3 + word[i:]


def algospeak(word, rng):
    """Community replacement word; words with no entry are left for the mix transform to handle."""
    return ALGOSPEAK.get(word.lower(), word)


STRATEGIES = {"leet_light": lambda w, r: leet(w, LEET_LIGHT, r), "leet_heavy": lambda w, r: leet(w, LEET_HEAVY, r),
              "homoglyph": homoglyph, "spacing": spacing, "repeat": repeat, "algospeak": algospeak}


def salient_words(train, min_count=5):
    """word -> log-odds of appearing in hateful training text, for words more common in hate."""
    texts, labels = train
    pos, neg = Counter(), Counter()
    for t, y in zip(texts, labels):
        (pos if y == 1 else neg).update(set(w.lower() for w in t.split()))
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    out = {}
    for w, c in pos.items():
        if c >= min_count and WORD.match(w):
            s = math.log((c + 1) / (n_pos + 2)) - math.log((neg[w] + 1) / (n_neg + 2))
            if s > 0:
                out[w] = s
    return out


def evade(text, strategy, salience, budget, seed):
    """Disguise the most hate-associated words of a text, changing at most budget * tokens of them."""
    rng = random.Random(f"{seed}|{text}")
    tokens = text.split()
    limit = max(1, int(budget * len(tokens)))
    scored = [(salience.get(t.lower(), 0.0), i) for i, t in enumerate(tokens) if WORD.match(t)]
    order = [i for s, i in sorted(scored, reverse=True)]
    changed = 0
    for i in order:
        if changed == limit:
            break
        word = tokens[i]
        # 'mix' picks a strategy per word, the closest thing here to how people really write.
        name = rng.choice(["leet_light", "leet_heavy", "homoglyph", "spacing", "repeat"]) if strategy == "mix" else strategy
        if strategy == "mix" and word.lower() in ALGOSPEAK:
            name = "algospeak"
        new = STRATEGIES[name](word, rng)
        if new != word:
            tokens[i] = new
            changed += 1
    return " ".join(tokens)


def evade_set(texts, labels, strategy, salience, budget, seed=0):
    """Apply a transform to hateful texts only, as in the proposal's definition of degradation."""
    return [evade(t, strategy, salience, budget, seed) if y == 1 else t for t, y in zip(texts, labels)]


# ---------------------------------------------------------------------------------------------
# Mitigation 1: input normalization.
# ---------------------------------------------------------------------------------------------

UNGLYPH = {v: k for k, v in HOMOGLYPHS.items()}
UNLEET = {"4": "a", "3": "e", "1": "i", "0": "o", "$": "s", "7": "t", "9": "g", "8": "b", "@": "a", "5": "s"}


def normalize_token(tok):
    """Undo one token's disguise. Plain numbers are not words, so they are left alone."""
    tok = "".join(UNGLYPH.get(c, c) for c in tok)
    core = tok.lower()
    if core in REVERSE_ALGOSPEAK:
        return REVERSE_ALGOSPEAK[core]
    if any(c.isalpha() for c in tok) and any(c in UNLEET for c in tok):
        tok = "".join(UNLEET.get(c, c) for c in tok.lower())
    # A separator inside letters: wo.rd, wo-rd, wo*rd. Only '.' and '*' are undone, as real hyphens are common.
    tok = re.sub(r"(?<=[a-zA-Z])[.*](?=[a-zA-Z])", "", tok)
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1", tok)


def normalize(text):
    """Normalize every token of a text."""
    return " ".join(normalize_token(t) for t in text.split())
