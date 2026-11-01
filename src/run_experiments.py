"""Train the classifiers and measure them on clean and evaded text.

Usage: python src/run_experiments.py --model roberta-base --task dyna_imp --seed 0 [--augment] [--attack]
       python src/run_experiments.py --baseline
Each run appends one JSON file to results/. Only aggregate numbers are saved, never evaded texts.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import pipeline as P

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN, BATCH, EPOCHS, LR = 128, 16, 2, 2e-5
STRATEGIES = ("leet_light", "leet_heavy", "homoglyph", "spacing", "repeat", "algospeak", "mix")
NEIGHBORS, MIN_SIM, VOCAB = 15, 0.6, 200000


class Classifier:
    """A fine-tuned transformer with a predict() that returns P(hateful)."""

    def __init__(self, name, seed):
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.net = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEVICE)

    def encode(self, texts):
        return {k: v.to(DEVICE) for k, v in self.tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True,
                                                     return_tensors="pt").items()}

    @torch.no_grad()
    def predict(self, texts, batch=64):
        self.net.eval()
        out = []
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), batch):
            logits = self.net(**self.encode([texts[j] for j in order[i:i + batch]])).logits
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        probs = np.zeros(len(texts))
        if len(texts):
            probs[order] = np.concatenate(out)
        return probs

    def fit(self, train, val, seed):
        """Train for EPOCHS and keep the weights of the epoch with the best validation F1."""
        texts, labels = train
        opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * (len(texts) // BATCH + 1)
        sched = get_linear_schedule_with_warmup(opt, steps // 10, steps)
        rng = np.random.RandomState(seed)
        best, best_state = -1.0, None
        for epoch in range(EPOCHS):
            self.net.train()
            order = rng.permutation(len(texts))
            for i in range(0, len(texts), BATCH):
                idx = order[i:i + BATCH]
                batch = self.encode([texts[j] for j in idx])
                loss = self.net(**batch, labels=torch.tensor(labels[idx], device=DEVICE)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            score = P.metrics(val[1], self.predict(val[0]))["f1"]
            print(f"  epoch {epoch}: val F1 {score:.4f}", flush=True)
            if score > best:
                best, best_state = score, {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        return best


class Baseline:
    """TF-IDF word n-grams + logistic regression, as a classical reference."""

    def __init__(self, seed):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(C=10, max_iter=2000, random_state=seed)

    def fit(self, train, val, seed):
        self.clf.fit(self.vec.fit_transform(train[0]), train[1])
        return P.metrics(val[1], self.predict(val[0]))["f1"]

    def predict(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


def augmented(train, salience, seed):
    """Mitigation 2: add one evaded copy of every hateful training text."""
    texts, labels = train
    rng = random.Random(seed)
    names = [n for n in STRATEGIES if n != "algospeak"]
    extra = [P.evade(t, rng.choice(names), salience, rng.choice(P.BUDGETS), seed) for t, y in zip(texts, labels) if y == 1]
    return list(texts) + extra, np.concatenate([labels, np.ones(len(extra), dtype=int)])


def load_vectors():
    """The VOCAB most frequent word2vec words, unit length. Scanning all 3M per lookup took over a second each."""
    kv = KeyedVectors.load(os.path.expanduser("~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.kv"), mmap="r")
    vectors = np.array(kv.vectors[:VOCAB])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    words = kv.index_to_key[:VOCAB]
    return vectors, words, {w: i for i, w in enumerate(words)}


def substitutes(word, kv, cache):
    """Nearest word2vec neighbours that are plain lowercase words, the ML attack's search space."""
    if word not in cache:
        cands = []
        vectors, words, index = kv
        if word in index:
            sims = vectors @ vectors[index[word]]
            for j in np.argsort(-sims)[1:61]:
                w = words[j].lower()
                if sims[j] >= MIN_SIM and w.isalpha() and w != word and w not in cands:
                    cands.append(w)
                if len(cands) == NEIGHBORS:
                    break
        cache[word] = cands
    return cache[word]


def attack(model, text, kv, cache, budget):
    """Greedy importance-ranked synonym attack (TextFooler family), stopping once the prediction flips.

    Returns (substitutions used, or None if it failed, queries). The attack queries the model, unlike
    organic evasion, and is restricted to the same budget.
    """
    tokens = text.split()
    limit = max(1, int(budget * len(tokens)))
    base = model.predict([text])[0]
    queries = 1
    if base < 0.5:
        return 0, queries
    variants = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    drops = base - model.predict(variants)
    queries += len(variants)
    used, current = 0, base
    for i in np.argsort(-drops):
        if used == limit:
            break
        cands = substitutes(tokens[i].lower(), kv, cache)
        if not cands:
            continue
        options = [" ".join(tokens[:i] + [c] + tokens[i + 1:]) for c in cands]
        probs = model.predict(options)
        queries += len(options)
        best = int(np.argmin(probs))
        if probs[best] < current:
            tokens[i] = cands[best]
            current = probs[best]
            used += 1
        if current < 0.5:
            return used, queries
    return None, queries


def evaluate(model, data, salience):
    """All measurements for one trained model on the shared evaluation set."""
    texts, y = P.eval_set(data["test"])
    res = {"clean_full_test": P.metrics(data["test"][1], model.predict(data["test"][0])), "clean": P.metrics(y, model.predict(texts))}
    res["n_eval"], res["n_hateful"] = len(y), int(y.sum())
    res["organic"] = {}
    norm_clean = [P.normalize(t) for t in texts]
    res["normalized_clean"] = P.metrics(y, model.predict(norm_clean))
    res["benign_changed_by_normalizer"] = float(np.mean([a != b for a, b, l in zip(texts, norm_clean, y) if l == 0]))
    for name in STRATEGIES:
        for budget in P.BUDGETS:
            if name != "mix" and budget != 0.2:
                continue
            evaded = P.evade_set(texts, y, name, salience, budget)
            row = P.metrics(y, model.predict(evaded))
            row["normalized"] = P.metrics(y, model.predict([P.normalize(t) for t in evaded]))
            row["changed"] = float(np.mean([a != b for a, b, l in zip(texts, evaded, y) if l == 1]))
            res["organic"][f"{name}@{budget}"] = row
    return res


def run_attack(model, data):
    """Attack every hateful evaluation text once at the largest budget; smaller budgets follow from it."""
    texts, y = P.eval_set(data["test"])
    kv = load_vectors()
    probs = model.predict(texts)
    cache, out = {}, {}
    records = []
    start = time.time()
    for t, label, p in zip(texts, y, probs):
        if label == 1 and p >= 0.5:
            records.append(attack(model, t, kv, cache, max(P.BUDGETS)) + (max(1, int(max(P.BUDGETS) * len(t.split()))), len(t.split())))
    detected = len(records)
    for budget in P.BUDGETS:
        flips = sum(1 for used, _, _, n in records if used is not None and used <= max(1, int(budget * n)))
        recall = (detected - flips) / int(y.sum())
        tp = detected - flips
        fp = int(((probs >= 0.5) & (y == 0)).sum())
        precision = tp / max(tp + fp, 1)
        out[str(budget)] = {"recall": recall, "precision": precision, "f1": 2 * precision * recall / max(precision + recall, 1e-9),
                            "attack_success": flips / max(detected, 1)}
    used = [u for u, _, _, _ in records if u]
    out.update(avg_queries=float(np.mean([q for _, q, _, _ in records])), avg_substitutions=float(np.mean(used)) if used else 0.0,
               seconds=time.time() - start)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--task"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true"); ap.add_argument("--attack", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    os.makedirs(P.RESULTS, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    jobs = [("tfidf_lr", t, s) for t in P.TASKS for s in (0,)] if args.baseline else [(args.model, args.task, args.seed)]
    for name, task, seed in jobs:
        tag = f"{name.replace('/', '_')}__{task}__s{seed}{'__aug' if args.augment else ''}"
        if os.path.exists(os.path.join(P.RESULTS, tag + ".json")):
            print(tag, "already done", flush=True)
            continue
        data = P.load_task(task)
        salience = P.salient_words(data["train"])
        train = augmented(data["train"], salience, seed) if args.augment else data["train"]
        start = time.time()
        model = Baseline(seed) if name == "tfidf_lr" else Classifier(name, seed)
        best = model.fit(train, data["val"], seed)
        res = {"model": name, "task": task, "seed": seed, "augment": args.augment, "val_f1": best,
               "train_n": len(train[0]), "train_pos": float(np.mean(train[1])), "train_seconds": time.time() - start}
        res.update(evaluate(model, data, salience))
        if args.attack:
            res["attack"] = run_attack(model, data)
        json.dump(res, open(os.path.join(P.RESULTS, tag + ".json"), "w"), indent=1)
        print(tag, "clean F1", round(res["clean"]["f1"], 4), "mix@0.2 F1", round(res["organic"]["mix@0.2"]["f1"], 4),
              "attack" if args.attack else "", res.get("attack", {}).get("0.2", ""), f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
