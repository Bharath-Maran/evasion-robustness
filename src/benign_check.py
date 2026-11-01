"""Does a model treat disguised spelling itself as a sign of hate? Applies each transform to benign text too.

Usage: python src/benign_check.py --task dyna_exp [--augment]
The main evaluation disguises hateful texts only, so a model that flags any odd spelling would look robust there.
"""

import argparse
import json
import os
import random

import numpy as np

import pipeline as P
from run_experiments import STRATEGIES, Classifier, augmented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    tag = f"benign_check__{args.task}__s{args.seed}{'__aug' if args.augment else ''}"
    path = os.path.join(P.RESULTS, tag + ".json")
    if os.path.exists(path):
        return
    random.seed(args.seed); np.random.seed(args.seed)
    data = P.load_task(args.task)
    salience = P.salient_words(data["train"])
    train = augmented(data["train"], salience, args.seed) if args.augment else data["train"]
    model = Classifier("roberta-base", args.seed)
    model.fit(train, data["val"], args.seed)
    texts, y = P.eval_set(data["test"])
    benign = [t for t, l in zip(texts, y) if l == 0]
    res = {"task": args.task, "seed": args.seed, "augment": args.augment, "n_benign": len(benign),
           "clean": P.metrics(y, model.predict(texts)), "fpr_clean": float((model.predict(benign) >= 0.5).mean()), "strategies": {}}
    for name in STRATEGIES:
        evaded = [P.evade(t, name, salience, 0.2, 0) for t in benign]
        both = P.evade_set(texts, np.ones(len(y), dtype=int), name, salience, 0.2)
        res["strategies"][name] = {"fpr": float((model.predict(evaded) >= 0.5).mean()),
                                   "fpr_normalized": float((model.predict([P.normalize(t) for t in evaded]) >= 0.5).mean()),
                                   "changed": float(np.mean([a != b for a, b in zip(benign, evaded)])),
                                   "both_classes": P.metrics(y, model.predict(both))}
    json.dump(res, open(path, "w"), indent=1)
    print(tag, "clean F1", round(res["clean"]["f1"], 4), "fpr", round(res["fpr_clean"], 3), "mix fpr", round(res["strategies"]["mix"]["fpr"], 3), flush=True)


if __name__ == "__main__":
    main()
