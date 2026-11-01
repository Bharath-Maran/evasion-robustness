# Evasion-Robust Hate Speech Detection

Code and aggregate results for a CS 59300-03 NLP final project: how much do hate speech classifiers degrade under
organic evasion (leetspeak, homoglyphs, spacing, repeated letters, algospeak) compared with an ML-optimized
synonym attack at the same perturbation budget, on explicit and implicit hate?

**Content warning:** this project concerns offensive language. Datasets are not redistributed, and the repository
holds aggregate numbers only, never evaded or generated hateful text.

## Layout

- `src/pipeline.py`: data loading, pair-aware DynaHate split, evasion transforms, normalizer, metrics
- `src/run_experiments.py`: training, evaluation, and the greedy synonym attack
- `src/run_all.sh`: reproduces every run (finished runs are skipped)
- `src/aggregate.py`: builds `results/summary.md` and the LaTeX tables
- `results/`: one JSON file per run

## Reproducing

Place these in `data/`: `hatexplain.json` and `hatexplain_splits.json` (HateXplain), and `dynahate.csv` (DynaHate).
The attack reads word2vec-google-news-300 from `~/gensim-data/` in gensim `.kv` format.

    pip install -r requirements.txt
    cd src && ./run_all.sh && python aggregate.py
