#!/bin/bash
# Reproduces every result in results/. Run from src/.
set -u
PY=${PY:-python}
$PY run_experiments.py --baseline
for task in hatexplain dyna_exp dyna_imp; do
  $PY run_experiments.py --model roberta-base --task $task --seed 0 --attack
  $PY run_experiments.py --model bert-base-uncased --task $task --seed 0 --attack
  $PY run_experiments.py --model roberta-base --task $task --seed 0 --augment
done
for seed in 1 2; do
  for task in hatexplain dyna_exp dyna_imp; do
    $PY run_experiments.py --model roberta-base --task $task --seed $seed
  done
done
for task in hatexplain dyna_exp dyna_imp; do
  $PY benign_check.py --task $task
  $PY benign_check.py --task $task --augment
done
$PY aggregate.py
