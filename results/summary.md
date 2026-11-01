# Results summary

hatexplain tfidf_lr n=1 clean F1 0.795 benign changed by normalizer 0.019
hatexplain roberta-base n=3 clean F1 0.854 (0.004) benign changed by normalizer 0.019
hatexplain bert-base-uncased n=1 clean F1 0.835 benign changed by normalizer 0.019
dyna_exp tfidf_lr n=1 clean F1 0.523 benign changed by normalizer 0.105
dyna_exp roberta-base n=3 clean F1 0.746 (0.009) benign changed by normalizer 0.105
dyna_exp bert-base-uncased n=1 clean F1 0.697 benign changed by normalizer 0.105
dyna_imp tfidf_lr n=1 clean F1 0.127 benign changed by normalizer 0.105
dyna_imp roberta-base n=3 clean F1 0.388 (0.034) benign changed by normalizer 0.105
dyna_imp bert-base-uncased n=1 clean F1 0.320 benign changed by normalizer 0.105

## Clean

HateXplain | TF-IDF+LR | 1 | 0.827 | 0.850 | 0.807 | 0.782
HateXplain | RoBERTa | 3 | 0.884 (0.004) | 0.897 (0.003) | 0.889 (0.016) | 0.821 (0.009)
HateXplain | BERT | 1 | 0.860 | 0.877 | 0.860 | 0.811
DynaHate-Exp | TF-IDF+LR | 1 | 0.628 | 0.682 | 0.588 | 0.471
DynaHate-Exp | RoBERTa | 3 | 0.816 (0.004) | 0.839 (0.003) | 0.798 (0.006) | 0.701 (0.015)
DynaHate-Exp | BERT | 1 | 0.786 | 0.809 | 0.787 | 0.625
DynaHate-Imp | TF-IDF+LR | 1 | 0.127 | 0.516 | 0.075 | 0.425
DynaHate-Imp | RoBERTa | 3 | 0.388 (0.034) | 0.652 (0.019) | 0.280 (0.030) | 0.637 (0.062)
DynaHate-Imp | BERT | 1 | 0.320 | 0.617 | 0.216 | 0.620

## Recall at budget 0.2

HateXplain | TF-IDF+LR | 0.81 | 0.27 | 0.21 | 0.21 | 0.17 | 0.15 | 0.81 | 0.19
HateXplain | RoBERTa | 0.89 (0.02) | 0.50 | 0.42 | 0.39 | 0.60 | 0.77 | 0.89 | 0.57
HateXplain | BERT | 0.86 | 0.40 | 0.25 | 0.32 | 0.45 | 0.61 | 0.85 | 0.41
DynaHate-Exp | TF-IDF+LR | 0.59 | 0.27 | 0.22 | 0.17 | 0.21 | 0.14 | 0.57 | 0.20
DynaHate-Exp | RoBERTa | 0.80 (0.01) | 0.96 | 0.96 | 0.79 | 0.89 | 0.87 | 0.80 | 0.92
DynaHate-Exp | BERT | 0.79 | 0.99 | 0.97 | 0.81 | 0.83 | 0.83 | 0.79 | 0.93
DynaHate-Imp | TF-IDF+LR | 0.07 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.07 | 0.02
DynaHate-Imp | RoBERTa | 0.28 (0.03) | 0.17 | 0.16 | 0.12 | 0.20 | 0.14 | 0.27 | 0.17
DynaHate-Imp | BERT | 0.22 | 0.19 | 0.15 | 0.19 | 0.19 | 0.15 | 0.22 | 0.18

## Attack, seed 0

HateXplain | RoBERTa | 0.89 | 0.65 | 0.79 | 0.57 | 0.75 | 0.52 | 0.72 | 0.19 | 56
HateXplain | BERT | 0.86 | 0.52 | 0.74 | 0.41 | 0.70 | 0.32 | 0.65 | 0.24 | 57
DynaHate-Exp | RoBERTa | 0.80 | 0.90 | 0.49 | 0.93 | 0.39 | 0.95 | 0.31 | 0.61 | 46
DynaHate-Exp | BERT | 0.79 | 0.90 | 0.51 | 0.93 | 0.43 | 0.95 | 0.37 | 0.53 | 47
DynaHate-Imp | RoBERTa | 0.30 | 0.24 | 0.09 | 0.22 | 0.04 | 0.24 | 0.03 | 0.91 | 43
DynaHate-Imp | BERT | 0.22 | 0.18 | 0.09 | 0.18 | 0.08 | 0.20 | 0.06 | 0.73 | 56

## Mitigations, RoBERTa seed 0

HateXplain | 0.89 | 0.57 | 0.87 | 1.00 | 0.10 | 0.05 | 0.09 | 0.97
DynaHate-Exp | 0.80 | 0.93 | 0.80 | 0.99 | 0.14 | 0.58 | 0.19 | 0.98
DynaHate-Imp | 0.30 | 0.22 | 0.30 | 0.99 | 0.05 | 0.07 | 0.06 | 0.98
