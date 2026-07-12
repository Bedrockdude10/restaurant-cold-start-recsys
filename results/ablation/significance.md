# Ablation significance (full_model vs. ablation)

Δ = full − ablation. Paired t-test where per-seed runs exist for both sides, else Welch two-sample from `summary.csv`. p-values Holm–Bonferroni corrected; **bold** = significant at α=0.05.

| Comparison | Split | Metric | full | abl. | Δ | 95% CI | p | p (Holm) | method |
|---|---|---|---:|---:|---:|:---:|---:|---:|---|
| Full vs No Restaurant Context | warm | Hit@5 | 0.2718 | 0.2620 | **+0.0098** | [+0.0069, +0.0128] | <0.001 | 0.002 | welch-t |
| Full vs No Restaurant Context | cold_restaurant | Hit@5 | 0.0814 | 0.0671 | +0.0143 | [+0.0007, +0.0279] | 0.042 | 0.433 | welch-t |
| Full vs No Restaurant Context | cold_user | Hit@5 | 0.2921 | 0.2901 | +0.0021 | [-0.0100, +0.0141] | 0.700 | 1.000 | welch-t |
| Full vs No Restaurant Context | warm | NDCG@10 | 0.2311 | 0.2243 | **+0.0069** | [+0.0049, +0.0089] | <0.001 | 0.001 | welch-t |
| Full vs No Restaurant Context | cold_restaurant | NDCG@10 | 0.0772 | 0.0650 | +0.0122 | [+0.0001, +0.0243] | 0.048 | 0.433 | welch-t |
| Full vs No Restaurant Context | cold_user | NDCG@10 | 0.2563 | 0.2515 | +0.0048 | [-0.0065, +0.0160] | 0.351 | 1.000 | welch-t |
| Full vs No Restaurant Content | warm | Hit@5 | 0.2718 | 0.2548 | **+0.0170** | [+0.0129, +0.0211] | <0.001 | 0.001 | welch-t |
| Full vs No Restaurant Content | cold_restaurant | Hit@5 | 0.0814 | 0.0862 | -0.0048 | [-0.0185, +0.0089] | 0.385 | 1.000 | welch-t |
| Full vs No Restaurant Content | cold_user | Hit@5 | 0.2921 | 0.2995 | -0.0074 | [-0.0245, +0.0097] | 0.340 | 1.000 | welch-t |
| Full vs No Restaurant Content | warm | NDCG@10 | 0.2311 | 0.2167 | **+0.0145** | [+0.0109, +0.0180] | <0.001 | 0.002 | welch-t |
| Full vs No Restaurant Content | cold_restaurant | NDCG@10 | 0.0772 | 0.0777 | -0.0004 | [-0.0115, +0.0106] | 0.916 | 1.000 | welch-t |
| Full vs No Restaurant Content | cold_user | NDCG@10 | 0.2563 | 0.2619 | -0.0057 | [-0.0181, +0.0067] | 0.314 | 1.000 | welch-t |
| Full vs No User Context | warm | Hit@5 | 0.2718 | 0.2640 | **+0.0077** | [+0.0049, +0.0106] | <0.001 | 0.005 | welch-t |
| Full vs No User Context | cold_restaurant | Hit@5 | 0.0814 | 0.0668 | +0.0146 | [+0.0010, +0.0282] | 0.039 | 0.433 | welch-t |
| Full vs No User Context | cold_user | Hit@5 | 0.2921 | 0.2506 | **+0.0416** | [+0.0287, +0.0545] | <0.001 | 0.002 | welch-t |
| Full vs No User Context | warm | NDCG@10 | 0.2311 | 0.2243 | **+0.0069** | [+0.0046, +0.0091] | <0.001 | 0.003 | welch-t |
| Full vs No User Context | cold_restaurant | NDCG@10 | 0.0772 | 0.0624 | +0.0149 | [+0.0033, +0.0264] | 0.018 | 0.220 | welch-t |
| Full vs No User Context | cold_user | NDCG@10 | 0.2563 | 0.2154 | **+0.0409** | [+0.0331, +0.0486] | <0.001 | <0.001 | welch-t |
| Full vs No User Content | warm | Hit@5 | 0.2718 | 0.2748 | -0.0030 | [-0.0062, +0.0002] | 0.063 | 0.441 | welch-t |
| Full vs No User Content | cold_restaurant | Hit@5 | 0.0814 | 0.0321 | **+0.0492** | [+0.0356, +0.0628] | <0.001 | 0.003 | welch-t |
| Full vs No User Content | cold_user | Hit@5 | 0.2921 | 0.3178 | **-0.0256** | [-0.0368, -0.0145] | 0.001 | 0.021 | welch-t |
| Full vs No User Content | warm | NDCG@10 | 0.2311 | 0.2332 | -0.0020 | [-0.0040, -0.0000] | 0.047 | 0.433 | welch-t |
| Full vs No User Content | cold_restaurant | NDCG@10 | 0.0772 | 0.0306 | **+0.0466** | [+0.0358, +0.0575] | <0.001 | 0.002 | welch-t |
| Full vs No User Content | cold_user | NDCG@10 | 0.2563 | 0.2861 | **-0.0298** | [-0.0378, -0.0218] | <0.001 | 0.001 | welch-t |
| Full vs Content × Content | warm | Hit@5 | 0.2718 | 0.2502 | **+0.0216** | [+0.0172, +0.0260] | <0.001 | <0.001 | welch-t |
| Full vs Content × Content | cold_restaurant | Hit@5 | 0.0814 | 0.0551 | +0.0263 | [+0.0066, +0.0460] | 0.016 | 0.206 | welch-t |
| Full vs Content × Content | cold_user | Hit@5 | 0.2921 | 0.2381 | **+0.0540** | [+0.0402, +0.0679] | <0.001 | <0.001 | welch-t |
| Full vs Content × Content | warm | NDCG@10 | 0.2311 | 0.2147 | **+0.0164** | [+0.0134, +0.0194] | <0.001 | <0.001 | welch-t |
| Full vs Content × Content | cold_restaurant | NDCG@10 | 0.0772 | 0.0522 | +0.0251 | [+0.0088, +0.0413] | 0.008 | 0.112 | welch-t |
| Full vs Content × Content | cold_user | NDCG@10 | 0.2563 | 0.2074 | **+0.0488** | [+0.0403, +0.0574] | <0.001 | <0.001 | welch-t |
| Full vs Context × Context | warm | Hit@5 | 0.2718 | 0.2762 | -0.0044 | [-0.0069, -0.0018] | 0.005 | 0.072 | welch-t |
| Full vs Context × Context | cold_restaurant | Hit@5 | 0.0814 | 0.0299 | **+0.0514** | [+0.0380, +0.0649] | <0.001 | 0.003 | welch-t |
| Full vs Context × Context | cold_user | Hit@5 | 0.2921 | 0.3166 | -0.0245 | [-0.0387, -0.0103] | 0.004 | 0.066 | welch-t |
| Full vs Context × Context | warm | NDCG@10 | 0.2311 | 0.2347 | **-0.0035** | [-0.0054, -0.0016] | 0.003 | 0.043 | welch-t |
| Full vs Context × Context | cold_restaurant | NDCG@10 | 0.0772 | 0.0292 | **+0.0481** | [+0.0370, +0.0592] | <0.001 | <0.001 | welch-t |
| Full vs Context × Context | cold_user | NDCG@10 | 0.2563 | 0.2861 | **-0.0298** | [-0.0401, -0.0196] | <0.001 | 0.003 | welch-t |
