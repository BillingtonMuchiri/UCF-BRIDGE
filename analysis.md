## 1. Core comparison with up to 15 seeds (CodeT5→PLBART, CONCODE)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native>plbart|none|full|ep3|q0|concode_java | 15 | 29.62 ± 0.33 [29.44, 29.81] | 27.5 | 20.2 | 99.2 | 96.4 | – | 139.2 | 0.338 | 9.2 |
| codet5>plbart|naive|full|ep3|q0|concode_java | 15 | 17.77 ± 0.38 [17.56, 17.98] | 18.4 | 0.8 | 98.7 | 60.3 | – | 95.9 | 0.577 | 13.4 |
| codet5>plbart|resampler|full|ep3|q32|concode_java | 15 | 22.85 ± 0.47 [22.58, 23.11] | 23.2 | 6.3 | 96.9 | 93.1 | – | 110.7 | 0.501 | 13.5 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| linear − naive | 15 | +7.88 (sd 0.46) [+7.62, +8.13] (15/15 seeds > 0) | +7.88 [+7.05, +8.70] | +10.1 [+8.5, +11.9] | – | 0.0002 | 0.0000 |
| linear − resampler | 15 | +2.80 (sd 0.60) [+2.47, +3.13] (15/15 seeds > 0) | +2.80 [+2.39, +3.22] | +4.7 [+3.8, +5.7] | – | 0.0002 | 0.0000 |
| linear − native | 15 | -3.98 (sd 0.46) [-4.23, -3.72] (0/15 seeds > 0) | -3.98 [-4.76, -3.19] | -9.2 [-10.9, -7.6] | – | 0.0002 | 0.0000 |
| resampler − naive | 15 | +5.08 (sd 0.52) [+4.79, +5.36] (15/15 seeds > 0) | +5.08 [+4.34, +5.83] | +5.5 [+4.3, +6.7] | – | 0.0002 | 0.0000 |

Replication check against Table X (first five seeds; differences > ~0.7 BLEU suggest the data subset or implementation differs from ucf_revision_kaggle.py):

| system | seed | this script | Table X | diff |
|---|---|---|---|---|
| native | 13 | 30.0 | 25.8 | +4.2 |
| native | 42 | 29.5 | 24.6 | +4.9 |
| native | 2024 | 29.8 | 24.7 | +5.1 |
| native | 7 | 29.9 | 25.2 | +4.7 |
| native | 99 | 30.0 | 25.4 | +4.6 |
| naive | 13 | 18.2 | 14.5 | +3.7 |
| naive | 42 | 17.9 | 15.8 | +2.1 |
| naive | 2024 | 18.2 | 14.6 | +3.6 |
| naive | 7 | 18.0 | 15.8 | +2.2 |
| naive | 99 | 18.2 | 15.6 | +2.6 |
| resampler | 13 | 23.4 | 20.0 | +3.4 |
| resampler | 42 | 22.6 | 19.5 | +3.1 |
| resampler | 2024 | 22.9 | 19.5 | +3.4 |
| resampler | 7 | 21.9 | 19.7 | +2.2 |
| resampler | 99 | 23.4 | 20.1 | +3.3 |
| linear | 13 | 25.3 | 23.2 | +2.1 |
| linear | 42 | 26.1 | 22.5 | +3.6 |
| linear | 2024 | 26.2 | 22.2 | +4.0 |
| linear | 7 | 25.4 | 22.9 | +2.5 |
| linear | 99 | 26.4 | 23.1 | +3.3 |

First five seeds vs all seeds (does the 5-seed interval in the paper hold up?):

| system | mean (5 seeds) | mean (all) | sd (5) | sd (all) |
|---|---|---|---|---|
| native | 29.85 | 29.62 | 0.20 | 0.33 |
| naive | 18.11 | 17.77 | 0.16 | 0.38 |
| resampler | 22.84 | 22.85 | 0.63 | 0.47 |
| linear | 25.89 | 25.65 | 0.53 | 0.49 |

## 1b. CodeT5→CodeGPT with up to 15 seeds

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native | 15 | 25.73 ± 0.39 [25.52, 25.95] | 26.5 | 14.7 | 93.8 | 95.1 | – | 124.4 | 0.348 | 16.8 |
| naive | 15 | 12.93 ± 0.65 [12.57, 13.30] | 14.1 | 0.2 | 97.6 | 17.8 | – | 124.4 | 0.572 | 23.4 |
| linear | 15 | 23.37 ± 1.34 [22.63, 24.11] | 24.9 | 9.2 | 93.3 | 91.3 | – | 125.0 | 0.412 | 23.5 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| linear − naive | 15 | +10.44 (sd 1.60) [+9.55, +11.33] (15/15 seeds > 0) | +10.44 [+9.62, +11.30] | +8.9 [+7.5, +10.5] | – | 0.0002 | 0.0000 |
| linear − native CodeGPT | 15 | -2.36 (sd 1.54) [-3.21, -1.51] (0/15 seeds > 0) | -2.36 [-3.08, -1.65] | -5.5 [-7.1, -4.1] | – | 0.0002 | 0.0000 |

## 2. Non-CodeT5 encoders (fixed decoder = PLBART; CodeGPT with UniXcoder)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|concode_java | 15 | 17.77 ± 0.38 [17.56, 17.98] | 18.4 | 0.8 | 98.7 | 60.3 | – | 95.9 | 0.577 | 13.4 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |
| codet5p>plbart|naive|full|ep3|q0|concode_java | 5 | 18.13 ± 0.29 [17.77, 18.50] | 18.6 | 0.6 | 98.0 | 57.8 | – | 95.9 | 0.587 | 13.7 |
| codet5p>plbart|linear|full|ep3|q0|concode_java | 5 | 25.70 ± 0.24 [25.41, 26.00] | 25.0 | 12.3 | 97.0 | 93.7 | – | 96.5 | 0.417 | 13.6 |
| unixcoder>plbart|naive|full|ep3|q0|concode_java | 5 | 10.33 ± 0.52 [9.69, 10.98] | 11.5 | 0.3 | 67.6 | 65.9 | – | 95.9 | 1.169 | 11.1 |
| unixcoder>plbart|linear|full|ep3|q0|concode_java | 5 | 16.35 ± 0.32 [15.96, 16.75] | 17.4 | 0.9 | 97.7 | 83.9 | – | 96.5 | 0.553 | 11.5 |
| roberta>plbart|naive|full|ep3|q0|concode_java | 5 | 11.12 ± 1.06 [9.80, 12.43] | 12.4 | 0.3 | 98.9 | 25.4 | – | 95.9 | 0.610 | 10.9 |
| roberta>plbart|linear|full|ep3|q0|concode_java | 5 | 20.44 ± 0.49 [19.83, 21.05] | 21.0 | 3.2 | 97.0 | 85.5 | – | 96.5 | 0.529 | 11.5 |
| codebert>plbart|naive|full|ep3|q0|concode_java | 5 | 14.54 ± 0.79 [13.57, 15.52] | 15.3 | 0.4 | 97.5 | 31.2 | – | 95.9 | 0.606 | 11.0 |
| codebert>plbart|linear|full|ep3|q0|concode_java | 5 | 18.13 ± 0.19 [17.90, 18.37] | 18.6 | 0.9 | 97.7 | 72.3 | – | 96.5 | 0.553 | 11.6 |
| unixcoder>codegpt|naive|full|ep3|q0|concode_java | 5 | 11.35 ± 0.28 [11.01, 11.70] | 12.7 | 0.2 | 97.6 | 27.9 | – | 124.4 | 0.572 | 20.5 |
| unixcoder>codegpt|linear|full|ep3|q0|concode_java | 5 | 14.40 ± 0.40 [13.90, 14.91] | 15.7 | 0.3 | 94.9 | 67.6 | – | 125.0 | 0.538 | 20.9 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| unixcoder→PLBART linear − unixcoder→PLBART naive | 5 | +6.02 (sd 0.49) [+5.42, +6.62] (5/5 seeds > 0) | +6.02 [+5.32, +6.72] | +0.6 [+0.1, +1.1] | – | 0.0002 | 0.0000 |
| unixcoder→PLBART linear − codet5→PLBART linear | 5 | -9.53 (sd 0.49) [-10.14, -8.92] (0/5 seeds > 0) | -9.53 [-10.54, -8.60] | -10.4 [-12.2, -8.8] | – | 0.0002 | 0.0000 |
| roberta→PLBART linear − roberta→PLBART naive | 5 | +9.33 (sd 1.24) [+7.78, +10.87] (5/5 seeds > 0) | +9.33 [+8.44, +10.25] | +2.9 [+1.9, +3.9] | – | 0.0002 | 0.0001 |
| roberta→PLBART linear − codet5→PLBART linear | 5 | -5.45 (sd 0.86) [-6.51, -4.38] (0/5 seeds > 0) | -5.45 [-6.19, -4.75] | -8.1 [-9.6, -6.6] | – | 0.0002 | 0.0001 |
| codebert→PLBART linear − codebert→PLBART naive | 5 | +3.59 (sd 0.74) [+2.67, +4.50] (5/5 seeds > 0) | +3.59 [+2.87, +4.32] | +0.4 [+0.0, +0.9] | – | 0.0002 | 0.0004 |
| codebert→PLBART linear − codet5→PLBART linear | 5 | -7.75 (sd 0.60) [-8.50, -7.01] (0/5 seeds > 0) | -7.75 [-8.58, -6.97] | -10.4 [-12.2, -8.8] | – | 0.0002 | 0.0000 |
| unixcoder→CodeGPT linear − unixcoder→CodeGPT naive | 5 | +3.05 (sd 0.22) [+2.78, +3.32] (5/5 seeds > 0) | +3.05 [+2.37, +3.73] | +0.2 [+0.0, +0.4] | – | 0.0002 | 0.0000 |

## 3. Frozen-decoder (bridge-only) condition

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|frozen|ep3|q0|concode_java | 1 | 0.52 ± nan [nan, nan] | 11.2 | 0.0 | 5.1 | 83.3 | – | 0.0 | 1.681 | 0.0 |
| codet5>plbart|linear|frozen|ep3|q0|concode_java | 5 | 25.15 ± 0.22 [24.87, 25.42] | 24.8 | 10.0 | 93.9 | 93.7 | – | 0.6 | 0.428 | 10.9 |
| codet5>plbart|linear|frozen|ep10|q0|concode_java | 5 | 26.63 ± 0.38 [26.15, 27.10] | 26.2 | 10.7 | 96.4 | 96.3 | – | 0.6 | 0.484 | 38.2 |
| codet5>plbart|resampler|frozen|ep3|q32|concode_java | 5 | 22.08 ± 0.26 [21.76, 22.40] | 22.6 | 4.7 | 94.2 | 91.3 | – | 14.8 | 0.515 | 10.8 |
| codet5>plbart|calm|frozen|ep3|q0|concode_java | 5 | 23.00 ± 0.60 [22.26, 23.74] | 23.3 | 5.2 | 95.0 | 90.6 | – | 7.7 | 0.505 | 11.8 |
| codet5>codegpt|naive|frozen|ep3|q0|concode_java | 1 | 2.19 ± nan [nan, nan] | 12.1 | 0.0 | 0.0 | 99.5 | – | 0.0 | 2.789 | 0.0 |
| codet5>codegpt|linear|frozen|ep3|q0|concode_java | 5 | 14.59 ± 0.73 [13.68, 15.51] | 18.1 | 0.5 | 76.3 | 60.7 | – | 0.6 | 0.657 | 18.6 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| linear frozen − linear full-tune | 5 | -0.74 (sd 0.48) [-1.33, -0.15] (0/5 seeds > 0) | -0.74 [-1.23, -0.27] | -1.2 [-2.1, -0.3] | – | 0.0018 | 0.0254 |
| linear frozen − resampler frozen | 5 | +3.06 (sd 0.16) [+2.86, +3.26] (5/5 seeds > 0) | +3.06 [+2.55, +3.61] | +5.3 [+4.2, +6.5] | – | 0.0002 | 0.0000 |
| CodeGPT linear frozen − CodeGPT linear full | 5 | -8.95 (sd 1.82) [-11.21, -6.68] (0/5 seeds > 0) | -8.95 [-10.28, -7.55] | -9.0 [-10.6, -7.5] | – | 0.0002 | 0.0004 |

## 4. Longer resampler training (equal budget for the linear bridge)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|resampler|full|ep3|q32|concode_java | 15 | 22.85 ± 0.47 [22.58, 23.11] | 23.2 | 6.3 | 96.9 | 93.1 | – | 110.7 | 0.501 | 13.5 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |
| codet5>plbart|resampler|full|ep6|q32|concode_java | 5 | 24.57 ± 0.51 [23.94, 25.21] | 25.2 | 7.7 | 98.5 | 95.5 | – | 110.7 | 0.564 | 27.3 |
| codet5>plbart|linear|full|ep6|q0|concode_java | 5 | 27.66 ± 0.33 [27.24, 28.07] | 27.2 | 11.4 | 98.8 | 96.1 | – | 96.5 | 0.485 | 28.4 |
| codet5>plbart|resampler|full|ep12|q32|concode_java | 5 | 25.95 ± 0.36 [25.50, 26.39] | 26.3 | 7.9 | 99.2 | 95.9 | – | 110.7 | 0.669 | 53.8 |
| codet5>plbart|linear|full|ep12|q0|concode_java | 5 | 28.81 ± 0.35 [28.37, 29.25] | 28.1 | 12.1 | 99.5 | 96.6 | – | 96.5 | 0.584 | 57.8 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| linear ep3 − resampler ep3 | 15 | +2.80 (sd 0.60) [+2.47, +3.13] (15/15 seeds > 0) | +2.80 [+2.39, +3.22] | +4.7 [+3.8, +5.7] | – | 0.0002 | 0.0000 |
| linear ep6 − resampler ep6 | 5 | +3.08 (sd 0.32) [+2.69, +3.48] (5/5 seeds > 0) | +3.08 [+2.49, +3.66] | +3.7 [+2.7, +4.8] | – | 0.0002 | 0.0000 |
| linear ep12 − resampler ep12 | 5 | +2.86 (sd 0.60) [+2.11, +3.61] (5/5 seeds > 0) | +2.86 [+2.23, +3.51] | +4.2 [+3.1, +5.4] | – | 0.0002 | 0.0004 |

Validation BPC by epoch (mean over seeds):

- codet5>plbart|linear|frozen|ep10|q0|concode_java: ep1=0.512, ep2=0.437, ep3=0.423, ep4=0.422, ep5=0.434, ep6=0.444, ep7=0.452, ep8=0.466, ep9=0.476, ep10=0.484
- codet5>plbart|linear|full|ep12|q0|concode_java: ep1=0.503, ep2=0.440, ep3=0.444, ep4=0.460, ep5=0.487, ep6=0.506, ep7=0.529, ep8=0.545, ep9=0.558, ep10=0.569, ep11=0.579, ep12=0.584
- codet5>plbart|linear|full|ep20|q0|mbpp_py: ep1=0.752, ep2=0.656, ep3=0.664, ep4=0.692, ep5=0.704, ep6=0.724, ep7=0.739, ep8=0.749, ep9=0.761
- codet5>plbart|linear|full|ep6|q0|concode_java: ep1=0.480, ep2=0.435, ep3=0.439, ep4=0.454, ep5=0.473, ep6=0.485
- codet5>plbart|resampler|full|ep12|q32|concode_java: ep1=0.556, ep2=0.526, ep3=0.529, ep4=0.544, ep5=0.569, ep6=0.590, ep7=0.611, ep8=0.627, ep9=0.640, ep10=0.652, ep11=0.662, ep12=0.669
- codet5>plbart|resampler|full|ep6|q32|concode_java: ep1=0.548, ep2=0.518, ep3=0.518, ep4=0.528, ep5=0.552, ep6=0.564
- native>plbart|none|full|ep20|q0|mbpp_py: ep1=0.571, ep2=0.563, ep3=0.596, ep4=0.631, ep5=0.648, ep6=0.665

## 5. Additional dataset: csn_java

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native>plbart|none|full|ep3|q0|csn_java | 5 | 18.16 ± 0.24 [17.87, 18.46] | 21.2 | 0.0 | 94.9 | 99.9 | – | 139.2 | 0.491 | 7.8 |
| native>plbart|none|lora|ep3|q0|csn_java | 5 | 16.67 ± 0.23 [16.38, 16.95] | 20.1 | 0.0 | 93.8 | 99.8 | – | 0.9 | 0.494 | 5.4 |
| native>plbart|none|adapter64|ep3|q0|csn_java | 5 | 16.75 ± 0.20 [16.50, 17.00] | 20.2 | 0.0 | 93.6 | 99.8 | – | 1.2 | 0.488 | 4.9 |
| codet5>plbart|naive|full|ep3|q0|csn_java | 5 | 10.40 ± 0.14 [10.22, 10.58] | 16.8 | 0.0 | 93.8 | 97.3 | – | 95.9 | 0.607 | 9.9 |
| codet5>plbart|linear|full|ep3|q0|csn_java | 5 | 16.27 ± 0.43 [15.73, 16.81] | 20.0 | 0.0 | 93.3 | 99.8 | – | 96.5 | 0.529 | 9.8 |
| codet5>plbart|resampler|full|ep3|q32|csn_java | 5 | 15.01 ± 0.13 [14.85, 15.17] | 19.2 | 0.0 | 92.8 | 99.7 | – | 110.7 | 0.545 | 10.1 |
| codet5>plbart|calm|full|ep3|q0|csn_java | 5 | 14.77 ± 0.19 [14.53, 15.00] | 19.0 | 0.0 | 92.2 | 99.9 | – | 103.6 | 0.560 | 9.8 |
| codet5>plbart|linear|lora|ep3|q0|csn_java | 5 | 15.94 ± 0.31 [15.55, 16.33] | 19.4 | 0.0 | 92.2 | 99.9 | – | 1.2 | 0.528 | 8.2 |
| plbart>plbart|naive|full|ep3|q0|csn_java | 5 | 16.15 ± 0.12 [16.00, 16.30] | 19.8 | 0.0 | 94.4 | 99.8 | – | 95.9 | 0.502 | 7.5 |
| plbart>plbart|linear|full|ep3|q0|csn_java | 5 | 16.02 ± 0.28 [15.67, 16.37] | 19.7 | 0.0 | 92.6 | 99.9 | – | 96.5 | 0.520 | 7.6 |
| codet5>codegpt|naive|full|ep3|q0|csn_java | 5 | 4.23 ± 0.51 [3.59, 4.87] | 12.7 | 0.0 | 59.7 | 6.1 | – | 124.4 | 0.418 | 16.1 |
| codet5>codegpt|linear|full|ep3|q0|csn_java | 5 | 5.86 ± 2.16 [3.18, 8.55] | 13.7 | 0.0 | 61.6 | 46.9 | – | 125.0 | 0.405 | 16.0 |
| native>codegpt|none|full|ep3|q0|csn_java | 5 | 15.32 ± 0.22 [15.04, 15.59] | 19.2 | 0.0 | 56.7 | 99.8 | – | 124.4 | 0.346 | 12.9 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CodeT5→PLBART naive | 5 | +5.87 (sd 0.35) [+5.43, +6.31] (5/5 seeds > 0) | +5.87 [+5.29, +6.49] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − CodeT5→PLBART resampler | 5 | +1.26 (sd 0.54) [+0.59, +1.93] (5/5 seeds > 0) | +1.26 [+0.96, +1.56] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0063 |
| CodeT5→PLBART linear − native PLBART | 5 | -1.89 (sd 0.51) [-2.53, -1.25] (0/5 seeds > 0) | -1.89 [-2.35, -1.43] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0012 |
| CodeT5→PLBART linear − native PLBART+LoRA | 5 | -0.39 (sd 0.41) [-0.90, +0.12] (0/5 seeds > 0) | -0.39 [-0.92, +0.16] | +0.0 [+0.0, +0.0] | – | 0.1536 | 0.0990 |
| CodeT5→PLBART linear − PLBART→PLBART naive | 5 | +0.12 (sd 0.43) [-0.41, +0.65] (3/5 seeds > 0) | +0.12 [-0.43, +0.67] | +0.0 [+0.0, +0.0] | – | 0.6649 | 0.5662 |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive | 5 | +1.64 (sd 1.66) [-0.43, +3.70] (5/5 seeds > 0) | +1.64 [+1.40, +1.87] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0927 |
| CodeT5→CodeGPT linear − native CodeGPT | 5 | -9.45 (sd 2.17) [-12.15, -6.76] (0/5 seeds > 0) | -9.45 [-10.22, -8.69] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0006 |

## 5. Additional dataset: csn_python

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native>plbart|none|full|ep3|q0|csn_python | 5 | 37.44 ± 0.28 [37.09, 37.79] | 33.0 | 0.0 | 0.1 | 100.0 | – | 139.2 | 0.487 | 6.9 |
| native>plbart|none|lora|ep3|q0|csn_python | 5 | 35.74 ± 0.05 [35.67, 35.80] | 32.0 | 0.0 | 0.0 | 100.0 | – | 0.9 | 0.495 | 4.6 |
| native>plbart|none|adapter64|ep3|q0|csn_python | 5 | 36.22 ± 0.24 [35.93, 36.52] | 32.4 | 0.0 | 0.0 | 100.0 | – | 1.2 | 0.489 | 4.3 |
| codet5>plbart|naive|full|ep3|q0|csn_python | 5 | 13.60 ± 0.05 [13.54, 13.67] | 14.9 | 0.0 | 0.0 | 99.9 | – | 95.9 | 0.757 | 7.7 |
| codet5>plbart|linear|full|ep3|q0|csn_python | 5 | 33.45 ± 0.22 [33.17, 33.72] | 29.7 | 0.0 | 0.1 | 100.0 | – | 96.5 | 0.523 | 8.1 |
| codet5>plbart|resampler|full|ep3|q32|csn_python | 5 | 26.22 ± 0.29 [25.85, 26.58] | 23.9 | 0.0 | 0.1 | 100.0 | – | 110.7 | 0.580 | 7.9 |
| codet5>plbart|calm|full|ep3|q0|csn_python | 5 | 30.15 ± 0.29 [29.79, 30.51] | 27.1 | 0.0 | 0.1 | 100.0 | – | 103.6 | 0.560 | 8.2 |
| codet5>plbart|linear|lora|ep3|q0|csn_python | 5 | 32.58 ± 0.18 [32.36, 32.81] | 29.1 | 0.0 | 0.0 | 100.0 | – | 1.2 | 0.534 | 6.1 |
| plbart>plbart|naive|full|ep3|q0|csn_python | 5 | 36.80 ± 0.30 [36.43, 37.17] | 32.5 | 0.0 | 0.0 | 100.0 | – | 95.9 | 0.492 | 6.5 |
| plbart>plbart|linear|full|ep3|q0|csn_python | 5 | 36.78 ± 0.21 [36.52, 37.04] | 32.4 | 0.0 | 0.1 | 100.0 | – | 96.5 | 0.501 | 6.4 |
| codet5>codegpt|naive|full|ep3|q0|csn_python | 5 | 7.47 ± 0.06 [7.40, 7.55] | 12.5 | 0.0 | 63.6 | 23.3 | – | 124.4 | 0.532 | 12.5 |
| codet5>codegpt|linear|full|ep3|q0|csn_python | 5 | 22.25 ± 7.71 [12.68, 31.83] | 21.9 | 0.0 | 55.0 | 99.7 | – | 125.0 | 0.365 | 12.0 |
| native>codegpt|none|full|ep3|q0|csn_python | 5 | 33.34 ± 0.08 [33.25, 33.44] | 30.3 | 0.0 | 64.4 | 100.0 | – | 124.4 | 0.297 | 12.2 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CodeT5→PLBART naive | 5 | +19.84 (sd 0.23) [+19.56, +20.12] (5/5 seeds > 0) | +19.84 [+18.77, +20.86] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − CodeT5→PLBART resampler | 5 | +7.23 (sd 0.48) [+6.63, +7.82] (5/5 seeds > 0) | +7.23 [+6.51, +7.94] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − native PLBART | 5 | -3.99 (sd 0.28) [-4.34, -3.65] (0/5 seeds > 0) | -3.99 [-4.50, -3.51] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − native PLBART+LoRA | 5 | -2.29 (sd 0.21) [-2.55, -2.03] (0/5 seeds > 0) | -2.29 [-2.82, -1.76] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − PLBART→PLBART naive | 5 | -3.35 (sd 0.27) [-3.69, -3.01] (0/5 seeds > 0) | -3.35 [-3.85, -2.85] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive | 5 | +14.78 (sd 7.66) [+5.27, +24.29] (5/5 seeds > 0) | +14.78 [+14.11, +15.49] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0125 |
| CodeT5→CodeGPT linear − native CodeGPT | 5 | -11.09 (sd 7.70) [-20.65, -1.53] (0/5 seeds > 0) | -11.09 [-11.80, -10.35] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0322 |

## 5. Additional dataset: py_alpaca

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native>plbart|none|full|ep3|q0|py_alpaca | 5 | 22.20 ± 0.54 [21.54, 22.87] | 18.5 | 0.2 | 13.8 | 98.6 | – | 139.2 | 0.444 | 1.4 |
| native>plbart|none|lora|ep3|q0|py_alpaca | 5 | 15.24 ± 0.52 [14.60, 15.88] | 15.2 | 0.1 | 12.1 | 98.8 | – | 0.9 | 0.511 | 0.9 |
| native>plbart|none|adapter64|ep3|q0|py_alpaca | 5 | 17.31 ± 0.55 [16.63, 17.99] | 15.3 | 0.1 | 13.8 | 99.5 | – | 1.2 | 0.490 | 0.8 |
| codet5>plbart|naive|full|ep3|q0|py_alpaca | 5 | 9.04 ± 0.41 [8.53, 9.56] | 13.1 | 0.0 | 3.2 | 56.3 | – | 95.9 | 0.567 | 1.5 |
| codet5>plbart|linear|full|ep3|q0|py_alpaca | 5 | 15.71 ± 0.27 [15.38, 16.05] | 15.2 | 0.0 | 12.7 | 97.6 | – | 96.5 | 0.547 | 1.5 |
| codet5>plbart|resampler|full|ep3|q32|py_alpaca | 5 | 15.02 ± 0.45 [14.46, 15.58] | 15.0 | 0.0 | 12.7 | 92.9 | – | 110.7 | 0.541 | 1.6 |
| codet5>plbart|calm|full|ep3|q0|py_alpaca | 5 | 16.81 ± 0.33 [16.41, 17.21] | 15.7 | 0.0 | 13.1 | 93.4 | – | 103.6 | 0.532 | 1.5 |
| codet5>plbart|linear|lora|ep3|q0|py_alpaca | 5 | 11.29 ± 0.34 [10.87, 11.71] | 12.4 | 0.0 | 16.1 | 98.4 | – | 1.2 | 0.600 | 1.2 |
| plbart>plbart|naive|full|ep3|q0|py_alpaca | 5 | 20.86 ± 0.25 [20.56, 21.17] | 17.7 | 0.4 | 13.5 | 99.1 | – | 95.9 | 0.457 | 1.3 |
| plbart>plbart|linear|full|ep3|q0|py_alpaca | 5 | 15.38 ± 0.65 [14.57, 16.18] | 15.0 | 0.0 | 11.8 | 97.0 | – | 96.5 | 0.543 | 1.3 |
| codet5>codegpt|naive|full|ep3|q0|py_alpaca | 5 | 3.97 ± 1.46 [2.16, 5.78] | 11.7 | 0.0 | 75.0 | 9.9 | – | 124.4 | 0.538 | 2.4 |
| codet5>codegpt|linear|full|ep3|q0|py_alpaca | 5 | 8.53 ± 0.99 [7.31, 9.76] | 16.2 | 0.0 | 83.3 | 40.5 | – | 125.0 | 0.522 | 2.3 |
| native>codegpt|none|full|ep3|q0|py_alpaca | 5 | 17.08 ± 0.17 [16.87, 17.29] | 19.8 | 0.2 | 67.2 | 99.4 | – | 124.4 | 0.442 | 2.1 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CodeT5→PLBART naive | 5 | +6.67 (sd 0.50) [+6.05, +7.30] (5/5 seeds > 0) | +6.67 [+5.74, +7.61] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − CodeT5→PLBART resampler | 5 | +0.70 (sd 0.64) [-0.09, +1.49] (4/5 seeds > 0) | +0.70 [+0.06, +1.31] | +0.0 [+0.0, +0.0] | – | 0.0282 | 0.0708 |
| CodeT5→PLBART linear − native PLBART | 5 | -6.49 (sd 0.64) [-7.28, -5.70] (0/5 seeds > 0) | -6.49 [-7.69, -5.42] | -0.2 [-0.6, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − native PLBART+LoRA | 5 | +0.47 (sd 0.33) [+0.06, +0.89] (5/5 seeds > 0) | +0.47 [-0.75, +1.68] | -0.1 [-0.2, +0.0] | – | 0.4491 | 0.0337 |
| CodeT5→PLBART linear − PLBART→PLBART naive | 5 | -5.15 (sd 0.40) [-5.64, -4.66] (0/5 seeds > 0) | -5.15 [-6.27, -4.07] | -0.4 [-1.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive | 5 | +4.56 (sd 1.39) [+2.84, +6.29] (5/5 seeds > 0) | +4.56 [+3.72, +5.50] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0018 |
| CodeT5→CodeGPT linear − native CodeGPT | 5 | -8.55 (sd 1.09) [-9.90, -7.20] (0/5 seeds > 0) | -8.55 [-9.94, -7.31] | -0.2 [-0.5, +0.0] | – | 0.0002 | 0.0001 |

## 8. Dimension-mismatched bridging (hidden sizes 512 ↔ 768; 'padtrunc' = non-learned zero-pad/truncate)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5_small>plbart|padtrunc|full|ep3|q0|concode_java | 5 | 18.18 ± 0.36 [17.73, 18.64] | 18.5 | 0.6 | 97.7 | 52.2 | – | 95.9 | 0.587 | 7.0 |
| codet5_small>plbart|linear|full|ep3|q0|concode_java | 5 | 24.13 ± 0.26 [23.81, 24.46] | 24.0 | 8.6 | 97.0 | 92.5 | – | 96.3 | 0.451 | 7.1 |
| codet5_small>plbart|resampler|full|ep3|q32|concode_java | 5 | 21.63 ± 0.67 [20.79, 22.46] | 22.0 | 4.8 | 97.2 | 90.3 | – | 110.5 | 0.519 | 7.0 |
| codet5>codet5_small|padtrunc|full|ep3|q0|concode_java | 5 | 11.75 ± 0.47 [11.16, 12.33] | 13.7 | 0.3 | 97.4 | 24.4 | – | 41.6 | 0.660 | 13.6 |
| codet5>codet5_small|linear|full|ep3|q0|concode_java | 5 | 23.20 ± 0.70 [22.33, 24.06] | 24.1 | 11.3 | 95.2 | 90.9 | – | 42.0 | 0.477 | 14.2 |
| codet5>codet5_small|resampler|full|ep3|q32|concode_java | 5 | 22.35 ± 0.99 [21.12, 23.58] | 23.7 | 8.7 | 94.1 | 90.4 | – | 48.3 | 0.532 | 13.6 |
| unixcoder>codet5_small|padtrunc|full|ep3|q0|concode_java | 5 | 8.58 ± 0.27 [8.24, 8.92] | 9.5 | 0.0 | 89.8 | 21.3 | – | 41.6 | 1.171 | 12.4 |
| unixcoder>codet5_small|linear|full|ep3|q0|concode_java | 5 | 13.25 ± 0.38 [12.78, 13.72] | 14.6 | 0.5 | 94.6 | 70.9 | – | 42.0 | 0.633 | 12.1 |
| unixcoder>codet5_small|resampler|full|ep3|q32|concode_java | 5 | 14.68 ± 0.49 [14.07, 15.29] | 15.7 | 0.8 | 94.8 | 72.3 | – | 48.3 | 0.619 | 12.7 |
| codet5_small>codet5_small|naive|full|ep3|q0|concode_java | 5 | 22.27 ± 0.36 [21.82, 22.72] | 23.8 | 16.6 | 97.5 | 92.5 | – | 41.6 | 0.412 | 7.1 |
| codet5_small>codet5_small|linear|full|ep3|q0|concode_java | 5 | 22.16 ± 0.20 [21.91, 22.41] | 22.9 | 13.3 | 94.5 | 89.4 | – | 41.9 | 0.470 | 7.8 |
| native>codet5_small|none|full|ep3|q0|concode_java | 5 | 23.61 ± 0.28 [23.26, 23.96] | 25.5 | 19.2 | 97.8 | 95.2 | – | 60.5 | 0.389 | 11.3 |
| native>plbart|none|full|ep3|q0|concode_java | 15 | 29.62 ± 0.33 [29.44, 29.81] | 27.5 | 20.2 | 99.2 | 96.4 | – | 139.2 | 0.338 | 9.2 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| codet5_small→plbart linear − padtrunc | 5 | +5.95 (sd 0.32) [+5.55, +6.35] (5/5 seeds > 0) | +5.95 [+5.19, +6.74] | +8.0 [+6.5, +9.6] | – | 0.0002 | 0.0000 |
| codet5_small→plbart linear − resampler | 5 | +2.51 (sd 0.68) [+1.67, +3.35] (5/5 seeds > 0) | +2.51 [+2.02, +3.02] | +3.8 [+2.8, +4.8] | – | 0.0002 | 0.0011 |
| codet5_small→plbart linear − native plbart | 5 | -5.72 (sd 0.33) [-6.12, -5.31] (0/5 seeds > 0) | -5.72 [-6.64, -4.78] | -11.6 [-13.5, -9.8] | – | 0.0002 | 0.0000 |
| codet5→codet5_small linear − padtrunc | 5 | +11.45 (sd 0.62) [+10.68, +12.22] (5/5 seeds > 0) | +11.45 [+10.49, +12.42] | +10.9 [+9.2, +12.7] | – | 0.0002 | 0.0000 |
| codet5→codet5_small linear − resampler | 5 | +0.85 (sd 0.86) [-0.21, +1.91] (5/5 seeds > 0) | +0.85 [+0.37, +1.30] | +2.6 [+1.6, +3.6] | – | 0.0006 | 0.0910 |
| codet5→codet5_small linear − native codet5_small | 5 | -0.41 (sd 0.76) [-1.36, +0.54] (1/5 seeds > 0) | -0.41 [-1.26, +0.43] | -7.9 [-9.5, -6.4] | – | 0.2943 | 0.2942 |
| unixcoder→codet5_small linear − padtrunc | 5 | +4.67 (sd 0.49) [+4.06, +5.28] (5/5 seeds > 0) | +4.67 [+3.94, +5.38] | +0.5 [+0.1, +0.9] | – | 0.0002 | 0.0000 |
| unixcoder→codet5_small linear − resampler | 5 | -1.43 (sd 0.35) [-1.87, -0.99] (0/5 seeds > 0) | -1.43 [-1.91, -0.96] | -0.3 [-0.7, +0.1] | – | 0.0002 | 0.0008 |
| unixcoder→codet5_small linear − native codet5_small | 5 | -10.36 (sd 0.51) [-10.99, -9.72] (0/5 seeds > 0) | -10.36 [-11.69, -9.07] | -18.7 [-21.1, -16.4] | – | 0.0002 | 0.0000 |

## 9. Execution-based functional correctness (MBPP test, pass@1; Python only)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| native>plbart|none|full|ep5|q0|mbpp_py | 5 | 23.01 ± 0.50 [22.39, 23.64] | 19.5 | 1.1 | 3.8 | 99.6 | 0.8 ± 0.3 (HumanEval 0.0) | 139.2 | 0.611 | 6.5 |
| native>plbart|none|lora|ep5|q0|mbpp_py | 5 | 13.88 ± 0.49 [13.27, 14.49] | 12.8 | 0.0 | 6.6 | 99.6 | 0.6 ± 0.3 (HumanEval 1.2) | 0.9 | 0.560 | 4.5 |
| native>plbart|none|adapter64|ep5|q0|mbpp_py | 5 | 15.76 ± 0.35 [15.32, 16.20] | 14.4 | 0.1 | 7.8 | 99.5 | 0.5 ± 0.2 (HumanEval 1.1) | 1.2 | 0.542 | 4.1 |
| native>codegpt|none|full|ep5|q0|mbpp_py | 5 | 16.70 ± 0.39 [16.22, 17.18] | 17.2 | 0.2 | 91.2 | 99.8 | 5.8 ± 0.6 (HumanEval 3.4) | 124.4 | 0.514 | 10.3 |
| codet5>plbart|naive|full|ep5|q0|mbpp_py | 5 | 13.70 ± 0.38 [13.22, 14.17] | 14.4 | 0.0 | 8.0 | 91.5 | 0.0 ± 0.0 (HumanEval 0.0) | 95.9 | 0.729 | 8.6 |
| codet5>plbart|linear|full|ep5|q0|mbpp_py | 5 | 18.54 ± 0.59 [17.81, 19.28] | 17.3 | 0.3 | 4.6 | 99.8 | 0.4 ± 0.3 (HumanEval 0.6) | 96.5 | 0.667 | 8.7 |
| codet5>plbart|resampler|full|ep5|q32|mbpp_py | 5 | 17.33 ± 0.37 [16.88, 17.79] | 16.9 | 0.2 | 6.7 | 99.5 | 0.2 ± 0.1 (HumanEval 0.7) | 110.7 | 0.717 | 8.7 |
| codet5>plbart|calm|full|ep5|q0|mbpp_py | 5 | 19.55 ± 0.55 [18.87, 20.23] | 17.9 | 0.2 | 5.0 | 99.6 | 0.3 ± 0.2 (HumanEval 0.7) | 103.6 | 0.731 | 8.7 |
| plbart>plbart|naive|full|ep5|q0|mbpp_py | 5 | 20.18 ± 0.42 [19.66, 20.70] | 17.5 | 0.7 | 5.0 | 99.6 | 0.6 ± 0.3 (HumanEval 0.2) | 95.9 | 0.594 | 6.1 |
| plbart>plbart|linear|full|ep5|q0|mbpp_py | 5 | 19.21 ± 0.59 [18.48, 19.93] | 17.3 | 0.6 | 5.7 | 99.7 | 0.7 ± 0.3 (HumanEval 1.3) | 96.5 | 0.642 | 6.2 |
| codet5>codegpt|naive|full|ep5|q0|mbpp_py | 5 | 6.70 ± 0.49 [6.09, 7.31] | 11.1 | 0.0 | 94.9 | 45.7 | 0.0 ± 0.0 (HumanEval 0.0) | 124.4 | 0.673 | 14.3 |
| codet5>codegpt|linear|full|ep5|q0|mbpp_py | 5 | 13.37 ± 0.27 [13.04, 13.69] | 15.4 | 0.1 | 90.8 | 99.4 | 1.4 ± 0.4 (HumanEval 1.3) | 125.0 | 0.583 | 14.3 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CodeT5→PLBART naive | 5 | +4.85 (sd 0.71) [+3.96, +5.73] (5/5 seeds > 0) | +4.85 [+3.68, +6.09] | +0.3 [+0.1, +0.6] | +0.4 [+0.1, +0.8] (HumanEval +0.6, p=0.508) | 0.0002 | 0.0001 |
| CodeT5→PLBART linear − CodeT5→PLBART resampler | 5 | +1.21 (sd 0.91) [+0.08, +2.33] (5/5 seeds > 0) | +1.21 [+0.41, +1.99] | +0.1 [-0.2, +0.3] | +0.2 [-0.0, +0.5] (HumanEval -0.1, p=1.000) | 0.0028 | 0.0407 |
| CodeT5→PLBART linear − native PLBART | 5 | -4.47 (sd 0.85) [-5.53, -3.42] (0/5 seeds > 0) | -4.47 [-5.44, -3.54] | -0.8 [-1.7, -0.1] | -0.4 [-1.0, +0.1] (HumanEval +0.6, p=0.508) | 0.0002 | 0.0003 |
| CodeT5→PLBART linear − native PLBART+LoRA | 5 | +4.66 (sd 0.73) [+3.76, +5.57] (5/5 seeds > 0) | +4.66 [+3.46, +5.96] | +0.3 [+0.1, +0.6] | -0.2 [-0.8, +0.4] (HumanEval -0.6, p=0.591) | 0.0002 | 0.0001 |
| CodeT5→PLBART linear − PLBART→PLBART linear | 5 | -0.67 (sd 0.59) [-1.40, +0.07] (1/5 seeds > 0) | -0.67 [-1.41, +0.14] | -0.3 [-0.7, +0.0] | -0.4 [-0.9, +0.1] (HumanEval -0.7, p=0.198) | 0.0896 | 0.0661 |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive | 5 | +6.66 (sd 0.40) [+6.16, +7.16] (5/5 seeds > 0) | +6.66 [+5.57, +7.90] | +0.1 [+0.0, +0.3] | +1.4 [+0.7, +2.1] (HumanEval +1.3, p=0.117) | 0.0002 | 0.0000 |
| CodeT5→CodeGPT linear − native CodeGPT | 5 | -3.34 (sd 0.42) [-3.86, -2.81] (0/5 seeds > 0) | -3.34 [-4.22, -2.47] | -0.1 [-0.4, +0.2] | -4.5 [-6.2, -2.9] (HumanEval -2.1, p=0.046) | 0.0002 | 0.0001 |

## 9b. Training until functional correctness (early stopping on validation pass@1; test and HumanEval scored once at the best epoch)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|linear|full|ep20|q0|mbpp_py | 5 | 15.73 ± 1.82 [13.47, 17.99] | 15.6 | 0.1 | 4.8 | 99.4 | 0.1 ± 0.2 (HumanEval 0.5) | 96.5 | 0.665 | 13.2 |
| native>plbart|none|full|ep20|q0|mbpp_py | 5 | 15.79 ± 2.16 [13.10, 18.48] | 14.2 | 0.1 | 8.5 | 99.6 | 0.5 ± 0.2 (HumanEval 0.5) | 139.2 | 0.564 | 8.3 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| codet5→plbart linear − native plbart (same protocol) | 5 | -0.06 (sd 2.63) [-3.33, +3.21] (2/5 seeds > 0) | -0.06 [-0.90, +0.66] | -0.0 [-0.3, +0.2] | -0.4 [-0.8, -0.1] (HumanEval +0.0, p=1.000) | 0.8662 | 0.9619 |

Stopping behaviour (means over seeds). Validation pass@1 is the selection criterion and is optimistically biased; the test column is the unbiased estimate. 'max_epochs' means the cap, not convergence, ended training.

| system | seeds | epochs run | best epoch | val pass@1 (selected) | test pass@1 | stop reasons |
|---|---|---|---|---|---|---|
| codet5>plbart|linear|full|ep20|q0|mbpp_py | 5 | 6.8 | 2.8 | 0.4 | 0.1 | patience 5 |
| native>plbart|none|full|ep20|q0|mbpp_py | 5 | 5.6 | 1.6 | 0.9 | 0.5 | patience 5 |

## 10. External composition and parameter-efficient baselines (concode_java)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|concode_java | 15 | 17.77 ± 0.38 [17.56, 17.98] | 18.4 | 0.8 | 98.7 | 60.3 | – | 95.9 | 0.577 | 13.4 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |
| codet5>plbart|resampler|full|ep3|q32|concode_java | 15 | 22.85 ± 0.47 [22.58, 23.11] | 23.2 | 6.3 | 96.9 | 93.1 | – | 110.7 | 0.501 | 13.5 |
| native>plbart|none|full|ep3|q0|concode_java | 15 | 29.62 ± 0.33 [29.44, 29.81] | 27.5 | 20.2 | 99.2 | 96.4 | – | 139.2 | 0.338 | 9.2 |
| native>plbart|none|lora|ep3|q0|concode_java | 5 | 26.03 ± 0.12 [25.88, 26.18] | 24.8 | 20.1 | 97.8 | 95.4 | – | 0.9 | 0.316 | 7.0 |
| plbart>plbart|naive|full|ep3|q0|concode_java | 5 | 25.70 ± 0.51 [25.06, 26.34] | 24.8 | 15.7 | 97.8 | 95.1 | – | 95.9 | 0.361 | 8.9 |
| codet5>plbart|calm|full|ep3|q0|concode_java | 5 | 24.04 ± 0.29 [23.68, 24.40] | 24.3 | 7.0 | 97.8 | 92.4 | – | 103.6 | 0.497 | 13.8 |
| codet5>plbart|calm|frozen|ep3|q0|concode_java | 5 | 23.00 ± 0.60 [22.26, 23.74] | 23.3 | 5.2 | 95.0 | 90.6 | – | 7.7 | 0.505 | 11.8 |
| codet5>codegpt|naive|full|ep3|q0|concode_java | 15 | 12.93 ± 0.65 [12.57, 13.30] | 14.1 | 0.2 | 97.6 | 17.8 | – | 124.4 | 0.572 | 23.4 |
| codet5>codegpt|linear|full|ep3|q0|concode_java | 15 | 23.37 ± 1.34 [22.63, 24.11] | 24.9 | 9.2 | 93.3 | 91.3 | – | 125.0 | 0.412 | 23.5 |
| codet5>codegpt|calm|full|ep3|q0|concode_java | 5 | 20.78 ± 0.39 [20.30, 21.26] | 22.6 | 0.8 | 91.0 | 70.7 | – | 139.2 | 0.534 | 16.3 |
| native>codegpt|none|full|ep3|q0|concode_java | 15 | 25.73 ± 0.39 [25.52, 25.95] | 26.5 | 14.7 | 93.8 | 95.1 | – | 124.4 | 0.348 | 16.8 |
| native>plbart|none|lora4|ep3|q0|concode_java | 5 | 24.28 ± 0.94 [23.12, 25.44] | 23.4 | 18.7 | 96.6 | 95.2 | – | 0.2 | 0.327 | 7.0 |
| native>plbart|none|lora64|ep3|q0|concode_java | 5 | 27.18 ± 0.61 [26.42, 27.94] | 25.8 | 20.0 | 97.9 | 96.2 | – | 3.5 | 0.314 | 7.1 |
| native>plbart|none|lora256|ep3|q0|concode_java | 5 | 27.95 ± 0.65 [27.14, 28.76] | 26.3 | 19.9 | 98.6 | 96.2 | – | 14.2 | 0.326 | 7.3 |
| native>plbart|none|adapter16|ep3|q0|concode_java | 5 | 25.75 ± 0.62 [24.99, 26.52] | 24.3 | 19.3 | 96.4 | 95.8 | – | 0.3 | 0.317 | 6.3 |
| native>plbart|none|adapter64|ep3|q0|concode_java | 5 | 26.26 ± 0.40 [25.75, 26.76] | 25.0 | 19.9 | 97.2 | 95.3 | – | 1.2 | 0.310 | 6.3 |
| native>plbart|none|adapter256|ep3|q0|concode_java | 5 | 27.03 ± 0.16 [26.84, 27.23] | 25.7 | 20.0 | 98.7 | 95.7 | – | 4.7 | 0.315 | 6.4 |
| codet5>plbart|linear|adapter64|ep3|q0|concode_java | 5 | 25.65 ± 0.35 [25.22, 26.08] | 25.2 | 11.8 | 95.4 | 93.9 | – | 1.2 | 0.399 | 12.1 |
| codet5p>plbart|naive|full|ep3|q0|concode_java | 5 | 18.13 ± 0.29 [17.77, 18.50] | 18.6 | 0.6 | 98.0 | 57.8 | – | 95.9 | 0.587 | 13.7 |
| codet5p>plbart|linear|full|ep3|q0|concode_java | 5 | 25.70 ± 0.24 [25.41, 26.00] | 25.0 | 12.3 | 97.0 | 93.7 | – | 96.5 | 0.417 | 13.6 |
| native>codet5p|none|full|ep3|q0|concode_java | 5 | 34.12 ± 0.45 [33.56, 34.69] | 34.5 | 22.5 | 97.9 | 96.4 | – | 222.9 | 0.303 | 43.9 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) | 5 | +1.85 (sd 0.45) [+1.29, +2.41] (5/5 seeds > 0) | +1.85 [+1.37, +2.34] | +4.3 [+3.2, +5.5] | – | 0.0002 | 0.0008 |
| CALM-style frozen − linear frozen | 5 | -2.14 (sd 0.76) [-3.09, -1.20] (0/5 seeds > 0) | -2.14 [-2.64, -1.65] | -4.8 [-6.0, -3.7] | – | 0.0002 | 0.0032 |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT CALM-style | 5 | +2.76 (sd 1.25) [+1.22, +4.31] (5/5 seeds > 0) | +2.76 [+1.99, +3.72] | +8.7 [+7.2, +10.3] | – | 0.0002 | 0.0077 |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) | 5 | -0.15 (sd 0.60) [-0.89, +0.60] (3/5 seeds > 0) | -0.15 [-1.04, +0.77] | -8.8 [-10.6, -7.2] | – | 0.7275 | 0.6159 |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) | 5 | +0.19 (sd 0.47) [-0.40, +0.78] (3/5 seeds > 0) | +0.19 [-0.64, +0.98] | -4.4 [-5.9, -3.0] | – | 0.6481 | 0.4222 |
| CodeT5→PLBART linear − native PLBART + adapters (64) | 5 | -0.37 (sd 0.33) [-0.78, +0.04] (0/5 seeds > 0) | -0.37 [-1.20, +0.43] | -8.7 [-10.4, -7.1] | – | 0.3425 | 0.0663 |
| CodeT5→PLBART linear − native PLBART + LoRA r=4 | 5 | +1.61 (sd 0.80) [+0.62, +2.59] (5/5 seeds > 0) | +1.61 [+0.67, +2.55] | -7.4 [-9.1, -5.8] | – | 0.0004 | 0.0108 |
| CodeT5→PLBART linear − native PLBART + LoRA r=64 | 5 | -1.29 (sd 0.93) [-2.45, -0.13] (0/5 seeds > 0) | -1.29 [-2.15, -0.44] | -8.8 [-10.5, -7.2] | – | 0.0010 | 0.0362 |
| CodeT5→PLBART linear − native PLBART + LoRA r=256 | 5 | -2.07 (sd 0.87) [-3.15, -0.99] (0/5 seeds > 0) | -2.07 [-2.90, -1.26] | -8.6 [-10.3, -7.1] | – | 0.0002 | 0.0061 |
| bridge + adapter decoder − native PLBART + adapters (64) | 5 | -0.60 (sd 0.25) [-0.91, -0.30] (0/5 seeds > 0) | -0.60 [-1.47, +0.24] | -8.1 [-9.7, -6.6] | – | 0.1218 | 0.0055 |
| CodeT5+→PLBART linear − CodeT5→PLBART linear | 5 | -0.18 (sd 0.33) [-0.59, +0.23] (2/5 seeds > 0) | -0.18 [-0.71, +0.32] | +1.0 [+0.1, +2.0] | – | 0.4963 | 0.2859 |
| CodeT5+→PLBART linear − CodeT5+→PLBART naive | 5 | +7.57 (sd 0.45) [+7.02, +8.13] (5/5 seeds > 0) | +7.57 [+6.67, +8.51] | +11.7 [+9.9, +13.6] | – | 0.0002 | 0.0000 |
| CodeT5+→PLBART linear − native CodeT5+ (220M) | 5 | -8.42 (sd 0.33) [-8.82, -8.01] (0/5 seeds > 0) | -8.42 [-9.52, -7.39] | -10.2 [-12.0, -8.6] | – | 0.0002 | 0.0000 |

## 10. External composition and parameter-efficient baselines (csn_java)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|csn_java | 5 | 10.40 ± 0.14 [10.22, 10.58] | 16.8 | 0.0 | 93.8 | 97.3 | – | 95.9 | 0.607 | 9.9 |
| codet5>plbart|linear|full|ep3|q0|csn_java | 5 | 16.27 ± 0.43 [15.73, 16.81] | 20.0 | 0.0 | 93.3 | 99.8 | – | 96.5 | 0.529 | 9.8 |
| codet5>plbart|resampler|full|ep3|q32|csn_java | 5 | 15.01 ± 0.13 [14.85, 15.17] | 19.2 | 0.0 | 92.8 | 99.7 | – | 110.7 | 0.545 | 10.1 |
| native>plbart|none|full|ep3|q0|csn_java | 5 | 18.16 ± 0.24 [17.87, 18.46] | 21.2 | 0.0 | 94.9 | 99.9 | – | 139.2 | 0.491 | 7.8 |
| native>plbart|none|lora|ep3|q0|csn_java | 5 | 16.67 ± 0.23 [16.38, 16.95] | 20.1 | 0.0 | 93.8 | 99.8 | – | 0.9 | 0.494 | 5.4 |
| plbart>plbart|naive|full|ep3|q0|csn_java | 5 | 16.15 ± 0.12 [16.00, 16.30] | 19.8 | 0.0 | 94.4 | 99.8 | – | 95.9 | 0.502 | 7.5 |
| codet5>plbart|calm|full|ep3|q0|csn_java | 5 | 14.77 ± 0.19 [14.53, 15.00] | 19.0 | 0.0 | 92.2 | 99.9 | – | 103.6 | 0.560 | 9.8 |
| codet5>codegpt|naive|full|ep3|q0|csn_java | 5 | 4.23 ± 0.51 [3.59, 4.87] | 12.7 | 0.0 | 59.7 | 6.1 | – | 124.4 | 0.418 | 16.1 |
| codet5>codegpt|linear|full|ep3|q0|csn_java | 5 | 5.86 ± 2.16 [3.18, 8.55] | 13.7 | 0.0 | 61.6 | 46.9 | – | 125.0 | 0.405 | 16.0 |
| native>codegpt|none|full|ep3|q0|csn_java | 5 | 15.32 ± 0.22 [15.04, 15.59] | 19.2 | 0.0 | 56.7 | 99.8 | – | 124.4 | 0.346 | 12.9 |
| native>plbart|none|adapter64|ep3|q0|csn_java | 5 | 16.75 ± 0.20 [16.50, 17.00] | 20.2 | 0.0 | 93.6 | 99.8 | – | 1.2 | 0.488 | 4.9 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) | 5 | +1.51 (sd 0.35) [+1.08, +1.94] (5/5 seeds > 0) | +1.51 [+1.16, +1.86] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0006 |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) | 5 | -0.39 (sd 0.41) [-0.90, +0.12] (0/5 seeds > 0) | -0.39 [-0.92, +0.16] | +0.0 [+0.0, +0.0] | – | 0.1536 | 0.0990 |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) | 5 | +0.12 (sd 0.43) [-0.41, +0.65] (3/5 seeds > 0) | +0.12 [-0.43, +0.67] | +0.0 [+0.0, +0.0] | – | 0.6649 | 0.5662 |
| CodeT5→PLBART linear − native PLBART + adapters (64) | 5 | -0.48 (sd 0.46) [-1.04, +0.09] (0/5 seeds > 0) | -0.48 [-1.00, +0.07] | +0.0 [+0.0, +0.0] | – | 0.0836 | 0.0796 |

## 10. External composition and parameter-efficient baselines (py_alpaca)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|py_alpaca | 5 | 9.04 ± 0.41 [8.53, 9.56] | 13.1 | 0.0 | 3.2 | 56.3 | – | 95.9 | 0.567 | 1.5 |
| codet5>plbart|linear|full|ep3|q0|py_alpaca | 5 | 15.71 ± 0.27 [15.38, 16.05] | 15.2 | 0.0 | 12.7 | 97.6 | – | 96.5 | 0.547 | 1.5 |
| codet5>plbart|resampler|full|ep3|q32|py_alpaca | 5 | 15.02 ± 0.45 [14.46, 15.58] | 15.0 | 0.0 | 12.7 | 92.9 | – | 110.7 | 0.541 | 1.6 |
| native>plbart|none|full|ep3|q0|py_alpaca | 5 | 22.20 ± 0.54 [21.54, 22.87] | 18.5 | 0.2 | 13.8 | 98.6 | – | 139.2 | 0.444 | 1.4 |
| native>plbart|none|lora|ep3|q0|py_alpaca | 5 | 15.24 ± 0.52 [14.60, 15.88] | 15.2 | 0.1 | 12.1 | 98.8 | – | 0.9 | 0.511 | 0.9 |
| plbart>plbart|naive|full|ep3|q0|py_alpaca | 5 | 20.86 ± 0.25 [20.56, 21.17] | 17.7 | 0.4 | 13.5 | 99.1 | – | 95.9 | 0.457 | 1.3 |
| codet5>plbart|calm|full|ep3|q0|py_alpaca | 5 | 16.81 ± 0.33 [16.41, 17.21] | 15.7 | 0.0 | 13.1 | 93.4 | – | 103.6 | 0.532 | 1.5 |
| codet5>codegpt|naive|full|ep3|q0|py_alpaca | 5 | 3.97 ± 1.46 [2.16, 5.78] | 11.7 | 0.0 | 75.0 | 9.9 | – | 124.4 | 0.538 | 2.4 |
| codet5>codegpt|linear|full|ep3|q0|py_alpaca | 5 | 8.53 ± 0.99 [7.31, 9.76] | 16.2 | 0.0 | 83.3 | 40.5 | – | 125.0 | 0.522 | 2.3 |
| native>codegpt|none|full|ep3|q0|py_alpaca | 5 | 17.08 ± 0.17 [16.87, 17.29] | 19.8 | 0.2 | 67.2 | 99.4 | – | 124.4 | 0.442 | 2.1 |
| native>plbart|none|adapter64|ep3|q0|py_alpaca | 5 | 17.31 ± 0.55 [16.63, 17.99] | 15.3 | 0.1 | 13.8 | 99.5 | – | 1.2 | 0.490 | 0.8 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) | 5 | -1.10 (sd 0.49) [-1.70, -0.49] (0/5 seeds > 0) | -1.10 [-1.77, -0.44] | +0.0 [+0.0, +0.0] | – | 0.0008 | 0.0073 |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) | 5 | +0.47 (sd 0.33) [+0.06, +0.89] (5/5 seeds > 0) | +0.47 [-0.75, +1.68] | -0.1 [-0.2, +0.0] | – | 0.4491 | 0.0337 |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) | 5 | -5.15 (sd 0.40) [-5.64, -4.66] (0/5 seeds > 0) | -5.15 [-6.27, -4.07] | -0.4 [-1.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − native PLBART + adapters (64) | 5 | -1.59 (sd 0.58) [-2.31, -0.88] (0/5 seeds > 0) | -1.59 [-2.89, -0.34] | -0.1 [-0.2, +0.0] | – | 0.0166 | 0.0035 |

## 10. External composition and parameter-efficient baselines (csn_python)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|csn_python | 5 | 13.60 ± 0.05 [13.54, 13.67] | 14.9 | 0.0 | 0.0 | 99.9 | – | 95.9 | 0.757 | 7.7 |
| codet5>plbart|linear|full|ep3|q0|csn_python | 5 | 33.45 ± 0.22 [33.17, 33.72] | 29.7 | 0.0 | 0.1 | 100.0 | – | 96.5 | 0.523 | 8.1 |
| codet5>plbart|resampler|full|ep3|q32|csn_python | 5 | 26.22 ± 0.29 [25.85, 26.58] | 23.9 | 0.0 | 0.1 | 100.0 | – | 110.7 | 0.580 | 7.9 |
| native>plbart|none|full|ep3|q0|csn_python | 5 | 37.44 ± 0.28 [37.09, 37.79] | 33.0 | 0.0 | 0.1 | 100.0 | – | 139.2 | 0.487 | 6.9 |
| native>plbart|none|lora|ep3|q0|csn_python | 5 | 35.74 ± 0.05 [35.67, 35.80] | 32.0 | 0.0 | 0.0 | 100.0 | – | 0.9 | 0.495 | 4.6 |
| plbart>plbart|naive|full|ep3|q0|csn_python | 5 | 36.80 ± 0.30 [36.43, 37.17] | 32.5 | 0.0 | 0.0 | 100.0 | – | 95.9 | 0.492 | 6.5 |
| codet5>plbart|calm|full|ep3|q0|csn_python | 5 | 30.15 ± 0.29 [29.79, 30.51] | 27.1 | 0.0 | 0.1 | 100.0 | – | 103.6 | 0.560 | 8.2 |
| codet5>codegpt|naive|full|ep3|q0|csn_python | 5 | 7.47 ± 0.06 [7.40, 7.55] | 12.5 | 0.0 | 63.6 | 23.3 | – | 124.4 | 0.532 | 12.5 |
| codet5>codegpt|linear|full|ep3|q0|csn_python | 5 | 22.25 ± 7.71 [12.68, 31.83] | 21.9 | 0.0 | 55.0 | 99.7 | – | 125.0 | 0.365 | 12.0 |
| native>codegpt|none|full|ep3|q0|csn_python | 5 | 33.34 ± 0.08 [33.25, 33.44] | 30.3 | 0.0 | 64.4 | 100.0 | – | 124.4 | 0.297 | 12.2 |
| native>plbart|none|adapter64|ep3|q0|csn_python | 5 | 36.22 ± 0.24 [35.93, 36.52] | 32.4 | 0.0 | 0.0 | 100.0 | – | 1.2 | 0.489 | 4.3 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) | 5 | +3.29 (sd 0.42) [+2.78, +3.81] (5/5 seeds > 0) | +3.29 [+2.88, +3.71] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0001 |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) | 5 | -2.29 (sd 0.21) [-2.55, -2.03] (0/5 seeds > 0) | -2.29 [-2.82, -1.76] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) | 5 | -3.35 (sd 0.27) [-3.69, -3.01] (0/5 seeds > 0) | -3.35 [-3.85, -2.85] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |
| CodeT5→PLBART linear − native PLBART + adapters (64) | 5 | -2.78 (sd 0.15) [-2.96, -2.60] (0/5 seeds > 0) | -2.78 [-3.29, -2.29] | +0.0 [+0.0, +0.0] | – | 0.0002 | 0.0000 |

## 11. Larger components and mismatched hidden sizes (1024 ↔ 768; 'padtrunc' = non-learned reference)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5p770>plbart|padtrunc|full|ep3|q0|concode_java | 3 | 17.05 ± 0.08 [16.86, 17.24] | 17.4 | 0.6 | 98.1 | 34.8 | – | 95.9 | 0.596 | 29.4 |
| codet5p770>plbart|linear|full|ep3|q0|concode_java | 3 | 26.84 ± 0.15 [26.47, 27.22] | 26.0 | 12.5 | 96.8 | 95.3 | – | 96.7 | 0.414 | 30.0 |
| codet5p770>plbart|resampler|full|ep3|q32|concode_java | 3 | 22.82 ± 0.17 [22.40, 23.24] | 23.1 | 7.0 | 96.6 | 92.0 | – | 110.9 | 0.493 | 29.4 |
| codet5p770>plbart|calm|full|ep3|q0|concode_java | 3 | 24.31 ± 0.46 [23.17, 25.44] | 24.5 | 7.2 | 96.9 | 92.4 | – | 103.8 | 0.498 | 29.9 |
| codet5>gpt2m|padtrunc|full|ep3|q0|concode_java | 3 | 13.94 ± 0.59 [12.47, 15.40] | 15.2 | 0.2 | 96.2 | 17.4 | – | 354.8 | 0.645 | 51.8 |
| codet5>gpt2m|linear|full|ep3|q0|concode_java | 3 | 23.76 ± 0.37 [22.84, 24.67] | 25.0 | 11.2 | 92.8 | 93.5 | – | 355.6 | 0.452 | 51.0 |
| codet5>gpt2m|calm|full|ep3|q0|concode_java | 3 | 21.05 ± 0.50 [19.81, 22.30] | 22.8 | 1.5 | 94.8 | 83.3 | – | 406.0 | 0.591 | 27.7 |
| native>gpt2m|none|full|ep3|q0|concode_java | 3 | 24.83 ± 0.24 [24.24, 25.42] | 26.0 | 15.8 | 96.4 | 94.9 | – | 354.8 | 0.402 | 45.6 |
| native>plbart|none|full|ep3|q0|concode_java | 15 | 29.62 ± 0.33 [29.44, 29.81] | 27.5 | 20.2 | 99.2 | 96.4 | – | 139.2 | 0.338 | 9.2 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 25.65 ± 0.49 [25.37, 25.92] | 25.3 | 10.9 | 97.5 | 94.3 | – | 96.5 | 0.427 | 13.7 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| CodeT5+770M→PLBART linear − padtrunc | 3 | +9.80 (sd 0.08) [+9.61, +9.98] (3/3 seeds > 0) | +9.80 [+8.78, +10.88] | +11.9 [+10.1, +13.8] | – | 0.0002 | 0.0000 |
| CodeT5+770M→PLBART linear − resampler | 3 | +4.02 (sd 0.09) [+3.81, +4.24] (3/3 seeds > 0) | +4.02 [+3.34, +4.74] | +5.5 [+4.3, +6.8] | – | 0.0002 | 0.0002 |
| CodeT5+770M→PLBART linear − CALM-style | 3 | +2.54 (sd 0.39) [+1.58, +3.50] (3/3 seeds > 0) | +2.54 [+1.94, +3.18] | +5.3 [+4.1, +6.6] | – | 0.0002 | 0.0076 |
| CodeT5+770M→PLBART linear − CodeT5→PLBART linear (smaller encoder) | 3 | +0.96 (sd 0.40) [-0.03, +1.95] (3/3 seeds > 0) | +0.96 [+0.30, +1.61] | +0.9 [-0.1, +2.0] | – | 0.0040 | 0.0530 |
| CodeT5+770M→PLBART linear − native PLBART | 3 | -2.94 (sd 0.39) [-3.91, -1.97] (0/3 seeds > 0) | -2.94 [-3.82, -2.05] | -7.5 [-9.1, -6.0] | – | 0.0002 | 0.0058 |
| CodeT5→GPT-2 medium linear − padtrunc | 3 | +9.82 (sd 0.76) [+7.92, +11.72] (3/3 seeds > 0) | +9.82 [+8.96, +10.65] | +11.0 [+9.3, +12.8] | – | 0.0002 | 0.0020 |
| CodeT5→GPT-2 medium linear − CALM-style | 3 | +2.70 (sd 0.14) [+2.36, +3.04] (3/3 seeds > 0) | +2.70 [+1.97, +3.41] | +9.7 [+8.1, +11.4] | – | 0.0002 | 0.0008 |
| CodeT5→GPT-2 medium linear − native GPT-2 medium | 3 | -1.07 (sd 0.54) [-2.41, +0.26] (0/3 seeds > 0) | -1.07 [-1.80, -0.35] | -4.5 [-6.0, -3.2] | – | 0.0018 | 0.0737 |

## 12. Four times more training data (40,000 CONCODE examples; same validation/test subsets)

| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|concode_java_40k | 3 | 25.11 ± 0.26 [24.46, 25.77] | 25.1 | 7.8 | 96.6 | 91.8 | – | 95.9 | 0.472 | 55.2 |
| codet5>plbart|linear|full|ep3|q0|concode_java_40k | 3 | 29.72 ± 0.21 [29.21, 30.24] | 28.0 | 16.6 | 97.7 | 95.7 | – | 96.5 | 0.368 | 51.9 |
| codet5>plbart|resampler|full|ep3|q32|concode_java_40k | 3 | 25.35 ± 0.23 [24.78, 25.92] | 25.2 | 10.9 | 96.7 | 94.1 | – | 110.7 | 0.433 | 52.7 |
| native>plbart|none|full|ep3|q0|concode_java_40k | 3 | 32.64 ± 0.60 [31.15, 34.14] | 29.7 | 21.2 | 99.2 | 96.6 | – | 139.2 | 0.317 | 36.4 |
| native>plbart|none|lora|ep3|q0|concode_java_40k | 3 | 28.31 ± 0.25 [27.69, 28.93] | 26.7 | 20.9 | 98.3 | 96.0 | – | 0.9 | 0.300 | 27.9 |
| codet5>plbart|linear|lora|ep3|q0|concode_java_40k | 3 | 28.06 ± 0.21 [27.54, 28.58] | 26.7 | 16.2 | 97.4 | 94.7 | – | 1.2 | 0.350 | 45.6 |

| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |
|---|---|---|---|---|---|---|---|
| linear − naive | 3 | +4.61 (sd 0.32) [+3.81, +5.42] (3/3 seeds > 0) | +4.61 [+3.78, +5.47] | +8.7 [+7.1, +10.5] | – | 0.0002 | 0.0016 |
| linear − resampler | 3 | +4.38 (sd 0.05) [+4.26, +4.49] (3/3 seeds > 0) | +4.38 [+3.70, +5.08] | +5.7 [+4.4, +7.0] | – | 0.0002 | 0.0000 |
| linear − native PLBART | 3 | -2.92 (sd 0.50) [-4.15, -1.68] (0/3 seeds > 0) | -2.92 [-3.90, -1.94] | -4.7 [-6.2, -3.3] | – | 0.0002 | 0.0095 |
| linear − native PLBART + LoRA | 3 | +1.41 (sd 0.45) [+0.31, +2.52] (3/3 seeds > 0) | +1.41 [+0.39, +2.47] | -4.3 [-5.8, -2.9] | – | 0.0048 | 0.0315 |

## 13. Multiple-comparison control: pre-specified primary family (m = 6)

Family fixed in `primary_pairs()` before the extension runs. Holm-Bonferroni across the family; 'p perm' = paired approximate-randomisation test over test examples, 'p seed-t' = paired t-test over seeds (few seeds: low power). 'robust' = both Holm-adjusted p < 0.05. 'equivalent' = the whole bootstrap CI lies within ±1 BLEU.

| comparison | n seeds | ΔBLEU (seed mean) [bootstrap 95% CI] | dz | p perm | Holm p | BH q | p seed-t | Holm p (seeds) | verdict | equivalence |
|---|---|---|---|---|---|---|---|---|---|---|
| linear − naive (codet5→plbart; concode_java) | 15 | +7.88 [+7.05, +8.70] | +17.0 | 0.0002 | 0.0012 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| linear − naive (codet5→codegpt; concode_java) | 15 | +10.44 [+9.62, +11.30] | +6.5 | 0.0002 | 0.0012 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| linear ep3 − resampler ep3 (codet5→plbart; concode_java) | 15 | +2.80 [+2.39, +3.22] | +4.7 | 0.0002 | 0.0012 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) | 5 | -0.15 [-1.04, +0.77] | -0.2 | 0.7275 | 1.0000 | 0.7275 | 0.6159 | 0.8443 | not significant | - |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) | 5 | +0.19 [-0.64, +0.98] | +0.4 | 0.6481 | 1.0000 | 0.7275 | 0.4222 | 0.8443 | not significant | equivalent (±1.0 BLEU) |
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) | 5 | +1.85 [+1.37, +2.34] | +4.1 | 0.0002 | 0.0012 | 0.0003 | 0.0008 | 0.0023 | robust (test set + seeds) | - |

## 14. All other paired comparisons (exploratory family, m = 81)

Holm / BH across every remaining comparison printed above (all datasets and stages together). Read as descriptive.

| comparison | n seeds | ΔBLEU (seed mean) [bootstrap 95% CI] | dz | p perm | Holm p | BH q | p seed-t | Holm p (seeds) | verdict | equivalence |
|---|---|---|---|---|---|---|---|---|---|---|
| linear − native (codet5→plbart; concode_java) | 15 | -3.98 [-4.76, -3.19] | -8.6 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| resampler − naive (codet5→plbart; concode_java) | 15 | +5.08 [+4.34, +5.83] | +9.7 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| linear − native CodeGPT (codet5→codegpt; concode_java) | 15 | -2.36 [-3.08, -1.65] | -1.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0020 | robust (test set + seeds) | - |
| unixcoder→PLBART linear − unixcoder→PLBART naive | 5 | +6.02 [+5.32, +6.72] | +12.4 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0006 | robust (test set + seeds) | - |
| unixcoder→PLBART linear − codet5→PLBART linear | 5 | -9.53 [-10.54, -8.60] | -19.4 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| roberta→PLBART linear − roberta→PLBART naive | 5 | +9.33 [+8.44, +10.25] | +7.5 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0036 | robust (test set + seeds) | - |
| roberta→PLBART linear − codet5→PLBART linear | 5 | -5.45 [-6.19, -4.75] | -6.4 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0065 | robust (test set + seeds) | - |
| codebert→PLBART linear − codebert→PLBART naive | 5 | +3.59 [+2.87, +4.32] | +4.9 | 0.0002 | 0.0162 | 0.0003 | 0.0004 | 0.0169 | robust (test set + seeds) | - |
| codebert→PLBART linear − codet5→PLBART linear | 5 | -7.75 [-8.58, -6.97] | -12.9 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0005 | robust (test set + seeds) | - |
| unixcoder→CodeGPT linear − unixcoder→CodeGPT naive | 5 | +3.05 [+2.37, +3.73] | +13.8 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0004 | robust (test set + seeds) | - |
| linear frozen − linear full-tune (codet5→plbart; concode_java) | 5 | -0.74 [-1.23, -0.27] | -1.6 | 0.0018 | 0.0306 | 0.0022 | 0.0254 | 0.4823 | test-set only | - |
| linear frozen − resampler frozen (codet5→plbart; concode_java) | 5 | +3.06 [+2.55, +3.61] | +19.0 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| CodeGPT linear frozen − CodeGPT linear full (codet5→codegpt; concode_java) | 5 | -8.95 [-10.28, -7.55] | -4.9 | 0.0002 | 0.0162 | 0.0003 | 0.0004 | 0.0169 | robust (test set + seeds) | - |
| linear ep6 − resampler ep6 (codet5→plbart; concode_java) | 5 | +3.08 [+2.49, +3.66] | +9.7 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0015 | robust (test set + seeds) | - |
| linear ep12 − resampler ep12 (codet5→plbart; concode_java) | 5 | +2.86 [+2.23, +3.51] | +4.7 | 0.0002 | 0.0162 | 0.0003 | 0.0004 | 0.0183 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART naive [csn_java] | 5 | +5.87 [+5.29, +6.49] | +16.6 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0002 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART resampler [csn_java] | 5 | +1.26 [+0.96, +1.56] | +2.3 | 0.0002 | 0.0162 | 0.0003 | 0.0063 | 0.1644 | test-set only | - |
| CodeT5→PLBART linear − native PLBART [csn_java] | 5 | -1.89 [-2.35, -1.43] | -3.7 | 0.0002 | 0.0162 | 0.0003 | 0.0012 | 0.0420 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) [csn_java] | 5 | -0.39 [-0.92, +0.16] | -1.0 | 0.1536 | 1.0000 | 0.1659 | 0.0990 | 0.7932 | not significant | equivalent (±1.0 BLEU) |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) [csn_java] | 5 | +0.12 [-0.43, +0.67] | +0.3 | 0.6649 | 1.0000 | 0.6732 | 0.5662 | 1.0000 | not significant | equivalent (±1.0 BLEU) |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive [csn_java] | 5 | +1.64 [+1.40, +1.87] | +1.0 | 0.0002 | 0.0162 | 0.0003 | 0.0927 | 0.7932 | test-set only | - |
| CodeT5→CodeGPT linear − native CodeGPT [csn_java] | 5 | -9.45 [-10.22, -8.69] | -4.4 | 0.0002 | 0.0162 | 0.0003 | 0.0006 | 0.0248 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART naive [csn_python] | 5 | +19.84 [+18.77, +20.86] | +87.9 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART resampler [csn_python] | 5 | +7.23 [+6.51, +7.94] | +15.1 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0003 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART [csn_python] | 5 | -3.99 [-4.50, -3.51] | -14.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0004 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) [csn_python] | 5 | -2.29 [-2.82, -1.76] | -11.0 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0010 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) [csn_python] | 5 | -3.35 [-3.85, -2.85] | -12.2 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0006 | robust (test set + seeds) | - |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive [csn_python] | 5 | +14.78 [+14.11, +15.49] | +1.9 | 0.0002 | 0.0162 | 0.0003 | 0.0125 | 0.2498 | test-set only | - |
| CodeT5→CodeGPT linear − native CodeGPT [csn_python] | 5 | -11.09 [-11.80, -10.35] | -1.4 | 0.0002 | 0.0162 | 0.0003 | 0.0322 | 0.5666 | test-set only | - |
| CodeT5→PLBART linear − CodeT5→PLBART naive [py_alpaca] | 5 | +6.67 [+5.74, +7.61] | +13.3 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0005 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART resampler [py_alpaca] | 5 | +0.70 [+0.06, +1.31] | +1.1 | 0.0282 | 0.3101 | 0.0322 | 0.0708 | 0.7932 | not significant | - |
| CodeT5→PLBART linear − native PLBART [py_alpaca] | 5 | -6.49 [-7.69, -5.42] | -10.2 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0012 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=16 (paper setting) [py_alpaca] | 5 | +0.47 [-0.75, +1.68] | +1.4 | 0.4491 | 1.0000 | 0.4664 | 0.0337 | 0.5666 | not significant | - |
| CodeT5→PLBART linear − PLBART→PLBART naive (same family) [py_alpaca] | 5 | -5.15 [-6.27, -4.07] | -13.0 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0005 | robust (test set + seeds) | - |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive [py_alpaca] | 5 | +4.56 [+3.72, +5.50] | +3.3 | 0.0002 | 0.0162 | 0.0003 | 0.0018 | 0.0601 | test-set only | - |
| CodeT5→CodeGPT linear − native CodeGPT [py_alpaca] | 5 | -8.55 [-9.94, -7.31] | -7.9 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0031 | robust (test set + seeds) | - |
| codet5_small→plbart linear − padtrunc | 5 | +5.95 [+5.19, +6.74] | +18.6 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| codet5_small→plbart linear − resampler | 5 | +2.51 [+2.02, +3.02] | +3.7 | 0.0002 | 0.0162 | 0.0003 | 0.0011 | 0.0413 | robust (test set + seeds) | - |
| codet5_small→plbart linear − native plbart | 5 | -5.72 [-6.64, -4.78] | -17.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0002 | robust (test set + seeds) | - |
| codet5→codet5_small linear − padtrunc | 5 | +11.45 [+10.49, +12.42] | +18.6 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| codet5→codet5_small linear − resampler | 5 | +0.85 [+0.37, +1.30] | +1.0 | 0.0006 | 0.0162 | 0.0008 | 0.0910 | 0.7932 | test-set only | - |
| codet5→codet5_small linear − native codet5_small | 5 | -0.41 [-1.26, +0.43] | -0.5 | 0.2943 | 1.0000 | 0.3137 | 0.2942 | 1.0000 | not significant | - |
| unixcoder→codet5_small linear − padtrunc | 5 | +4.67 [+3.94, +5.38] | +9.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0016 | robust (test set + seeds) | - |
| unixcoder→codet5_small linear − resampler | 5 | -1.43 [-1.91, -0.96] | -4.1 | 0.0002 | 0.0162 | 0.0003 | 0.0008 | 0.0312 | robust (test set + seeds) | - |
| unixcoder→codet5_small linear − native codet5_small | 5 | -10.36 [-11.69, -9.07] | -20.3 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART naive [mbpp_py] | 5 | +4.85 [+3.68, +6.09] | +6.8 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0052 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CodeT5→PLBART resampler [mbpp_py] | 5 | +1.21 [+0.41, +1.99] | +1.3 | 0.0028 | 0.0420 | 0.0034 | 0.0407 | 0.5700 | test-set only | - |
| CodeT5→PLBART linear − native PLBART [mbpp_py] | 5 | -4.47 [-5.44, -3.54] | -5.3 | 0.0002 | 0.0162 | 0.0003 | 0.0003 | 0.0131 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART+LoRA [mbpp_py] | 5 | +4.66 [+3.46, +5.96] | +6.4 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0065 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − PLBART→PLBART linear [mbpp_py] | 5 | -0.67 [-1.41, +0.14] | -1.1 | 0.0896 | 0.8358 | 0.0994 | 0.0661 | 0.7932 | not significant | - |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT naive [mbpp_py] | 5 | +6.66 [+5.57, +7.90] | +16.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0002 | robust (test set + seeds) | - |
| CodeT5→CodeGPT linear − native CodeGPT [mbpp_py] | 5 | -3.34 [-4.22, -2.47] | -7.9 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0031 | robust (test set + seeds) | - |
| codet5→plbart linear − native plbart (same protocol) [mbpp_py] | 5 | -0.06 [-0.90, +0.66] | -0.0 | 0.8662 | 1.0000 | 0.8662 | 0.9619 | 1.0000 | not significant | equivalent (±1.0 BLEU) |
| CALM-style frozen − linear frozen (codet5→plbart; concode_java) | 5 | -2.14 [-2.64, -1.65] | -2.8 | 0.0002 | 0.0162 | 0.0003 | 0.0032 | 0.0994 | test-set only | - |
| CodeT5→CodeGPT linear − CodeT5→CodeGPT CALM-style | 5 | +2.76 [+1.99, +3.72] | +2.2 | 0.0002 | 0.0162 | 0.0003 | 0.0077 | 0.1835 | test-set only | - |
| CodeT5→PLBART linear − native PLBART + adapters (64) | 5 | -0.37 [-1.20, +0.43] | -1.1 | 0.3425 | 1.0000 | 0.3603 | 0.0663 | 0.7932 | not significant | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=4 | 5 | +1.61 [+0.67, +2.55] | +2.0 | 0.0004 | 0.0162 | 0.0005 | 0.0108 | 0.2267 | test-set only | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=64 | 5 | -1.29 [-2.15, -0.44] | -1.4 | 0.0010 | 0.0180 | 0.0013 | 0.0362 | 0.5666 | test-set only | - |
| CodeT5→PLBART linear − native PLBART + LoRA r=256 | 5 | -2.07 [-2.90, -1.26] | -2.4 | 0.0002 | 0.0162 | 0.0003 | 0.0061 | 0.1635 | test-set only | - |
| bridge + adapter decoder − native PLBART + adapters (64) (codet5→plbart; concode_java) | 5 | -0.60 [-1.47, +0.24] | -2.4 | 0.1218 | 0.9742 | 0.1333 | 0.0055 | 0.1586 | not significant | - |
| CodeT5+→PLBART linear − CodeT5→PLBART linear | 5 | -0.18 [-0.71, +0.32] | -0.6 | 0.4963 | 1.0000 | 0.5089 | 0.2859 | 1.0000 | not significant | equivalent (±1.0 BLEU) |
| CodeT5+→PLBART linear − CodeT5+→PLBART naive | 5 | +7.57 [+6.67, +8.51] | +16.9 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0002 | robust (test set + seeds) | - |
| CodeT5+→PLBART linear − native CodeT5+ (220M) | 5 | -8.42 [-9.52, -7.39] | -25.9 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0000 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) [csn_java] | 5 | +1.51 [+1.16, +1.86] | +4.3 | 0.0002 | 0.0162 | 0.0003 | 0.0006 | 0.0248 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + adapters (64) [csn_java] | 5 | -0.48 [-1.00, +0.07] | -1.0 | 0.0836 | 0.8358 | 0.0940 | 0.0796 | 0.7932 | not significant | - |
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) [py_alpaca] | 5 | -1.10 [-1.77, -0.44] | -2.3 | 0.0008 | 0.0162 | 0.0010 | 0.0073 | 0.1816 | test-set only | - |
| CodeT5→PLBART linear − native PLBART + adapters (64) [py_alpaca] | 5 | -1.59 [-2.89, -0.34] | -2.8 | 0.0166 | 0.1992 | 0.0192 | 0.0035 | 0.1050 | not significant | - |
| CodeT5→PLBART linear − CALM-style cross-attention (decoder tuned) [csn_python] | 5 | +3.29 [+2.88, +3.71] | +7.9 | 0.0002 | 0.0162 | 0.0003 | 0.0001 | 0.0031 | robust (test set + seeds) | - |
| CodeT5→PLBART linear − native PLBART + adapters (64) [csn_python] | 5 | -2.78 [-3.29, -2.29] | -19.1 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0001 | robust (test set + seeds) | - |
| CodeT5+770M→PLBART linear − padtrunc | 3 | +9.80 [+8.78, +10.88] | +130.5 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0011 | robust (test set + seeds) | - |
| CodeT5+770M→PLBART linear − resampler | 3 | +4.02 [+3.34, +4.74] | +46.3 | 0.0002 | 0.0162 | 0.0003 | 0.0002 | 0.0070 | robust (test set + seeds) | - |
| CodeT5+770M→PLBART linear − CALM-style | 3 | +2.54 [+1.94, +3.18] | +6.6 | 0.0002 | 0.0162 | 0.0003 | 0.0076 | 0.1835 | test-set only | - |
| CodeT5+770M→PLBART linear − CodeT5→PLBART linear (smaller encoder) | 3 | +0.96 [+0.30, +1.61] | +2.4 | 0.0040 | 0.0560 | 0.0048 | 0.0530 | 0.6887 | not significant | - |
| CodeT5+770M→PLBART linear − native PLBART | 3 | -2.94 [-3.82, -2.05] | -7.5 | 0.0002 | 0.0162 | 0.0003 | 0.0058 | 0.1633 | test-set only | - |
| CodeT5→GPT-2 medium linear − padtrunc | 3 | +9.82 [+8.96, +10.65] | +12.9 | 0.0002 | 0.0162 | 0.0003 | 0.0020 | 0.0643 | test-set only | - |
| CodeT5→GPT-2 medium linear − CALM-style | 3 | +2.70 [+1.97, +3.41] | +19.8 | 0.0002 | 0.0162 | 0.0003 | 0.0008 | 0.0314 | robust (test set + seeds) | - |
| CodeT5→GPT-2 medium linear − native GPT-2 medium | 3 | -1.07 [-1.80, -0.35] | -2.0 | 0.0018 | 0.0306 | 0.0022 | 0.0737 | 0.7932 | test-set only | - |
| linear − naive (codet5→plbart; concode_java_40k) | 3 | +4.61 [+3.78, +5.47] | +14.3 | 0.0002 | 0.0162 | 0.0003 | 0.0016 | 0.0556 | test-set only | - |
| linear − resampler (codet5→plbart; concode_java_40k) | 3 | +4.38 [+3.70, +5.08] | +94.2 | 0.0002 | 0.0162 | 0.0003 | 0.0000 | 0.0020 | robust (test set + seeds) | - |
| linear − native PLBART (codet5→plbart; concode_java_40k) | 3 | -2.92 [-3.90, -1.94] | -5.9 | 0.0002 | 0.0162 | 0.0003 | 0.0095 | 0.2096 | test-set only | - |
| linear − native PLBART + LoRA (codet5→plbart; concode_java_40k) | 3 | +1.41 [+0.39, +2.47] | +3.2 | 0.0048 | 0.0624 | 0.0056 | 0.0315 | 0.5666 | not significant | - |

## 15. Error analysis (stored predictions)

Length ratio = mean(predicted tokens / reference tokens); empty = share of empty outputs; BLEU by reference-length tercile (short / medium / long) shows whether a system only fails on long targets.

| system | n seeds | length ratio | empty % | BLEU short | BLEU medium | BLEU long |
|---|---|---|---|---|---|---|
| codet5>plbart|naive|full|ep3|q0|concode_java | 15 | 0.93 | 0.0 | 38.2 | 22.6 | 7.6 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 15 | 1.04 | 0.0 | 50.6 | 28.0 | 12.4 |
| codet5>plbart|resampler|full|ep3|q32|concode_java | 15 | 1.03 | 0.0 | 44.9 | 25.0 | 10.8 |
| codet5>plbart|calm|full|ep3|q0|concode_java | 5 | 1.06 | 0.0 | 45.0 | 25.4 | 12.0 |
| native>plbart|none|full|ep3|q0|concode_java | 15 | 0.94 | 0.0 | 64.7 | 38.0 | 14.0 |
| native>plbart|none|lora|ep3|q0|concode_java | 5 | 0.86 | 0.0 | 68.6 | 36.3 | 10.3 |
| codet5>codegpt|linear|full|ep3|q0|concode_java | 15 | 1.00 | 0.0 | 49.3 | 25.4 | 11.3 |
| native>codegpt|none|full|ep3|q0|concode_java | 15 | 1.00 | 0.0 | 56.0 | 29.3 | 12.3 |

## 6. BPC truncation sensitivity of the screening ranking

Complete screening seeds: [13, 42, 2024] (20 pairs). Training truncation 128; evaluation truncation varied. 'full' = characters of the untruncated reference (paper definition); 'trunc' = characters of the truncated reference.

| eval truncation | denominator | Spearman vs L=128/full (mean, min over seeds) | Kendall τ (mean) | top-4 overlap (mean) | rank of CodeT5→PLBART / CodeT5→CodeGPT (mean) | % refs truncated (mean over pairs) |
|---|---|---|---|---|---|---|
| 32 | full | 0.970, 0.961 | 0.891 | 3.3 | 1.0 / 10.3 | 50.2 |
| 32 | trunc | 0.970, 0.961 | 0.891 | 3.3 | 1.0 / 10.3 | 50.2 |
| 64 | full | 0.995, 0.992 | 0.972 | 3.7 | 1.0 / 10.0 | 19.8 |
| 64 | trunc | 0.996, 0.992 | 0.975 | 3.7 | 1.0 / 9.7 | 19.8 |
| 128 | full | 1.000, 1.000 | 1.000 | 4.0 | 1.0 / 9.7 | 0.4 |
| 128 | trunc | 0.997, 0.994 | 0.982 | 3.7 | 1.0 / 8.7 | 0.4 |
| 256 | full | 1.000, 1.000 | 1.000 | 4.0 | 1.0 / 9.7 | 0.0 |
| 256 | trunc | 0.997, 0.994 | 0.982 | 3.7 | 1.0 / 8.7 | 0.0 |
| 512 | full | 1.000, 1.000 | 1.000 | 4.0 | 1.0 / 9.7 | 0.0 |
| 512 | trunc | 0.997, 0.994 | 0.982 | 3.7 | 1.0 / 8.7 | 0.0 |

Between-seed reliability of the L=128 ranking: mean pairwise Spearman = 0.768 (min 0.671).

Mean screening BPC at L=128 (rank, pair):

 1. codet5→plbart: 0.691
 2. plbart→plbart: 0.711
 3. unixcoder→plbart: 0.716
 4. roberta→plbart: 0.746
 5. plbart→codegpt: 0.757
 6. codebert→plbart: 0.768
 7. unixcoder→codegpt: 0.806
 8. codebert→codegpt: 0.852
 9. codet5→gpt2: 0.853
10. plbart→gpt2: 0.875
11. unixcoder→gpt2: 0.897
12. codebert→gpt2: 0.969
13. codet5→codegpt: 1.006
14. roberta→codegpt: 1.112
15. plbart→codet5: 1.133
16. codet5→codet5: 1.159
17. roberta→gpt2: 1.190
18. codebert→codet5: 1.397
19. roberta→codet5: 1.430
20. unixcoder→codet5: 1.430

Screening BPC vs final BLEU, fixed decoder PLBART, linear bridge (codet5, unixcoder, roberta, codebert; n=4): Spearman ρ = -0.40 (descriptive only; lower BPC should mean higher BLEU, i.e. ρ<0).

## 7. Dedicated-hardware timing

GPU: Tesla T4
Tesla T4; reps per system: 5–5. Step time = median of timed steps (tokenisation excluded, includes frozen-encoder forward); full-run estimate = median step × 1,875 steps. Latency = fixed 64 new tokens, batch 32. Median [IQR] over repetitions.

| system | trainable M | total M | step ms | est. 3-epoch train min | latency ms/ex | merged-LoRA latency | train peak GB | latency peak GB | other GPU procs |
|---|---|---|---|---|---|---|---|---|---|
| codet5>codegpt|linear|full|ep3|q0|concode_java | 125.0 | 234.6 | 823.0 [748.6–823.7] | 25.7 | 51.2 [48.1–51.4] | – | 7.88 | 4.59 | 0 |
| codet5>plbart|calm|full|ep3|q0|concode_java | 103.6 | 213.2 | 478.0 [477.1–479.1] | 14.9 | 45.1 [44.8–45.1] | – | 3.82 | 2.64 | 0 |
| codet5>plbart|linear|frozen|ep3|q0|concode_java | 0.6 | 206.1 | 415.0 [365.8–416.3] | 13.0 | 38.6 [38.2–38.6] | – | 2.89 | 1.43 | 0 |
| codet5>plbart|linear|full|ep3|q0|concode_java | 96.5 | 206.1 | 433.5 [433.3–485.1] | 13.5 | 33.3 [33.2–35.0] | – | 4.00 | 2.51 | 0 |
| codet5>plbart|linear|lora|ep3|q0|concode_java | 1.2 | 206.7 | 432.8 [380.1–434.3] | 13.5 | 44.7 [43.8–44.8] | 38.5 [37.3–38.7] | 3.15 | 1.44 | 0 |
| codet5>plbart|resampler|full|ep3|q32|concode_java | 110.7 | 220.3 | 480.3 [468.9–481.9] | 15.0 | 34.7 [34.4–34.7] | – | 3.92 | 2.72 | 0 |
| native>codegpt|none|full|ep3|q0|concode_java | 124.4 | 124.4 | 540.8 [538.1–586.6] | 16.9 | 31.6 [31.4–33.8] | – | 9.76 | 4.11 | 0 |
| native>codet5|none|full|ep3|q0|concode_java | 222.9 | 222.9 | 1488.0 [1364.4–1489.6] | 46.5 | 60.6 [58.4–60.8] | – | 10.48 | 4.11 | 0 |
| native>plbart|none|adapter64|ep3|q0|concode_java | 1.2 | 140.4 | 198.9 [187.3–202.2] | 6.2 | 25.5 [25.2–25.6] | – | 3.44 | 0.89 | 0 |
| native>plbart|none|full|ep3|q0|concode_java | 139.2 | 139.2 | 320.5 [318.7–321.7] | 10.0 | 19.6 [19.5–19.7] | – | 5.22 | 2.63 | 0 |
| native>plbart|none|lora|ep3|q0|concode_java | 0.9 | 140.1 | 229.5 [226.3–233.5] | 7.2 | 30.4 [30.3–30.7] | 23.6 [23.5–23.7] | 3.80 | 0.88 | 0 |
| unixcoder>plbart|linear|full|ep3|q0|concode_java | 96.5 | 222.4 | 428.8 [387.3–429.0] | 13.4 | 33.0 [32.1–33.1] | – | 4.06 | 2.55 | 0 |
