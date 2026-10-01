| config | BLEU | trainable_M | BLEU_per_M_trainable | pareto_optimal |
|---|---|---|---|---|
| native:plbart | 38.303 | 254.432 | 0.151 | True |
| native:codet5 | 37.008 | 296.840 | 0.125 | False |
| codet5->plbart \| linear \| dec_tune | 33.916 | 173.304 | 0.196 | True |
| codet5->codet5 \| naive \| dec_tune | 33.160 | 187.233 | 0.177 | False |
| plbart->plbart \| naive \| dec_tune | 33.160 | 172.712 | 0.192 | True |
| plbart->plbart \| linear \| dec_tune | 32.629 | 173.304 | 0.188 | False |
| native:codegpt | 32.173 | 163.042 | 0.197 | True |
| codet5->codet5 \| resampler \| dec_tune | 30.526 | 202.027 | 0.151 | False |
| codet5->codet5 \| linear \| dec_tune | 30.485 | 187.826 | 0.162 | False |
| codet5->plbart \| resampler \| dec_tune | 30.027 | 187.506 | 0.160 | False |
| codet5->codegpt \| linear \| dec_tune | 28.714 | 163.634 | 0.175 | False |
| plbart->plbart \| resampler \| dec_tune | 28.507 | 187.506 | 0.152 | False |
| codet5->codegpt \| resampler \| dec_tune | 24.043 | 177.836 | 0.135 | False |
| unixcoder->plbart \| resampler \| dec_tune | 23.195 | 187.506 | 0.124 | False |
| codet5->plbart \| naive \| dec_tune | 22.892 | 172.712 | 0.133 | False |
| unixcoder->plbart \| linear \| dec_tune | 22.161 | 173.304 | 0.128 | False |
| codet5->codegpt \| naive \| dec_tune | 16.998 | 163.042 | 0.104 | False |
| unixcoder->plbart \| naive \| dec_tune | 14.550 | 172.712 | 0.084 | False |