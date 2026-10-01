| config | n | empty_rate | near_maxlen_rate | len_outlier_rate |
|---|---|---|---|---|
| codet5->codegpt \| linear \| dec_tune | 1000 | 0.000 | 0.021 | 0.018 |
| codet5->codegpt \| naive \| dec_tune | 1000 | 0.000 | 0.003 | 0.014 |
| codet5->codegpt \| resampler \| dec_tune | 1000 | 0.000 | 0.027 | 0.038 |
| codet5->codet5 \| linear \| dec_tune | 1000 | 0.000 | 0.009 | 0.006 |
| codet5->codet5 \| naive \| dec_tune | 1000 | 0.000 | 0.000 | 0.008 |
| codet5->codet5 \| resampler \| dec_tune | 1000 | 0.000 | 0.011 | 0.010 |
| codet5->plbart \| linear \| dec_tune | 1000 | 0.000 | 0.005 | 0.010 |
| codet5->plbart \| naive \| dec_tune | 1000 | 0.000 | 0.015 | 0.010 |
| codet5->plbart \| resampler \| dec_tune | 1000 | 0.000 | 0.005 | 0.011 |
| native:codegpt | 1000 | 0.000 | 0.012 | 0.009 |
| native:codet5 | 1000 | 0.000 | 0.002 | 0.004 |
| native:plbart | 1000 | 0.000 | 0.004 | 0.008 |
| plbart->plbart \| linear \| dec_tune | 1000 | 0.000 | 0.006 | 0.011 |
| plbart->plbart \| naive \| dec_tune | 1000 | 0.000 | 0.004 | 0.007 |
| plbart->plbart \| resampler \| dec_tune | 1000 | 0.000 | 0.004 | 0.007 |
| unixcoder->plbart \| linear \| dec_tune | 1000 | 0.000 | 0.004 | 0.010 |
| unixcoder->plbart \| naive \| dec_tune | 1000 | 0.000 | 0.039 | 0.022 |
| unixcoder->plbart \| resampler \| dec_tune | 1000 | 0.000 | 0.006 | 0.014 |