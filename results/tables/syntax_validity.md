| config | n | pred_syntax_valid_rate | ref_syntax_valid_rate |
|---|---|---|---|
| native:plbart | 3000 | 0.994 | 1.000 |
| native:codet5 | 3000 | 0.994 | 1.000 |
| codet5->codet5 \| naive \| dec_tune | 3000 | 0.991 | 1.000 |
| plbart->plbart \| naive \| dec_tune | 3000 | 0.990 | 1.000 |
| unixcoder->plbart \| linear \| dec_tune | 3000 | 0.988 | 1.000 |
| plbart->plbart \| resampler \| dec_tune | 3000 | 0.988 | 1.000 |
| codet5->plbart \| linear \| dec_tune | 3000 | 0.988 | 1.000 |
| plbart->plbart \| linear \| dec_tune | 3000 | 0.986 | 1.000 |
| codet5->plbart \| resampler \| dec_tune | 3000 | 0.985 | 1.000 |
| codet5->codet5 \| linear \| dec_tune | 3000 | 0.984 | 1.000 |
| codet5->plbart \| naive \| dec_tune | 3000 | 0.984 | 1.000 |
| codet5->codegpt \| naive \| dec_tune | 3000 | 0.983 | 1.000 |
| codet5->codet5 \| resampler \| dec_tune | 3000 | 0.982 | 1.000 |
| unixcoder->plbart \| resampler \| dec_tune | 3000 | 0.982 | 1.000 |
| native:codegpt | 3000 | 0.972 | 1.000 |
| codet5->codegpt \| linear \| dec_tune | 3000 | 0.963 | 1.000 |
| codet5->codegpt \| resampler \| dec_tune | 3000 | 0.945 | 1.000 |
| unixcoder->plbart \| naive \| dec_tune | 3000 | 0.790 | 1.000 |