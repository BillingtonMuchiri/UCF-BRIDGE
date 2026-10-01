| system | baseline | delta_sentBLEU | CI_lo | CI_hi | p_boot | cohens_d_seeds | welch_p_seeds | p_holm | significant_holm_0.05 |
|---|---|---|---|---|---|---|---|---|---|
| codet5->codegpt \| linear \| dec_tune | native:codegpt | -4.631 | -5.687 | -3.573 | 0.000 | -4.798 | 0.024 | 0.003 | True |
| codet5->codegpt \| linear \| dec_tune | codet5->codegpt \| naive \| dec_tune | 13.939 | 12.654 | 15.259 | 0.000 | 13.109 | 0.000 | 0.003 | True |
| codet5->codegpt \| naive \| dec_tune | native:codegpt | -18.571 | -20.125 | -17.015 | 0.000 | -27.073 | 0.000 | 0.003 | True |
| codet5->codegpt \| resampler \| dec_tune | native:codegpt | -13.637 | -15.015 | -12.304 | 0.000 | -15.255 | 0.002 | 0.003 | True |
| codet5->codegpt \| resampler \| dec_tune | codet5->codegpt \| naive \| dec_tune | 4.934 | 4.051 | 5.839 | 0.000 | 9.388 | 0.000 | 0.003 | True |
| codet5->codet5 \| linear \| dec_tune | native:codet5 | -6.610 | -7.673 | -5.544 | 0.000 | -21.015 | 0.000 | 0.003 | True |
| codet5->codet5 \| linear \| dec_tune | codet5->codet5 \| naive \| dec_tune | -2.451 | -3.409 | -1.494 | 0.000 | -5.859 | 0.003 | 0.003 | True |
| codet5->codet5 \| naive \| dec_tune | native:codet5 | -4.159 | -5.155 | -3.157 | 0.000 | -9.562 | 0.002 | 0.003 | True |
| codet5->codet5 \| resampler \| dec_tune | native:codet5 | -8.330 | -9.405 | -7.256 | 0.000 | -10.420 | 0.004 | 0.003 | True |
| codet5->codet5 \| resampler \| dec_tune | codet5->codet5 \| naive \| dec_tune | -4.170 | -5.225 | -3.139 | 0.000 | -3.729 | 0.016 | 0.003 | True |
| codet5->plbart \| linear \| dec_tune | native:plbart | -6.619 | -7.774 | -5.426 | 0.000 | -7.190 | 0.001 | 0.003 | True |
| codet5->plbart \| linear \| dec_tune | codet5->plbart \| naive \| dec_tune | 14.050 | 12.679 | 15.457 | 0.000 | 15.927 | 0.000 | 0.003 | True |
| codet5->plbart \| naive \| dec_tune | native:plbart | -20.668 | -22.254 | -19.063 | 0.000 | -21.315 | 0.000 | 0.003 | True |
| codet5->plbart \| resampler \| dec_tune | native:plbart | -11.622 | -12.911 | -10.325 | 0.000 | -13.023 | 0.000 | 0.003 | True |
| codet5->plbart \| resampler \| dec_tune | codet5->plbart \| naive \| dec_tune | 9.046 | 7.862 | 10.231 | 0.000 | 9.985 | 0.000 | 0.003 | True |
| plbart->plbart \| linear \| dec_tune | native:plbart | -7.303 | -8.405 | -6.186 | 0.000 | -11.921 | 0.002 | 0.003 | True |
| plbart->plbart \| linear \| dec_tune | plbart->plbart \| naive \| dec_tune | -1.747 | -2.728 | -0.747 | 0.000 | -1.247 | 0.243 | 0.003 | True |
| plbart->plbart \| naive \| dec_tune | native:plbart | -5.557 | -6.759 | -4.394 | 0.000 | -8.449 | 0.001 | 0.003 | True |
| plbart->plbart \| resampler \| dec_tune | native:plbart | -12.786 | -14.053 | -11.503 | 0.000 | -13.675 | 0.000 | 0.003 | True |
| plbart->plbart \| resampler \| dec_tune | plbart->plbart \| naive \| dec_tune | -7.229 | -8.382 | -6.097 | 0.000 | -6.802 | 0.002 | 0.003 | True |
| unixcoder->plbart \| linear \| dec_tune | native:plbart | -24.935 | -26.794 | -23.015 | 0.000 | -25.380 | 0.000 | 0.003 | True |
| unixcoder->plbart \| linear \| dec_tune | unixcoder->plbart \| naive \| dec_tune | 6.131 | 4.886 | 7.443 | 0.000 | 16.615 | 0.001 | 0.003 | True |
| unixcoder->plbart \| naive \| dec_tune | native:plbart | -31.066 | -33.078 | -29.036 | 0.000 | -50.514 | 0.000 | 0.003 | True |
| unixcoder->plbart \| resampler \| dec_tune | native:plbart | -23.941 | -25.780 | -22.082 | 0.000 | -23.561 | 0.000 | 0.003 | True |
| unixcoder->plbart \| resampler \| dec_tune | unixcoder->plbart \| naive \| dec_tune | 7.125 | 5.810 | 8.469 | 0.000 | 18.581 | 0.001 | 0.003 | True |