# UCF-Bridge

Code and results for the paper **"UCF-Bridge: Learned Lightweight Bridges for Composing Heterogeneous Pretrained Code Models in Java Code Generation"**.

UCF-Bridge joins a frozen pretrained encoder to a different pretrained decoder through a small learned bridge (naive, linear, or resampler), so that models with different tokenizers can be composed for Java code generation on CodeXGLUE CONCODE.

## Authors

Billington Muchiri Muria, Solomon Mwanjele, Joash Kiprotich Bii

## Contents

| File / folder | Description |
|---|---|
| `ucf_bridge_experiment.py` | Main pipeline: screening, main study, compatibility measures, analysis |
| `kaggle_full_run.py` | Kaggle runner that executes the pipeline phases end to end |
| `__huggingface_repos__.json` | Exact Hugging Face commit hashes of every checkpoint and dataset used (paper Table VIII) |
| `requirements.txt` | Python dependencies and versions (see below) |
| `results/` | Run records, tables, figures and logs; `per_example_and_predictions.zip` holds predictions and per-example scores |

## Environment

- Hardware used for the reported results: Kaggle notebook, 2 x NVIDIA T4 GPUs, total wall-clock time 5 h 46 min.
- Key versions: PyTorch 2.10.0 (CUDA 12.8 build), Transformers 5.0.0, Tokenizers 0.22.2, Datasets 5.0.0 (full list in `requirements.txt`). sacreBLEU 2.6.0, javalang 0.13.0, Python 3.12.13.

```bash
pip install -r requirements.txt
```

## Reproducing the paper

Full study (medium preset, seeds 13, 42 and 2024, top three heterogeneous pairs):

```bash
python ucf_bridge_experiment.py --phase all --preset medium --out_dir results
```

Individual phases:

```bash
python ucf_bridge_experiment.py --phase selftest --out_dir results   # quick environment check
python ucf_bridge_experiment.py --phase screen   --out_dir results   # 400-step BPC screening of 20 pairs
python ucf_bridge_experiment.py --phase main     --out_dir results   # main study (54 runs)
python ucf_bridge_experiment.py --phase compat   --out_dir results   # CKA and similarity measures
python ucf_bridge_experiment.py --phase analyze  --out_dir results   # regenerate tables and figures
```

The pipeline is resumable: each run is stored as one JSON record keyed by a hash of its hyper-parameters, so an interrupted run continues where it stopped. On Kaggle, upload both scripts and run `kaggle_full_run.py`.

To regenerate the paper's tables and figures from the stored records without retraining, run the `analyze` phase on the provided `results/` folder.

## Checkpoints

All pretrained models and the dataset are pinned to the commits listed in `__huggingface_repos__.json`:
Salesforce/codet5-base, uclanlp/plbart-base, microsoft/codebert-base, microsoft/unixcoder-base, roberta-base, microsoft/CodeGPT-small-java-adaptedGPT2, gpt2 (screening only), and google/code_x_glue_tc_text_to_code.

## Notes on the evaluation

- Syntactic validity (parses with javalang) is a proxy only. It does not establish functional correctness.
- Seed-level confidence intervals use n = 3 seeds and describe run-to-run variability. System comparisons use a paired bootstrap over test examples (10,000 resamples) with Holm correction over 25 comparisons.

## Citation

```
[Add the final citation / DOI here after acceptance]
```

## License

MIT (see `LICENSE`).
