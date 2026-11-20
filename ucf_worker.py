#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ucf_extensions_kaggle.py
========================
Extension experiments for "Learned Bridges for Composing Heterogeneous Pretrained Code Models"
(UCF-Bridge).  One resumable file; every run writes one JSON record, so every table can be rebuilt.

COPY-PASTE / NOTEBOOK USE (switch based, ONE cell)
--------------------------------------------------
Paste this whole file into ONE Kaggle notebook cell (Accelerator: GPU T4 x2; Internet on), set MODE ("smoke" or "full") in the
block just below this docstring, and run the cell.  Nothing else is needed: missing or wrongly versioned packages
(transformers 4.46.3, peft 0.13.2, accelerate 1.1.1, ...) are installed by the cell itself (OPTIONS["auto_install"]) before the
first GPU stage, with no kernel restart.  Stages run in a fixed order, and with two
GPUs every job stage is worked on by both GPUs automatically (second GPU = a background worker process; the two share the
job queue through claim files).  Run the cell again after any interruption: finished runs are skipped, dead workers' claims
are taken over, long runs continue from their checkpoints.
  First MODE = "smoke" (about 1-2 h), then MODE = "full".
The same file still works from a terminal:  python ucf_extensions_kaggle_v3.py --stage baselines --device cuda:0   (arguments override MODE).

LIMITATION                              STAGE              (T4 GPU-hours, rough)
-------------------------------------   -----------------  ---------------------
~15 seeds for the core comparison       seeds15            ~14   (60 runs)
  (+ decoder-only target, optional)     seeds15_gpt        ~19   (45 runs)
non-CodeT5 encoder pair(s)              encoders           ~6
frozen-decoder (bridge-only) condition  frozen_dec         ~5
longer resampler training               long_resampler     ~8    (+ matched linear bridge)
dimension-mismatched bridging           dimmismatch        ~4    (512<->768; padtrunc / linear / resampler)
execution-based functional correctness  functional         ~5    (MBPP pass@1, Python; runs generated code!)
additional dataset                      newdata            ~7    (--dataset csn_java | csn_python | py_alpaca)
BPC truncation sensitivity              bpc_trunc          ~5    (20 pairs x 3 screening seeds x 5 lengths)
dedicated-hardware timing               timing             ~1    (run ALONE on one GPU)
tables / CIs / bootstrap                analyze            CPU
CodeBLEU backfill from saved preds       codebleu           CPU (after fixing the install)
pin checkpoint commit hashes (once)     pinrev             needs internet; writes revs.json
packages / checkpoints / datasets       check              ~5 min, needs internet

REVISION ADDITIONS (v3) -- written against the reviewer's request for "much more comprehensive results, evaluations,
comparison and details" (review form IJCS_261003Py):
external / parameter-efficient baselines   baselines          ~15   (75 new runs at n_seeds 5; CALM-style cross-attention, adapters, LoRA ranks, CodeT5+)
larger / dimension-mismatched components   scale              ~8    (CodeT5+ 770M encoder 1024->768, GPT-2 medium decoder)
more training data (40k, 3 seeds)          data40k            ~14   (same test set as the 10k study)
HumanEval transfer pass@1                  functional         free  (evaluated inside the MBPP runs)
multiple-comparison control, permutation tests, error analysis, CSV tables, publication figures:  analyze  (CPU)
Seed counts: every stage that used 3 seeds now uses --n_seeds (default 5; costs above and below scale with it).
Suggested priority when GPU time is short:  baselines > encoders > functional > frozen_dec > scale > newdata > data40k.

Typical use (Kaggle, 2 x T4; two workers share a job queue through claim files):
                    sacrebleu javalang codebleu sentencepiece scipy protobuf
    (restart the Kaggle kernel after installing so the pinned transformers is the one imported)

Smoke test first (tiny data, a few steps, ~10 min per stage):  add  --smoke
If a session died and left stale claims:                         add  --clear_claims

SIMPLEST RESUME (v5): AUTO-SYNC TO A KAGGLE DATASET
---------------------------------------------------
Set OPTIONS["sync_dataset"] = "your-username/ucf-ext-records" (or just a slug).  Needs Internet on, and Kaggle credentials: Add-ons >
Secrets > KAGGLE_USERNAME and KAGGLE_KEY (kaggle.com > Settings > API > Create New Token).  The records zip (a few MB) is then uploaded
as a new dataset version about every 60 min, whenever new runs finished, and at the end of the cell, so a timeout or crash loses at most
the last hour.  ONE-TIME: after the first upload exists, Add Input -> Your Datasets -> that dataset.  From then on a new session is just
"run the cell again": the zip is imported automatically.  No Save Version needed (it still works as before if you prefer it).

CSV RESULTS FILE (v6)
---------------------
Every pack (automatic, every 20 min, and at the end) also writes ucf_ext_records.csv: one row per finished run with readable columns
(file, kind, system, seed, bleu, em, valid, codebleu, train_s) plus the column record_json holding the complete record.  It opens in
Excel / pandas for a quick look AND it is a resume file: attach the folder / dataset containing it as an input in the next session and
the runs are imported automatically (or pass --resume_from path/ucf_ext_records.csv).  Do not edit the record_json column.
Without the sync option: download ucf_ext_records.csv from the notebook's Output, upload it as a dataset, Add Input, run the cell.

RESUMING FROM A PREVIOUS SESSION (v3.1)
---------------------------------------
Everything is keyed by (system, seed): a finished run is one JSON record, so a new session simply continues.
  1. Kaggle: at the end of a session press "Save Version" (Save & Run All, or Quick Save) so /kaggle/working/ucf_ext is kept as
     the notebook's output.  In the next session: Add Input -> "Notebook Output Files" -> your notebook.  Nothing else to do:
     every stage auto-imports records (and revs.json) found under /kaggle/input/*/ucf_ext/records or
     /kaggle/input/*/ucf_ext_records.zip.  Explicit alternatives:  --resume_from PATH [PATH ...]   or   UCF_PREV=PATH.
     `--stage pack` writes ucf_ext_records.zip (one file to download / upload as a Kaggle dataset); `--stage import` only imports.
  2. `--stage status` shows done / in-progress / stale / failed per stage and prints the commands that continue the work.
  3. Records are written atomically; a truncated or corrupt file is quarantined (*.corrupt) and the run is redone.
  4. Claim files carry pid + host + a heartbeat: the claims of a dead worker (killed kernel, session timeout) are taken over
     automatically (dead pid on this host, or no heartbeat for UCF_STALE_S = 600 s).  No manual --clear_claims needed.
  5. A run that raises is retried at most twice (counter in <record>.fail); `--retry_failed` resets the counters.
  6. Long runs (>= --ckpt_min_steps = 3000 steps: 40k data, 6/12-epoch resampler, 10-epoch frozen) write a training checkpoint
     (trainable weights + optimizer + scheduler + data-order state) every --ckpt_every_min = 15 min into ucf_ext/ckpt and, after
     a restart, continue from it.  Resumed runs are not bit-identical to uninterrupted ones (dropout / CUDA streams); the record
     lists the resume points in "resumes".  Checkpoints are large (~1.7 GB for a fully tuned PLBART): they are only carried over
     to a new session with --with_ckpt (or by keeping the same working directory).
"""
# =====================================================================================================================
#  THE ONE SWITCH  --  set MODE, run the cell.
#
#    MODE = "smoke"   end-to-end test of EVERY stage on tiny data (1 seed, a few training steps each): selftest, environment
#                     check, all GPU stages, timing, analysis.  Starts clean every time (its own folder ucf_ext/smoke, never
#                     mixed with real results).  Roughly 1-2 h on 2 x T4 (estimate).  Run it first in every new environment.
#    MODE = "full"    the real experiments: pin checkpoint revisions, then every stage in priority order (baselines, encoders,
#                     functional, functional_best, frozen_dec, scale, newdata, data40k, seeds15, seeds15_gpt, long_resampler,
#                     dimmismatch, bpc_trunc), dedicated-hardware timing on one GPU, then CodeBLEU backfill, analysis
#                     (tables, tests, figures), and a zip of all records.  About 120 GPU-hours in total by the script's own
#                     estimates (5 seeds) -- far more than one Kaggle session: just run the cell again in each new session;
#                     finished runs are skipped, interrupted ones continue.  To drop stages, list them in OPTIONS["skip_stages"].
# =====================================================================================================================
MODE = "full"          # "smoke" | "full"

# ---- advanced options: the defaults are right for a normal run -------------------------------------------------------
OPTIONS = dict(
    auto_install=True,               # pip-install missing / wrongly versioned packages inside this cell (needs Internet)
    gpus=["cuda:0", "cuda:1"],       # one entry = single GPU; two = both GPUs share the queue (timing always uses the first only)
    skip_stages=[],                  # e.g. ["seeds15_gpt", "data40k"] to leave out the most expensive stages of the full run
    n_seeds=5,                       # seeds for the secondary stages (1-5); the smoke mode always uses 1
    out_dir=None,                    # None = /kaggle/working/ucf_ext (or env UCF_OUT)
    resume_from=[],                  # extra paths of previous output (dirs or ucf_ext_records.zip); /kaggle/input is scanned anyway
    kaggle_username=None, kaggle_key=None,   # alternative to Add-ons > Secrets: paste them here (they are then stored in your notebook
                                     # version: keep the notebook PRIVATE, and never share a notebook containing a real key)
    sync_every_min=20,               # how often (minutes) the records CSV/zip is written and (if sync_dataset is set) uploaded, after a run finishes
    sync_dataset=None,               # "user/slug" (or "slug"): auto-upload ucf_ext_records.zip as a Kaggle dataset version (see SIMPLEST RESUME)
    with_ckpt=False,                 # also import training checkpoints of a previous session (large)
    retry_failed=False,              # reset the failure counters of runs that raised before
    clear_claims=False,              # normally unnecessary: dead workers' claims are taken over automatically
    newdata_datasets=["csn_java", "csn_python", "py_alpaca"],   # any of: csn_java, csn_python, py_alpaca (all three = the full plan)
    baselines_datasets=["concode_java"],
    max_hours=10.5,                  # session time budget (Kaggle limit 12 h): after this no NEW run is started; the launcher then packs
                                     # the records and prints the resume steps.  None = no limit.  Lower it if your runs are long.
    allow_val_fallback=False,        # True = knowingly accept a validation-carved CONCODE test set when no labelled test set is found
    # --- functional_best protocol (best practice defaults: validation-based early stopping, hard cap) ---
    fb_enc="codet5", fb_dec="plbart", fb_bridge="linear",
    fb_max_epochs=20, fb_min_epochs=4, fb_patience=4,
    fb_target=None,                  # optional validation pass@1 (%) at which to stop; None = patience only
    fb_no_ref=False,                 # True = skip the matched native reference
    # --- rarely changed ---
    seeds=None, long_epochs=[6, 12], reps=5, timing_steps=150, ckpt_every_min=15.0, ckpt_min_steps=3000,
    no_figs=False, force_versions=False,
)

import importlib.util, os, sys, json, time, math, re, argparse, random, hashlib, subprocess, gc, ast, traceback
import threading, socket, signal, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import warnings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # the xet-read-token endpoint is what returned HTTP 429
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation after OOM retries
warnings.filterwarnings("ignore", message=r".*Detected call of .{0,3}lr_scheduler\.step.*")   # harmless: GradScaler skips the first step on inf grads
warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).{0,3} before.*")


def _load_hf_token():
    """A logged-in HF token lifts the anonymous rate limit.  Kaggle: Add-ons > Secrets > label HF_TOKEN (optional)."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
    except Exception:
        pass


_load_hf_token()

HAVE_TORCH = True
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:                      # lets analyze / selftest run on a CPU-only machine
    HAVE_TORCH = False
    class _NoTorch:                      # supplies only what decorators need at import time
        @staticmethod
        def no_grad():
            return lambda f: f
    torch = _NoTorch()
    class nn:                            # noqa
        Module = object

ROOT = Path(os.environ.get("UCF_OUT", "/kaggle/working/ucf_ext"))
ROOT0 = ROOT                             # main() derives ROOT from this (adds /smoke), so repeated calls in a notebook do not nest
REC = ROOT / "records"
SMOKE = False
SCRIPT_VERSION = "v3.3-oom-safe"
STALE_S = float(os.environ.get("UCF_STALE_S", 600))   # a claim without heartbeat for this long belongs to a dead worker
HEARTBEAT_S = 30
MAX_FAILS = 2                                          # a run that raised is retried at most this many times in total
CKPT_INTERVAL_S = 15 * 60                              # training checkpoint every 15 min (0 = off)
CKPT_MIN_STEPS = 3000                                  # only runs at least this long are checkpointed
# "Train until it passes" protocol (stage functional_best).  Best practice: stop on VALIDATION pass@1 (never on test),
# keep the best-validation epoch, hard epoch cap, patience, several seeds, same protocol for the reference system.
FB = dict(enc="codet5", dec="plbart", bridge="linear", max_epochs=20, min_epochs=4, patience=4, target=None, ref=True)


# Versions the code was written against (4.x API: generate(encoder_outputs=...), PLBart decoder_start_token_id,
# GPT-2 generate(inputs_embeds=...)).  transformers 5.x changed several of these; the guard below stops early.
PINNED = dict(transformers="4.46.3", peft="0.13.2", accelerate="1.1.1")


def version_guard(force=False):
    """Exit with a clear message if transformers is outside the tested major version."""
    try:
        import transformers
    except ImportError:
        return True
    v = transformers.__version__
    major = int(v.split(".")[0])
    if major != int(PINNED["transformers"].split(".")[0]):
        msg = (f"transformers {v} is installed but this script targets {PINNED['transformers']} (4.x).\n"
               f'Fix:  pip install "transformers=={PINNED["transformers"]}" "peft=={PINNED["peft"]}" '
               f'"accelerate=={PINNED["accelerate"]}"  and restart the kernel.  (--force_versions overrides.)')
        if not force:
            sys.exit(msg)
        print("WARNING:", msg, flush=True)
    elif v != PINNED["transformers"]:
        print(f"note: transformers {v} differs from the pinned {PINNED['transformers']} (same major; proceeding).", flush=True)
    return True

# --------------------------------------------------------------------------------------
# 0. Configuration  (mirrors Table II of the paper)
# --------------------------------------------------------------------------------------
CK = dict(codet5="Salesforce/codet5-base", codet5_small="Salesforce/codet5-small",   # small: hidden size 512
          plbart="uclanlp/plbart-base",
          codebert="microsoft/codebert-base", unixcoder="microsoft/unixcoder-base",
          roberta="roberta-base", gpt2="gpt2", gpt2m="gpt2-medium",
          codet5p="Salesforce/codet5p-220m", codet5p770="Salesforce/codet5p-770m",
          codegpt_java="microsoft/CodeGPT-small-java-adaptedGPT2",
          codegpt_python="microsoft/CodeGPT-small-py-adaptedGPT2")
# Optional: pin FULL 40-char commit hashes here (Table XIII lists only the first 8 characters), e.g.
# REV = {"Salesforce/codet5-base": "02cd2d31...", ...}
REV = {}
_REVF = Path(os.environ.get("UCF_REVS", str(ROOT / "revs.json")))
if _REVF.exists():                       # written by `--stage pinrev`; pins every checkpoint to an exact commit
    try:
        REV = json.load(open(_REVF))
    except Exception:
        REV = {}
ENC_KIND = dict(codet5="t5", codet5_small="t5", plbart="plbart", codebert="auto", unixcoder="auto", roberta="auto", codet5p="t5", codet5p770="t5")
DEC_KIND = dict(plbart="s2s", codet5="s2s", codet5_small="s2s", codet5p="s2s", codegpt="clm", gpt2="clm", gpt2m="clm")
HP = dict(lr_dec=5e-5, lr_bridge=5e-4, bs=16, warm=0.05, wd=0.01, clip=1.0,
          max_src=256, max_tgt=128, max_new=128)
SEEDS_CORE5 = [13, 42, 2024, 7, 99]
SEEDS_NEW10 = [1, 2, 3, 5, 11, 17, 21, 100, 123, 777]
SEEDS15 = SEEDS_CORE5 + SEEDS_NEW10
SEEDS3 = [13, 42, 2024]
SEEDS_MID = list(SEEDS_CORE5)          # seeds for every secondary stage; main() shortens it with --n_seeds
DS = {
    "concode_java": dict(lang="java", sizes=(10000, 500, 1000)),
    "csn_java":     dict(lang="java", sizes=(10000, 500, 1000)),     # CodeSearchNet docstring -> method (human-written)
    "csn_python":   dict(lang="python", sizes=(10000, 500, 1000)),   # CodeSearchNet docstring -> function
    "py_alpaca":    dict(lang="python", sizes=(2000, 179, 500)),     # dataset used in the paper's Python replication
    "mbpp_py":      dict(lang="python", sizes=(0, 0, 0)),            # MBPP (+ py_alpaca for training); executed unit tests; HumanEval transfer test
    "concode_java_40k": dict(lang="java", sizes=(40000, 500, 1000)), # same val/test draw as concode_java, 4x more training data
}
MAX_CODE_CHARS = 600
BPC_LENS = [32, 64, 128, 256, 512]
# Per-seed test BLEU from Table X of the paper (seeds 13, 42, 2024, 7, 99) for the replication check.
PAPER_TABLE_X = {
    "native": [25.8, 24.6, 24.7, 25.2, 25.4], "naive": [14.5, 15.8, 14.6, 15.8, 15.6],
    "resampler": [20.0, 19.5, 19.5, 19.7, 20.1], "linear": [23.2, 22.5, 22.2, 22.9, 23.1]}


def ckey(name, lang):
    return "codegpt_" + lang if name == "codegpt" else name


def S(enc, dec, bridge="linear", tune="full", nq=32, epochs=3, ds="concode_java"):
    """Run specification.  enc=None -> native model (no bridge)."""
    return dict(enc=enc, dec=dec, bridge=(bridge if enc else "none"), tune=tune,
                nq=(nq if (enc and bridge == "resampler") else 0), epochs=epochs, ds=ds, lang=DS[ds]["lang"])


def sys_id(sp):
    return f'{sp["enc"] or "native"}>{sp["dec"]}|{sp["bridge"]}|{sp["tune"]}|ep{sp["epochs"]}|q{sp["nq"]}|{sp["ds"]}'


def _h(*a):
    return hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------------------
def _sample(rows, n, rng):
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def _dedup(rows, banned=()):
    """Drop pairs whose (normalised) target code is already seen or is in `banned`; first occurrence wins."""
    seen, out = set(banned), []
    for a, b in rows:
        k = norm(b)
        if k not in seen:
            seen.add(k)
            out.append((a, b))
    return out


def _decontaminate(tr, va, te, tag):
    """Exact-duplicate removal inside each split + removal of train/val code that also occurs in test (and train code
    in val).  Applied to the datasets added in this script only; concode_java is left as in the paper."""
    te2 = _dedup(te)
    va2 = _dedup(va, {norm(b) for _, b in te2})
    tr2 = _dedup(tr, {norm(b) for _, b in te2} | {norm(b) for _, b in va2})
    print(f"[{tag}] dedup/decontamination removed train {len(tr) - len(tr2)}, val {len(va) - len(va2)}, "
          f"test {len(te) - len(te2)}", flush=True)
    return tr2, va2, te2


def _alpaca_rows():
    from datasets import load_dataset
    rows = []
    for r in load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]:
        src = (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")).strip()
        if src and r["output"] and r["output"].strip() and len(r["output"]) <= MAX_CODE_CHARS:
            rows.append((src, r["output"].strip()))
    return rows


_CODEXGLUE_RAW = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/text-to-code/"
CONCODE_TEST_URL = _CODEXGLUE_RAW + "dataset/concode/test.json"          # NL side only: its `code` field is EMPTY (same as the HF copy)
CONCODE_ANSWERS_URL = _CODEXGLUE_RAW + "evaluator/answers.json"          # only a 100-line demo sample of the references (evaluator example)
CONCODE_MIRROR_URLS = (                                                  # full labelled copies (2000 test rows with code), tried in order
    "https://huggingface.co/datasets/AhmedSSoliman/CodeXGLUE-CONCODE/resolve/main/test.json",
)


def _read_json_objs(kind, loc):
    """Read a JSON-lines file or a JSON array from a local path or a URL; returns a list of dicts (strings are wrapped as {'code': s})."""
    if kind == "file":
        txt = Path(loc).read_text(encoding="utf-8")
    else:
        import urllib.request
        txt = urllib.request.urlopen(loc, timeout=90).read().decode("utf-8")
    t = txt.lstrip()
    objs = json.loads(t) if t.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    return [o if isinstance(o, dict) else {"code": str(o)} for o in objs]


def _concode_external_test(min_rows, nl_hint=None):
    """The Hugging Face copy of CONCODE AND CodeXGLUE's dataset/concode/test.json both have an EMPTY `code` field for the test set:
    the labelled references are published separately as evaluator/answers.json in the CodeXGLUE repository (same order as test.json).
    Sources tried: (1) env UCF_CONCODE_TEST = a CodeXGLUE-style file with `nl` and `code` (JSON lines or array),
    (2) env UCF_CONCODE_ANSWERS = path of evaluator/answers.json, (3) the CodeXGLUE GitHub repository (needs Internet):
    test.json (nl) + evaluator/answers.json (code), paired by line number.  `nl_hint` = the HF test NL strings (used when no nl is found).
    Returns (rows, source) or (None, None)."""
    cands = []
    if os.environ.get("UCF_CONCODE_TEST"):
        cands.append(("file", os.environ["UCF_CONCODE_TEST"]))
    for u in CONCODE_MIRROR_URLS:                          # a mirror with the full labelled test set comes first
        cands.append(("url", u))
    cands.append(("url", CONCODE_TEST_URL))
    if os.environ.get("UCF_CONCODE_ANSWERS"):
        cands.append(("file", os.environ["UCF_CONCODE_ANSWERS"]))
    cands.append(("url", CONCODE_ANSWERS_URL))
    nls, codes = [], []                      # line-aligned nl / code lists collected from the candidates
    for kind, loc in cands:
        try:
            objs = _read_json_objs(kind, loc)
            rows = [(o["nl"].strip(), o["code"].strip()) for o in objs
                    if o.get("nl") and o.get("code") and o["nl"].strip() and o["code"].strip()]
            if len(rows) >= min_rows:
                return rows, loc
            if any(o.get("nl") for o in objs) and not nls:
                nls = [(o.get("nl") or "").strip() for o in objs]
            if any(o.get("code") for o in objs) and not codes:
                codes = [(o.get("code") or "").strip() for o in objs]
                answers_src = loc
            if not rows:
                print(f"[concode] {loc}: {len(objs)} lines, {len(rows)} with both nl and code (nl: {sum(bool(x) for x in nls)}, "
                      f"code: {sum(bool(x) for x in codes)} lines so far).", flush=True)
        except Exception as e:
            print(f"WARNING: CONCODE test candidate {loc} unusable ({type(e).__name__}: {str(e)[:100]}).", flush=True)
    nls = nls if any(nls) else list(nl_hint or [])
    if codes and nls:
        if len(codes) != len(nls):
            print(f"WARNING: CONCODE answers ({len(codes)} lines) and test nl ({len(nls)} lines) differ in length "
                  "(the GitHub answers.json is only a 100-line sample); cannot pair them safely.", flush=True)
        else:
            rows = [(n, c) for n, c in zip(nls, codes) if n and c]
            if len(rows) >= min_rows:
                return rows, f"{answers_src} + test nl (paired by line number)"
            print(f"WARNING: only {len(rows)} usable paired CONCODE test rows (need {min_rows}).", flush=True)
    return None, None


def _concode_splits(te_n):
    """HF train/validation + a labelled test split (HF's if usable, else an external CodeXGLUE test file).
    Returns (sp, test_source); test_source None means no labelled test set was found."""
    from datasets import load_dataset
    ok = lambda a, b: bool(a and b and a.strip() and b.strip())
    raw = load_dataset("google/code_x_glue_tc_text_to_code")
    sp = {k: [(r["nl"].strip(), r["code"].strip()) for r in raw[k] if ok(r["nl"], r["code"])]
          for k in ("train", "validation", "test")}
    print("[concode] usable rows / raw rows: " + ", ".join(f"{k} {len(sp[k])}/{len(raw[k])}" for k in sp), flush=True)
    if len(sp["test"]) >= te_n:
        return sp, "HF test split"
    print(f"[concode] the HF test split has only {len(sp['test'])} usable rows (columns {list(raw['test'].features)}); "
          "looking for the labelled CodeXGLUE test file ...", flush=True)
    ext, src = _concode_external_test(te_n, nl_hint=[(r["nl"] or "").strip() for r in raw["test"]])
    if ext is None:
        return sp, None
    sp["test"] = ext
    print(f"[concode] using the labelled test set from {src} ({len(ext)} rows)", flush=True)
    return sp, str(src)


def preflight_concode():
    """Cheap early check used by the launcher: is a labelled CONCODE test set available (or the fallback explicitly allowed)?"""
    sp, src = _concode_splits(1000)
    if src is not None:
        print(f"[launcher] CONCODE test set OK ({src}).", flush=True)
        return True
    if os.environ.get("UCF_ALLOW_VAL_FALLBACK"):
        print("[launcher] WARNING: no labelled CONCODE test set; the explicitly allowed validation-carved fallback will be used "
              "(results NOT comparable with the paper's CONCODE test).", flush=True)
        return True
    print("[launcher] STOP: no labelled CONCODE test set found (the Hugging Face test split has empty code).  Fix one of:\n"
          "   a) give Internet access so the labelled mirror (huggingface.co AhmedSSoliman/CodeXGLUE-CONCODE test.json) can be fetched, or\n"
          "   b) download a CONCODE test.json WITH code (JSON lines, 2000 rows, fields nl+code; CodeXGLUE's own test.json and the\n"
          "      HF google/code_x_glue copy have it empty), add it to the notebook as a Kaggle dataset and set\n"
          "      os.environ['UCF_CONCODE_TEST'] = '/kaggle/input/<name>/test.json'  before running, or\n"
          "   c) accept a validation-carved test set on purpose (not comparable with the paper's CONCODE test): "
          "OPTIONS['allow_val_fallback'] = True", flush=True)
    return False


def load_data(ds):
    """Returns {'train','val','test'} lists of (source, target).
    NOTE: the paper's subsets come from NumPy RandomState(seed 0) after removing empty entries.  The exact
    draw order in ucf_revision_kaggle.py is not visible here; if you need byte-identical subsets, replace
    `_sample` calls below by the sampling code of that script (the replication check in `analyze` will tell
    you whether the 5 overlapping seeds agree with Table X)."""
    from datasets import load_dataset
    tr_n, va_n, te_n = DS[ds]["sizes"]
    if SMOKE:
        tr_n, va_n, te_n = 64, 32, 32
    rng = np.random.RandomState(0)
    ok = lambda a, b: bool(a and b and a.strip() and b.strip())
    if ds.startswith("concode_java"):
        sp, tsrc = _concode_splits(te_n)
        if tsrc is not None:                               # labelled test set found: same draws as before
            return dict(train=_sample(sp["train"], tr_n, rng), val=_sample(sp["validation"], va_n, rng),
                        test=_sample(sp["test"], te_n, rng),
                        note=None if tsrc == "HF test split" else f"test from {tsrc} (HF test split has empty code)")
        if not os.environ.get("UCF_ALLOW_VAL_FALLBACK"):
            raise RuntimeError(f"[{ds}] no labelled CONCODE test set (HF test split has empty code, external CodeXGLUE test.json not "
                               "reachable).  Provide it via UCF_CONCODE_TEST=/path/test.json, or deliberately allow the "
                               "validation-carved fallback with UCF_ALLOW_VAL_FALLBACK=1 (OPTIONS['allow_val_fallback']=True).")
        pool, need = sp["validation"], va_n + te_n
        if len(pool) < need:
            raise RuntimeError(f"[{ds}] test split has {len(sp['test'])} usable rows (need {te_n}) and the validation split "
                               f"only {len(pool)} (need {need}); cannot build the dataset. Check the dataset columns.")
        print(f"WARNING: [{ds}] no labelled test set ({len(sp['test'])} usable HF test rows). "
              f"Falling back: validation/test are disjoint slices of the validation split. Results are NOT comparable with the "
              f"paper's CONCODE test set; the records carry data_note.", flush=True)
        perm = rng.permutation(len(pool))
        va = [pool[i] for i in perm[:va_n]]
        te = [pool[i] for i in perm[va_n:need]]
        banned = {norm(b) for _, b in va + te}
        tr = _sample([x for x in sp["train"] if norm(x[1]) not in banned], tr_n, rng)
        return dict(train=tr, val=va, test=te, note="test carved from the validation split (HF test split unusable)")
    if ds == "mbpp_py":
        try:
            raw = load_dataset("google-research-datasets/mbpp", "full")
        except Exception:
            raw = load_dataset("mbpp", "full")
        fmt = lambda r: (r["text"].strip() + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"]),
                         r["code"].replace("\r\n", "\n").strip())
        k = 16 if SMOKE else None
        mb_tr = [fmt(r) for r in list(raw["train"])[:k]]
        va_raw = list(raw["validation"])[:k]
        mb_va = [fmt(r) for r in va_raw]
        val_meta = [dict(tests=list(r["test_list"]), setup=r.get("test_setup_code", "") or "") for r in va_raw]
        te = list(raw["test"])[:k]
        he = []
        try:                                    # HumanEval: second, out-of-distribution execution test (never trained on)
            he = list(load_dataset("openai/openai_humaneval")["test"])[:(8 if SMOKE else None)]
        except Exception as e:
            print(f"WARNING: HumanEval could not be loaded ({type(e).__name__}); transfer evaluation skipped", flush=True)
        held = ({norm(r["code"].replace("\r\n", "\n").strip()) for r in te} | {norm(b) for _, b in mb_va}
                | {norm(r["prompt"] + r["canonical_solution"]) for r in he} | {norm(r["canonical_solution"]) for r in he})
        alp = _dedup(_alpaca_rows(), held)
        alp = [alp[i] for i in rng.permutation(len(alp))[:(32 if SMOKE else 4000)]]
        # MBPP's own 374 training tasks are repeated x3 so they are not swamped by the instruction data
        train = mb_tr * (1 if SMOKE else 3) + alp
        ret = dict(train=train, val=mb_va, val_meta=val_meta, test=[fmt(r) for r in te],
                   test_meta=[dict(tests=list(r["test_list"]), setup=r.get("test_setup_code", "") or "") for r in te])
        if he:
            # source = signature + docstring (the HumanEval prompt); the generated function must pass check(entry_point).
            # Import lines of the prompt (e.g. `from typing import List`) are prepended so annotations do not fail.
            ret["test2"] = [(r["prompt"].strip(), r["canonical_solution"]) for r in he]
            ret["test2_meta"] = [dict(tests=[r["test"], f'check({r["entry_point"]})'],
                                      setup="\n".join(l for l in r["prompt"].splitlines() if l.startswith(("import ", "from ")))) for r in he]
        return ret
    if ds in ("csn_java", "csn_python"):
        raw = load_dataset("google/code_x_glue_ct_code_to_text", DS[ds]["lang"])
        def rows(k):
            return [(r["docstring"].strip(), r["code"].strip()) for r in raw[k]
                    if ok(r["docstring"], r["code"]) and len(r["docstring"].split()) >= 3
                    and len(r["code"]) <= MAX_CODE_CHARS]
        tr, va, te = _decontaminate(rows("train"), rows("validation"), rows("test"), ds)
        return dict(train=_sample(tr, tr_n, rng), val=_sample(va, va_n, rng), test=_sample(te, te_n, rng))
    if ds == "py_alpaca":
        raw = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
        rows = []
        for r in raw:
            src = (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")).strip()
            if ok(src, r["output"]) and len(r["output"]) <= MAX_CODE_CHARS:
                rows.append((src, r["output"].strip()))
        n0 = len(rows)
        rows = _dedup(rows)
        print(f"[py_alpaca] exact-duplicate code removed: {n0 - len(rows)}", flush=True)
        perm = rng.permutation(len(rows))
        pick = lambda a, b: [rows[i] for i in perm[a:b]]
        return dict(train=pick(0, tr_n), val=pick(tr_n, tr_n + va_n), test=pick(tr_n + va_n, tr_n + va_n + te_n))
    raise ValueError(ds)


# --------------------------------------------------------------------------------------
# 2. Metrics (normalisation, BLEU statistics, validity)
# --------------------------------------------------------------------------------------
_TOK = re.compile(r"\w+|[^\w\s]")


def norm(s):
    return " ".join(_TOK.findall(s))


def bleu_stats(hyp, ref):
    """Sufficient statistics of corpus BLEU for one example (strings already normalised; tokenize='none')."""
    h, r = hyp.split(), ref.split()
    out = np.zeros(10)
    for n in range(1, 5):
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        out[n - 1] = sum(min(c, rc[g]) for g, c in hc.items())
        out[3 + n] = max(len(h) - n + 1, 0)
    out[8], out[9] = len(h), len(r)
    return out


def bleu_from_stats(St):
    """sacreBLEU formula (exp smoothing, brevity penalty). St: (K,10) summed statistics -> (K,) scores."""
    St = np.atleast_2d(St)
    res = np.zeros(len(St))
    for k, row in enumerate(St):
        m, t, hl, rl = row[:4], row[4:8], row[8], row[9]
        if m[0] == 0 or hl == 0:
            continue
        sm, lp, eff = 1.0, 0.0, 4
        for n in range(4):
            if t[n] == 0:                  # no n-grams of this order in any hypothesis: sacreBLEU's corpus score is ~0
                eff = 0
                break
            if m[n] == 0:
                sm *= 2
                p = 1.0 / (sm * t[n])
            else:
                p = m[n] / t[n]
            lp += math.log(p)
        if eff == 0:
            continue
        bp = 1.0 if hl >= rl else math.exp(1 - rl / hl)
        res[k] = 100 * bp * math.exp(lp / eff)
    return res


def valid_java(code):
    import javalang
    try:
        javalang.parse.parse("class Dummy { " + code + " }")
        return True
    except Exception:
        return False


def valid_py(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    if not tree.body:
        return False
    if all(isinstance(n, ast.Expr) and isinstance(n.value, (ast.Name, ast.Constant)) for n in tree.body):
        return False                       # strict: reject bare names / constants
    return True


_PASS = "__UCF_PASS_9f3a__"
_GUARD = ("import builtins, sys, os, shutil, subprocess, resource\n"
          "resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30)); resource.setrlimit(resource.RLIMIT_CPU, (10, 10))\n"
          "builtins.exit = builtins.quit = sys.exit = os._exit = os.system = os.remove = os.unlink = os.rmdir = os.kill = None\n"
          "shutil.rmtree = None; subprocess.Popen = None\n")


def _run_one(prog, timeout):
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=timeout, cwd=d)
        return bool(r.returncode == 0 and _PASS in r.stdout)
    except Exception:
        return False


def exec_pass(preds, meta, timeout=10, workers=4):
    """Execution-based correctness: a prediction passes only if every assertion of the task runs without error.
    Each program runs in its own subprocess (time, CPU and memory limits, exit/os hooks disabled) and must reach a
    sentinel print.  This runs model-generated code: use it only inside a disposable container such as Kaggle."""
    from concurrent.futures import ThreadPoolExecutor
    progs = [_GUARD + (m.get("setup") or "") + "\n" + c + "\n" + "\n".join(m["tests"]) + f"\nprint('{_PASS}')\n"
             for c, m in zip(preds, meta)]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda q: _run_one(q, timeout), progs))


def score(preds, refs, lang):
    if len(preds) == 0 or len(preds) != len(refs):
        raise ValueError(f"score(): {len(preds)} predictions vs {len(refs)} references (empty or mismatched test set)")
    hn, rn = [norm(p) for p in preds], [norm(r) for r in refs]
    vf = valid_java if lang == "java" else valid_py
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hn, [rn], tokenize="none").score
        chrf = sacrebleu.corpus_chrf(hn, [rn]).score
    except ImportError:                    # same formula as sacreBLEU (checked by --stage selftest); chrF unavailable
        print("WARNING: sacrebleu missing; using the built-in BLEU (identical formula), chrF=None", flush=True)
        bleu = float(bleu_from_stats(sum(bleu_stats(h, r) for h, r in zip(hn, rn)))[0])
        chrf = None
    out = dict(bleu=bleu, chrf=chrf,
               em=100.0 * float(np.mean([a == b for a, b in zip(hn, rn)])),
               valid=100.0 * float(np.mean([vf(p) for p in preds])),
               distinct=100.0 * len(set(hn)) / max(len(hn), 1), codebleu=None)
    out["codebleu"] = codebleu_score(preds, refs, lang)
    return out


_CB_WARNED = False


def codebleu_score(preds, refs, lang):
    """CodeBLEU in [0,100], or None if the package is missing/broken (never fatal: run `--stage codebleu` later to
    backfill from the saved predictions once the package works)."""
    global _CB_WARNED
    try:
        from codebleu import calc_codebleu
        return 100.0 * calc_codebleu([[r] for r in refs], list(preds), lang=lang)["codebleu"]
    except Exception as e:
        if not _CB_WARNED:
            _CB_WARNED = True
            print(f"WARNING: CodeBLEU unavailable ({type(e).__name__}: {str(e)[:100]}); stored as None. "
                  "Backfill later with --stage codebleu.", flush=True)
        return None


# --------------------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------------------
def hub_call(fn, *a, **kw):
    """fn(*a, **kw) from the local cache if possible, otherwise from the Hub with exponential back-off on HTTP 429 /
    connection errors (two workers loading the same checkpoint at once used to trigger the rate limit)."""
    try:
        return fn(*a, local_files_only=True, **kw)
    except Exception:
        pass
    last = None
    for k in range(6):
        try:
            return fn(*a, **kw)
        except Exception as e:
            m = str(e).lower()
            if "429" in m or "too many requests" in m or "connection" in m or "timed out" in m or "timeout" in m:
                wait = 20 * 2 ** k
                print(f"   Hub busy ({type(e).__name__}); retry {k + 1}/6 in {wait}s", flush=True)
                time.sleep(wait)
                last = e
                continue
            raise
    raise last


def get_tok(key, lang):
    from transformers import AutoTokenizer
    kw = dict(revision=REV[CK[key]]) if CK[key] in REV else {}
    if key == "plbart":
        return hub_call(AutoTokenizer.from_pretrained, CK[key], src_lang="en_XX", tgt_lang=lang, **kw)
    tok = hub_call(AutoTokenizer.from_pretrained, CK[key], **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained(cls, key, **kw):
    if CK[key] in REV:
        kw["revision"] = REV[CK[key]]
    return hub_call(cls.from_pretrained, CK[key], **kw)


class Enc(nn.Module):
    """Frozen encoder (fp32, eval mode, no grad)."""
    def __init__(s, key, kind):
        super().__init__()
        from transformers import AutoModel, T5EncoderModel, PLBartModel
        if kind == "t5":
            s.m = _from_pretrained(T5EncoderModel, key)
        elif kind == "plbart":
            s.m = _from_pretrained(PLBartModel, key).get_encoder()
        else:
            s.m = _from_pretrained(AutoModel, key)
        for p in s.m.parameters():
            p.requires_grad = False
        s.dim = s.m.config.hidden_size
        s.m.eval()

    def train(s, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(s, ids, mask):
        return s.m(input_ids=ids, attention_mask=mask).last_hidden_state


class DecS2S(nn.Module):
    def __init__(s, name, key):
        super().__init__()
        from transformers import PLBartForConditionalGeneration, T5ForConditionalGeneration
        s.m = _from_pretrained(PLBartForConditionalGeneration if name == "plbart" else T5ForConditionalGeneration, key)
        s.dim = s.m.config.hidden_size


class DecCLM(nn.Module):
    def __init__(s, key):
        super().__init__()
        from transformers import GPT2LMHeadModel
        s.m = _from_pretrained(GPT2LMHeadModel, key)
        s.dim = s.m.config.n_embd


class Block(nn.Module):
    def __init__(s, d, heads, ff, drop):
        super().__init__()
        s.lq, s.lk, s.lf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(s, q, x, pad):
        k = s.lk(x)
        a, _ = s.att(s.lq(q), k, k, key_padding_mask=pad, need_weights=False)
        q = q + a
        return q + s.ff(s.lf(q))


class Resampler(nn.Module):
    """Input projection (d_in -> d) + 2 Perceiver-style blocks (8 heads, FF 4d); 14.8 M parameters at d = 768."""
    def __init__(s, d_in, d, nq, blocks=2, heads=8, drop=0.1):
        super().__init__()
        s.proj = nn.Linear(d_in, d)
        s.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        s.blocks = nn.ModuleList([Block(d, heads, 4 * d, drop) for _ in range(blocks)])
        s.out = nn.LayerNorm(d)

    def forward(s, h, m):
        x, pad = s.proj(h), (m == 0)
        q = s.q[None].expand(h.size(0), -1, -1)
        for b in s.blocks:
            q = b(q, x, pad)
        q = s.out(q)
        return q, torch.ones(q.shape[:2], dtype=torch.long, device=q.device)


class XBlock(nn.Module):
    """CALM-style gated cross-attention injected after a decoder layer:  x <- x + tanh(g) * CrossAttn(LN(x), z).
    z are the projected encoder states.  The gate starts at `gate0` (a small non-zero value so that the new attention
    receives gradient from step 0; Flamingo uses 0)."""
    def __init__(s, d, heads=None, drop=0.1, gate0=0.1):
        super().__init__()
        heads = heads or (12 if d % 12 == 0 else 8)
        s.ln = nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        s.g = nn.Parameter(torch.full((1,), float(gate0)))

    def forward(s, x, z, zm):
        a, _ = s.att(s.ln(x), z, z, key_padding_mask=(zm == 0), need_weights=False)
        return x + torch.tanh(s.g) * a


class Adapter(nn.Module):
    """Houlsby-style bottleneck adapter applied to a layer's output; the up-projection starts at zero (identity)."""
    def __init__(s, d, r):
        super().__init__()
        s.down, s.up = nn.Linear(d, r), nn.Linear(r, d)
        nn.init.zeros_(s.up.weight)
        nn.init.zeros_(s.up.bias)

    def forward(s, x):
        return x + s.up(F.gelu(s.down(x)))


def _swap0(out, f):
    return (f(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else f(out)


def attach_adapters(layers, adapters):
    return [L.register_forward_hook(lambda m, i, o, ad=ad: _swap0(o, ad)) for L, ad in zip(layers, adapters)]


def attach_calm(layers, ids, xb, ctx):
    """ctx = {'z': (B,S,d) projected encoder states, 'm': (B,S) mask}; set before every forward / generate call."""
    return [layers[i].register_forward_hook(
        lambda m, inp, o, j=j: _swap0(o, lambda h: xb[j](h, ctx["z"], ctx["m"]))) for j, i in enumerate(ids)]


def dec_layers(sy):
    m = sy.dec.m
    if sy.dkind == "clm":
        return list(m.transformer.h)
    return list(m.model.decoder.layers) if sy.spec["dec"] == "plbart" else list(m.decoder.block)


def enc_layers(sy):
    m = sy.dec.m
    return list(m.model.encoder.layers) if sy.spec["dec"] == "plbart" else list(m.encoder.block)


class Bridge(nn.Module):
    """kinds: naive | ln | linear | linear_noln | resampler | padtrunc | calm.
    d_in != d_out (dimension-mismatched pairs): naive / ln are undefined; 'padtrunc' (zero-pad or truncate, no
    parameters) is the non-learned reference, 'linear' is Linear(d_in, d_out) (+ LayerNorm(d_out)).
    'calm' (external baseline, CALM-style composition): Linear+LN projection of the encoder states, plus gated
    cross-attention blocks (`nx` of them, injected after every second decoder layer) that read those states; the
    decoder's own cross-attention / prefix receives only a one-position zero stub."""
    def __init__(s, kind, d_in, d_out, nq, gain=1.0, nx=0):
        super().__init__()
        s.kind, s.d_in, s.d_out = kind, d_in, d_out
        if kind in ("naive", "ln"):
            assert d_in == d_out, f"'{kind}' bridge needs equal hidden sizes; use 'padtrunc' as the non-learned reference"
        if kind in ("ln", "linear", "calm"):
            s.ln = nn.LayerNorm(d_out)
            nn.init.constant_(s.ln.weight, gain)       # GPT-2-style decoders: std of token embeddings
        if kind in ("linear", "linear_noln", "calm"):
            s.lin = nn.Linear(d_in, d_out)
        if kind == "calm":
            s.xb = nn.ModuleList([XBlock(d_out) for _ in range(nx)])
        if kind == "resampler":
            s.res = Resampler(d_in, d_out, nq)

    def forward(s, h, m):
        k = s.kind
        if k == "naive":
            return h, m
        if k == "padtrunc":
            return (F.pad(h, (0, s.d_out - s.d_in)) if s.d_in < s.d_out else h[..., :s.d_out]), m
        if k == "ln":
            return s.ln(h), m
        if k in ("linear", "calm"):
            return s.ln(s.lin(h)), m
        if k == "linear_noln":
            return s.lin(h), m
        return s.res(h, m)


def left_align(z, m):
    """Move padding of the soft prefix to the left (so it is masked and position ids start at 0)."""
    B, Sq, D = z.shape
    if Sq == 0:                                      # CALM-style systems use no prefix
        return z, m
    L = m.sum(1).long()
    ar = torch.arange(Sq, device=z.device)[None, :]
    idx = (ar - (Sq - L)[:, None]) % Sq
    return (torch.gather(z, 1, idx[:, :, None].expand(-1, -1, D)), (ar >= (Sq - L)[:, None]).long())


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


class System(nn.Module):
    def __init__(s, spec):
        super().__init__()
        s.spec, s.lang = spec, spec["lang"]
        s.native, s.tune = spec["enc"] is None, spec["tune"]
        s.calm = spec["bridge"] == "calm"
        s.adapters, s._hooks, s._ctx = nn.ModuleList(), [], {}
        d = spec["dec"]
        s.dkind = DEC_KIND[d]
        dkey = ckey(d, s.lang)
        s.dtok = get_tok(dkey, s.lang)
        s.dec = DecS2S(d, dkey) if s.dkind == "s2s" else DecCLM(dkey)
        s.amp = d not in ("codet5", "codet5_small", "codet5p")                      # T5 decoder runs in fp32 (fp16 overflow risk)
        if not s.native:
            e = spec["enc"]
            s.etok = get_tok(ckey(e, s.lang), s.lang)
            s.enc = Enc(ckey(e, s.lang), ENC_KIND[e])
            gain = float(s.dec.m.transformer.wte.weight.detach().std()) if s.dkind == "clm" else 1.0
            nx = 0
            if s.calm:
                s._calm_ids = list(range(1, len(dec_layers(s)), 2)) or [0]     # after every second decoder layer
                nx = len(s._calm_ids)
            s.bridge = Bridge(spec["bridge"], s.enc.dim, s.dec.dim, spec["nq"], gain, nx)
            if s.calm:
                s._hooks += attach_calm(dec_layers(s), s._calm_ids, s.bridge.xb, s._ctx)
        s._peft = []
        s._setup_tuning()

    # ---- parameter freezing / LoRA -------------------------------------------------
    def _setup_tuning(s):
        m = s.dec.m
        if s.dkind == "s2s" and not s.native:        # the unused encoder half of the decoder checkpoint
            dec_ids = {id(p) for p in m.get_decoder().parameters()}
            for p in m.get_encoder().parameters():
                if id(p) not in dec_ids:
                    p.requires_grad = False
        if s.tune == "frozen":
            for p in m.parameters():
                p.requires_grad = False
        elif s.tune.startswith("lora"):
            from peft import LoraConfig, get_peft_model
            rk = int(s.tune[4:] or 16)                        # "lora" = rank 16 (as in the paper); "lora64" = rank 64, ...
            assert s.spec["dec"] == "plbart", "LoRA is implemented for PLBART only"
            tm = ["q_proj", "v_proj"] if s.native else \
                r".*decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|v_proj)"
            s._peft.append(get_peft_model(m, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.05, target_modules=tm)))
            assert any("lora_" in n for n, _ in m.named_parameters()), "LoRA was not injected"
        elif s.tune.startswith("adapter"):                  # "adapter64" = bottleneck 64; all base weights frozen
            r = int(s.tune[7:] or 64)
            for p in m.parameters():
                p.requires_grad = False
            layers = dec_layers(s) + (enc_layers(s) if (s.native and s.dkind == "s2s") else [])
            s.adapters.extend([Adapter(s.dec.dim, r) for _ in layers])
            s._hooks += attach_adapters(layers, s.adapters)

    def train(s, mode=True):
        super().train(mode)
        if not s.native:
            s.enc.eval()
        if s.tune == "frozen":
            s.dec.eval()
        return s

    def param_groups(s):
        g = []
        lr_d = HP["lr_bridge"] if s.tune.startswith("lora") else HP["lr_dec"]
        if not s.native:
            b = list(s.bridge.parameters())
            if b:
                g.append(dict(params=b, lr=HP["lr_bridge"]))
        d = [p for p in s.dec.m.parameters() if p.requires_grad]
        if d:
            g.append(dict(params=d, lr=lr_d))
        a = list(s.adapters.parameters())
        if a:
            g.append(dict(params=a, lr=HP["lr_bridge"]))
        return g

    def count_params(s):
        ps = []
        if not s.native:
            ps += list(s.enc.parameters()) + list(s.bridge.parameters())
        m = s.dec.m
        if s.dkind == "s2s" and not s.native:
            dec_ids = {id(p) for p in m.get_decoder().parameters()}
            enc_only = {id(p) for p in m.get_encoder().parameters()} - dec_ids
            ps += [p for p in m.parameters() if id(p) not in enc_only]
        else:
            ps += list(m.parameters())
        ps += list(s.adapters.parameters())
        u = {id(p): p for p in ps}.values()
        return sum(p.numel() for p in u), sum(p.numel() for p in u if p.requires_grad)

    # ---- tokenisation -----------------------------------------------------------------
    def tokenize(s, srcs, tgts=None, max_tgt=None):
        max_tgt = max_tgt or HP["max_tgt"]
        if s.native and s.dkind == "clm":
            return s._native_clm(srcs, tgts, max_tgt)
        tk = s.dtok if s.native else s.etok
        e = tk(list(srcs), max_length=HP["max_src"], truncation=True, padding=True, return_tensors="pt")
        b = {"src": e["input_ids"], "smask": e["attention_mask"]}
        if tgts is not None:
            b.update(s._tgt(tgts, max_tgt))
        return b

    def _native_clm(s, srcs, tgts, L):
        """Native decoder-only baseline: [NL tokens][BOS][code][EOS]; loss on code+EOS only.
        'src'/'smask' hold the left-padded prompt (NL + BOS) used for generation."""
        tk = s.dtok
        bos = tk.bos_token_id if tk.bos_token_id is not None else tk.eos_token_id
        eos, pad = tk.eos_token_id, tk.pad_token_id
        sid = tk(list(srcs), add_special_tokens=False, truncation=True, max_length=HP["max_src"])["input_ids"]
        n, G = len(sid), max(len(x) for x in sid) + 1
        gs, gm = torch.full((n, G), pad), torch.zeros((n, G), dtype=torch.long)
        for i, x in enumerate(sid):
            q = x + [bos]
            gs[i, G - len(q):] = torch.tensor(q)
            gm[i, G - len(q):] = 1
        b = {"src": gs, "smask": gm}
        if tgts is not None:
            cid = tk(list(tgts), add_special_tokens=False, truncation=True, max_length=L - 1)["input_ids"]
            T = max(len(x) + len(c) + 1 for x, c in zip(sid, cid))
            inp, lab = torch.full((n, T), pad), torch.full((n, T), -100)
            tm = torch.zeros((n, T), dtype=torch.long)
            for i, (x, c) in enumerate(zip(sid, cid)):
                a, m_ = x + [bos] + c, [-100] * len(x) + c + [eos]
                inp[i, :len(a)], lab[i, :len(m_)], tm[i, :len(a)] = torch.tensor(a), torch.tensor(m_), 1
            b.update({"inp": inp, "lab": lab, "tmask": tm})
        return b

    def _tgt(s, tgts, L):
        tk = s.dtok
        if s.dkind == "s2s":
            t = tk(text_target=list(tgts), max_length=L, truncation=True, padding=True, return_tensors="pt")
            return {"labels": t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)}
        ids = tk(list(tgts), add_special_tokens=False, truncation=True, max_length=L - 1)["input_ids"]
        bos = tk.bos_token_id if tk.bos_token_id is not None else tk.eos_token_id
        eos, pad = tk.eos_token_id, tk.pad_token_id
        T = max(len(x) for x in ids) + 1
        inp = torch.full((len(ids), T), pad)
        lab = torch.full((len(ids), T), -100)
        tm = torch.zeros((len(ids), T), dtype=torch.long)
        for i, x in enumerate(ids):
            n = len(x) + 1
            inp[i, :n] = torch.tensor([bos] + x)
            lab[i, :n] = torch.tensor(x + [eos])
            tm[i, :n] = 1
        return {"inp": inp, "lab": lab, "tmask": tm}

    def ref_len(s, tgts):
        if s.dkind == "s2s":
            return [len(x) for x in s.dtok(text_target=list(tgts))["input_ids"]]
        return [len(x) + 1 for x in s.dtok(list(tgts), add_special_tokens=False)["input_ids"]]

    # ---- forward ----------------------------------------------------------------------
    def _ac(s):
        return torch.autocast("cuda", dtype=torch.float16, enabled=s.amp)

    def bridge_out(s, b):
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            h = s.enc(b["src"], b["smask"]).float()
        with s._ac():
            z, zm = s.bridge(h, b["smask"])
        if not s.calm:
            return z, zm
        s._ctx["z"], s._ctx["m"] = z, zm                  # read by the injected cross-attention hooks
        B = z.size(0)
        if s.dkind == "s2s":                               # the decoder's own cross-attention sees a one-position zero stub
            return (torch.zeros(B, 1, z.size(-1), device=z.device, dtype=z.dtype),
                    torch.ones(B, 1, dtype=torch.long, device=z.device))
        return z[:, :0], zm[:, :0]                         # decoder-only: no soft prefix

    def nll(s, b):
        from transformers.modeling_outputs import BaseModelOutput
        if s.native and s.dkind == "clm":
            pos = (b["tmask"].cumsum(1) - 1).clamp(min=0)
            with s._ac():
                lg = s.dec.m(input_ids=b["inp"], attention_mask=b["tmask"], position_ids=pos).logits
            lab = b["lab"]
        elif s.native:
            with s._ac():
                lg = s.dec.m(input_ids=b["src"], attention_mask=b["smask"], labels=b["labels"]).logits
            lab = b["labels"]
        else:
            z, zm = s.bridge_out(b)
            if s.dkind == "s2s":
                with s._ac():
                    lg = s.dec.m(encoder_outputs=BaseModelOutput(last_hidden_state=z), attention_mask=zm,
                                 labels=b["labels"]).logits
                lab = b["labels"]
            else:
                zl, ml = left_align(z, zm)
                x = torch.cat([zl, s.dec.m.transformer.wte(b["inp"])], 1)
                mk = torch.cat([ml, b["tmask"]], 1)
                pos = (mk.cumsum(1) - 1).clamp(min=0)
                with s._ac():                              # lm_head only on the target positions (the soft prefix needs no logits:
                    hs = s.dec.m.transformer(inputs_embeds=x, attention_mask=mk, position_ids=pos).last_hidden_state
                    lg = s.dec.m.lm_head(hs[:, zl.size(1):])   # saves ~1/3 of the logits memory; identical loss
                lab = b["lab"]
        tot = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), lab.reshape(-1), ignore_index=-100, reduction="sum")
        return tot, (lab != -100).sum()

    @torch.no_grad()
    def gen(s, b, max_new, min_new=None):
        from transformers.modeling_outputs import BaseModelOutput
        kw = dict(max_new_tokens=max_new, num_beams=1, do_sample=False)
        if min_new:
            kw["min_new_tokens"] = min_new
        if s.dkind == "s2s":
            if s.spec["dec"] == "plbart":
                kw["decoder_start_token_id"] = s.dtok.convert_tokens_to_ids(f"__{s.lang}__")
            if s.native:
                with s._ac():
                    return s.dec.m.generate(input_ids=b["src"], attention_mask=b["smask"], **kw)
            z, zm = s.bridge_out(b)
            with s._ac():
                return s.dec.m.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z), attention_mask=zm, **kw)
        if s.native:
            tk = s.dtok
            with s._ac():
                o = s.dec.m.generate(input_ids=b["src"], attention_mask=b["smask"], pad_token_id=tk.pad_token_id,
                                     eos_token_id=tk.eos_token_id, **kw)
            return o[:, b["src"].size(1):]
        z, zm = s.bridge_out(b)
        zl, ml = left_align(z, zm)
        tk = s.dtok
        bos = tk.bos_token_id if tk.bos_token_id is not None else tk.eos_token_id
        B = zl.size(0)
        x = torch.cat([zl, s.dec.m.transformer.wte(torch.full((B, 1), bos, device=zl.device))], 1)
        mk = torch.cat([ml, torch.ones(B, 1, dtype=torch.long, device=zl.device)], 1)
        with s._ac():
            return s.dec.m.generate(inputs_embeds=x, attention_mask=mk, pad_token_id=tk.pad_token_id,
                                    eos_token_id=tk.eos_token_id, **kw)

    @torch.no_grad()
    def generate_all(s, srcs, dev, bs=32, max_new=None, min_new=None):
        s.eval()
        out = []
        for i in range(0, len(srcs), bs):
            b = to_dev(s.tokenize(srcs[i:i + bs]), dev)
            out += s.dtok.batch_decode(s.gen(b, max_new or HP["max_new"], min_new), skip_special_tokens=True)
        return out

    @torch.no_grad()
    def eval_bpc(s, data, dev, max_tgt=None, bs=16):
        """Bits per character.  Numerator: NLL (bits) of the reference tokens kept after truncation to max_tgt
        tokens (incl. EOS / language id).  Denominators: 'full' = characters of the untruncated reference
        (the paper's definition) and 'trunc' = characters of the decoded, truncated reference."""
        s.eval()
        max_tgt = max_tgt or HP["max_tgt"]
        nll = cf = ct = 0.0
        ntr = 0
        for i in range(0, len(data), bs):
            ch = data[i:i + bs]
            srcs, tgts = [c[0] for c in ch], [c[1] for c in ch]
            b = to_dev(s.tokenize(srcs, tgts, max_tgt), dev)
            tot, _ = s.nll(b)
            nll += float(tot)
            cf += sum(len(t) for t in tgts)
            lab = b["labels"] if s.dkind == "s2s" else b["lab"]
            for row in lab.tolist():
                ct += len(s.dtok.decode([t for t in row if t != -100], skip_special_tokens=True))
            ntr += sum(1 for n in s.ref_len(tgts) if n > max_tgt)
        bits = nll / math.log(2)
        return dict(full=bits / cf, trunc=bits / max(ct, 1), frac_trunc=ntr / len(data), max_tgt=max_tgt)


# --------------------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(en):
    try:
        return torch.amp.GradScaler("cuda", enabled=en)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=en)


def make_opt(s, total):
    opt = torch.optim.AdamW(s.param_groups(), weight_decay=HP["wd"])
    w = max(1, int(HP["warm"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / w if t < w else max(0.0, (total - t) / max(total - w, 1)))
    return opt, sched


def _split_batch(b, k):
    B = next(iter(b.values())).size(0)
    cs = -(-B // k)
    return [{n: v[i:i + cs] for n, v in b.items()} for i in range(0, B, cs)]


def train_step(s, opt, sched, scaler, b, params):
    """One optimiser step on batch b.  On a CUDA out-of-memory error the batch is split into 2, 4, ... micro-batches whose
    gradients are accumulated (each micro-batch loss divided by the token count of the WHOLE batch), which gives the same
    gradient as the full batch; the split factor is remembered per system (s._mb) and stored in the run record."""
    B = next(iter(b.values())).size(0)
    n_all = (b["labels" if s.dkind == "s2s" else "lab"] != -100).sum().clamp(min=1)
    k = max(1, int(getattr(s, "_mb", 1)))
    while True:
        oom, lval = False, 0.0
        tot = loss = cb = _n = None
        try:
            opt.zero_grad(set_to_none=True)
            for cb in (_split_batch(b, k) if k > 1 else [b]):
                tot, _n = s.nll(cb)
                loss = tot / n_all
                scaler.scale(loss).backward()
                lval += float(loss.detach())
                tot = loss = _n = None
        except torch.cuda.OutOfMemoryError:
            oom = True
        if not oom:
            break
        # Outside the except block (the traceback no longer pins the tensors) AND with the loop's own references cleared:
        # after an OOM inside backward(), `tot` / `loss` still held the whole autograd graph of the failed micro-batch, so
        # every retry ran with that graph alive and failed again, down to micro-batches of 1.
        tot = loss = cb = _n = None
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        dv = next(s.parameters()).device
        if k >= B:
            raise torch.cuda.OutOfMemoryError(
                f"CUDA out of memory even with micro-batches of 1 (batch {B}); allocated "
                f"{torch.cuda.memory_allocated(dv) / 2 ** 30:.2f} GB, reserved {torch.cuda.memory_reserved(dv) / 2 ** 30:.2f} GB")
        k = min(k * 2, B)
        s._mb = k
        print(f"   CUDA out of memory: retrying this step with {k} micro-batches (same gradient, slower); "
              f"allocated after cleanup {torch.cuda.memory_allocated(dv) / 2 ** 30:.2f} GB", flush=True)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, HP["clip"])
    scaler.step(opt)
    scaler.update()
    sched.step()
    return lval


def env_info():
    d = dict(python=sys.version.split()[0], numpy=np.__version__, revs_pinned=bool(REV), script=SCRIPT_VERSION)
    if HAVE_TORCH:
        d["torch"] = torch.__version__
        for mod in ("transformers", "peft", "sacrebleu"):
            try:
                d[mod] = __import__(mod).__version__
            except Exception:
                pass
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
    return d


def _ckpt_file(spec, seed):
    return ROOT / "ckpt" / f"run_{_h(sys_id(spec), seed)}.pt"


def _trainable(s):
    return {n: p for n, p in s.named_parameters() if p.requires_grad}


def save_ckpt(path, s, opt, sched, scaler, rng, perm, nxt, step, losses, ep_bpc, elapsed, resumes, meta):
    """Atomic training checkpoint: trainable weights only (frozen weights are re-loaded from the hub), optimizer,
    scheduler, loss scaler, data-order state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(dict(meta=meta, weights={n: p.detach().cpu() for n, p in _trainable(s).items()}, opt=opt.state_dict(),
                    sched=sched.state_dict(), scaler=scaler.state_dict(), rng=rng.get_state(), perm=perm, next=nxt,
                    step=step, losses=losses, ep_bpc=ep_bpc, elapsed=elapsed, resumes=resumes), tmp)
    os.replace(tmp, path)


def load_ckpt(path, meta):
    if not path.exists():
        return None
    try:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
        if ck.get("meta") != meta:
            raise ValueError("checkpoint belongs to a different configuration")
        return ck
    except Exception as e:
        print(f"   checkpoint unusable ({type(e).__name__}: {str(e)[:90]}); this run restarts from step 0", flush=True)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def ckpt_clear(j):
    if j.get("type") == "train":
        try:
            _ckpt_file(j["spec"], j["seed"]).unlink()
        except OSError:
            pass


def run_train(spec, seed, data, dev, stage, max_steps=None, screen_lens=None, eval_epochs=False):
    seed_all(seed)
    s = System(spec).to(dev)
    s.train()
    total_p, train_p = s.count_params()
    train = data["train"]
    spe = math.ceil(len(train) / HP["bs"])
    total = max_steps or spec["epochs"] * spe
    if SMOKE:
        total = min(total, 6)
    params = [p for g in s.param_groups() for p in g["params"]]
    trainable = len(params) > 0
    if trainable:
        opt, sched = make_opt(s, total)
    scaler = make_scaler(s.amp)
    rng = np.random.RandomState(seed)
    step, losses, ep_bpc = 0, [], []
    perm, start, elapsed0, resumes = None, 0, 0.0, []
    ck_on = bool(trainable and not screen_lens and not SMOKE and CKPT_INTERVAL_S > 0 and total >= CKPT_MIN_STEPS)
    ck_file = _ckpt_file(spec, seed)
    meta = dict(sys_id=sys_id(spec), seed=seed, total=total, bs=HP["bs"], n_train=len(train), lr=(HP["lr_dec"], HP["lr_bridge"]))
    if ck_on:
        ck = load_ckpt(ck_file, meta)
        if ck is not None:
            try:                                           # states first (cannot damage the model), weights last
                cur = _trainable(s)
                if set(cur) != set(ck["weights"]):
                    raise ValueError("parameter set differs")
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                scaler.load_state_dict(ck["scaler"])
                with torch.no_grad():
                    for n, p in cur.items():
                        p.copy_(ck["weights"][n])
                rng.set_state(ck["rng"])
                perm, start, step = ck["perm"], ck["next"], ck["step"]
                losses, ep_bpc, elapsed0 = ck["losses"], ck["ep_bpc"], ck["elapsed"]
                resumes = list(ck.get("resumes", [])) + [step]
                seed_all(seed * 7919 + step)               # fresh dropout stream after the restart
                print(f"   resumed from checkpoint at step {step}/{total}", flush=True)
            except Exception as e:
                try:
                    ck_file.unlink()
                except OSError:
                    pass
                raise RuntimeError(f"checkpoint could not be applied ({type(e).__name__}: {str(e)[:90]}); it was deleted, "
                                   "the run is retried from step 0") from e
            del ck
    t0 = time.time()
    last_ck = t0
    _last_prog = t0
    while trainable and step < total:
        if perm is None:
            perm = rng.permutation(len(train))
            start = 0
        for i in range(start, len(perm), HP["bs"]):
            ch = [train[j] for j in perm[i:i + HP["bs"]]]
            b = to_dev(s.tokenize([c[0] for c in ch], [c[1] for c in ch]), dev)
            l = train_step(s, opt, sched, scaler, b, params)
            step += 1
            if step % 50 == 0:
                losses.append((step, l))
                if time.time() - _last_prog >= 300:               # a progress line every 5 min so a long run is not silent
                    _last_prog = time.time()
                    print(f"   step {step}/{total} | loss {l:.4f} | {(time.time() - t0) / 60:.1f} min | "
                          f"eta {(time.time() - t0) / max(step - (resumes[-1] if resumes else 0), 1) * (total - step) / 60:.0f} min", flush=True)
            if ck_on and step < total and time.time() - last_ck >= CKPT_INTERVAL_S:
                save_ckpt(ck_file, s, opt, sched, scaler, rng, perm, i + HP["bs"], step, losses, ep_bpc,
                          elapsed0 + time.time() - t0, resumes, meta)
                last_ck = time.time()
            if step >= total:
                break
        if eval_epochs and step % spe == 0:
            ep_bpc.append((step // spe, s.eval_bpc(data["val"], dev)["full"]))
            s.train()
        perm = None
    if ck_on and step >= total:                            # training finished: a crash during generation / execution
        save_ckpt(ck_file, s, opt, sched, scaler, rng, None, 0, step, losses, ep_bpc, elapsed0 + time.time() - t0, resumes, meta)
    train_s = elapsed0 + (time.time() - t0)
    rec = dict(spec=spec, sys_id=sys_id(spec), seed=seed, stage=stage, steps=step, total_M=total_p / 1e6,
               trainable_M=train_p / 1e6, train_s=train_s, loss_curve=losses, val_bpc_epochs=ep_bpc, env=env_info(),
               resumes=resumes, data_note=data.get("note"), micro_batches=int(getattr(s, "_mb", 1)))
    if screen_lens:
        rec.update(kind="screen", enc=spec["enc"], dec=spec["dec"],
                   bpc={str(L): s.eval_bpc(data["val"], dev, L) for L in screen_lens})
    else:
        rec["kind"] = "train"
        rec["val_bpc"] = s.eval_bpc(data["val"], dev)["full"]
        test = data["test"]
        if not test:
            raise ValueError("empty test set (see the data loader message above)")
        t1 = time.time()
        preds = s.generate_all([c[0] for c in test], dev, max_new=16 if SMOKE else None)
        rec["gen_s"] = time.time() - t1
        refs = [c[1] for c in test]
        rec.update(preds=preds, refs=refs, metrics=score(preds, refs, spec["lang"]))
        if data.get("test_meta"):
            rec["passed"] = exec_pass(preds, data["test_meta"])
            rec["metrics"]["pass1"] = 100.0 * float(np.mean(rec["passed"]))
        if data.get("test2"):                               # HumanEval transfer (same model, no extra training)
            p2 = s.generate_all([c[0] for c in data["test2"]], dev, max_new=16 if SMOKE else None)
            rec["he_preds"], rec["he_passed"] = p2, exec_pass(p2, data["test2_meta"])
            rec["metrics"]["he_pass1"] = 100.0 * float(np.mean(rec["he_passed"]))
    del s
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def run_until_pass(spec, seed, data, dev, stage, cfg):
    """Train epoch by epoch; after every epoch generate code for the MBPP VALIDATION tasks and execute their unit tests.
    Stop when (a) validation pass@1 has not improved for `patience` epochs (ties broken by lower validation BPC), (b) the
    optional validation target is reached, or (c) the epoch cap is hit; never before `min_epochs`.  The weights of the best
    validation epoch are restored and only then are the test set and HumanEval run, once.  Checkpointed every epoch."""
    assert data.get("val_meta"), "run_until_pass needs a dataset with executable validation tests (mbpp_py)"
    seed_all(seed)
    max_ep, min_ep, pat, target = cfg["max_epochs"], cfg["min_epochs"], cfg["patience"], cfg["target"]
    if SMOKE:
        max_ep, min_ep, pat = 2, 1, 1
    s = System(spec).to(dev)
    s.train()
    total_p, train_p = s.count_params()
    train, bs = data["train"], HP["bs"]
    spe = math.ceil(len(train) / bs)
    params = [p for g in s.param_groups() for p in g["params"]]
    assert params, "nothing to train"
    opt, sched = make_opt(s, max_ep * spe)                # LR horizon = the cap (early stopping simply ends it sooner)
    scaler = make_scaler(s.amp)
    rng = np.random.RandomState(seed)
    ep, hist, losses, best, best_ep, best_vp, best_vb, bad, stop = 0, [], [], None, 0, -1.0, float("inf"), 0, None
    elapsed0, resumes = 0.0, []
    ck_on = (not SMOKE) and CKPT_INTERVAL_S > 0
    ck_file = _ckpt_file(spec, seed)
    meta = dict(kind="until_pass", sys_id=sys_id(spec), seed=seed, max_ep=max_ep, min_ep=min_ep, pat=pat, target=target,
                bs=bs, n_train=len(train), lr=(HP["lr_dec"], HP["lr_bridge"]))
    if ck_on:
        ck = load_ckpt(ck_file, meta)
        if ck is not None:
            try:
                cur = _trainable(s)
                if set(cur) != set(ck["weights"]):
                    raise ValueError("parameter set differs")
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                scaler.load_state_dict(ck["scaler"])
                with torch.no_grad():
                    for n, p in cur.items():
                        p.copy_(ck["weights"][n])
                rng.set_state(ck["rng"])
                ep, hist, losses, best, best_ep = ck["ep"], ck["hist"], ck["losses"], ck["best"], ck["best_ep"]
                best_vp, best_vb, bad, stop, elapsed0 = ck["best_vp"], ck["best_vb"], ck["bad"], ck["stop"], ck["elapsed"]
                resumes = list(ck.get("resumes", [])) + [ep]
                seed_all(seed * 7919 + ep)
                print(f"   resumed after epoch {ep} (best epoch so far {best_ep}, val pass@1 {best_vp:.1f})", flush=True)
            except Exception as e:
                try:
                    ck_file.unlink()
                except OSError:
                    pass
                raise RuntimeError(f"checkpoint could not be applied ({type(e).__name__}: {str(e)[:90]}); it was deleted, "
                                   "the run is retried from step 0") from e
            del ck
    t0 = time.time()
    while stop is None:
        perm = rng.permutation(len(train))
        for i in range(0, len(perm), bs):
            ch = [train[j] for j in perm[i:i + bs]]
            b = to_dev(s.tokenize([c[0] for c in ch], [c[1] for c in ch]), dev)
            l = train_step(s, opt, sched, scaler, b, params)
            if (ep * spe + i // bs + 1) % 50 == 0:
                losses.append((ep * spe + i // bs + 1, l))
        ep += 1
        preds = s.generate_all([c[0] for c in data["val"]], dev, max_new=16 if SMOKE else None)
        vp = 100.0 * float(np.mean(exec_pass(preds, data["val_meta"])))
        vb = s.eval_bpc(data["val"], dev)["full"]
        s.train()
        hist.append((ep, vp, vb))
        if vp > best_vp + 1e-9 or (abs(vp - best_vp) <= 1e-9 and vb < best_vb - 1e-4):
            best, best_ep, best_vp, best_vb, bad = {n: p.detach().cpu().clone() for n, p in _trainable(s).items()}, ep, vp, vb, 0
        else:
            bad += 1
        print(f"   epoch {ep}: val pass@1 {vp:.1f} | val BPC {vb:.3f} | best {best_vp:.1f} @ep{best_ep} | no-improve {bad}", flush=True)
        if target is not None and vp >= target and ep >= min_ep:
            stop = "target_reached"
        elif ep >= min_ep and bad >= pat:
            stop = "patience"
        elif ep >= max_ep:
            stop = "max_epochs"
        if ck_on:
            save_ep_ckpt(ck_file, s, opt, sched, scaler, rng, ep, hist, losses, best, best_ep, best_vp, best_vb, bad, stop,
                         elapsed0 + time.time() - t0, resumes, meta)
    train_s = elapsed0 + (time.time() - t0)
    with torch.no_grad():                                  # restore the best-validation epoch
        cur = _trainable(s)
        for n, p in cur.items():
            p.copy_(best[n].to(p.device))
    s.eval()
    rec = dict(spec=spec, sys_id=sys_id(spec), seed=seed, stage=stage, steps=ep * spe, total_M=total_p / 1e6,
               trainable_M=train_p / 1e6, train_s=train_s, loss_curve=losses, kind="train", env=env_info(), resumes=resumes,
               val_bpc_epochs=[(e, b_) for e, _, b_ in hist], val_pass_curve=[(e, v) for e, v, _ in hist], epochs_run=ep,
               best_epoch=best_ep, best_val_pass1=best_vp, stop_reason=stop,
               protocol=dict(max_epochs=max_ep, min_epochs=min_ep, patience=pat, target=target, selection="validation pass@1, ties by validation BPC"))
    rec["val_bpc"] = s.eval_bpc(data["val"], dev)["full"]
    t1 = time.time()
    preds = s.generate_all([c[0] for c in data["test"]], dev, max_new=16 if SMOKE else None)
    rec["gen_s"] = time.time() - t1
    refs = [c[1] for c in data["test"]]
    rec.update(preds=preds, refs=refs, metrics=score(preds, refs, spec["lang"]))
    rec["passed"] = exec_pass(preds, data["test_meta"])
    rec["metrics"]["pass1"] = 100.0 * float(np.mean(rec["passed"]))
    if data.get("test2"):
        p2 = s.generate_all([c[0] for c in data["test2"]], dev, max_new=16 if SMOKE else None)
        rec["he_preds"], rec["he_passed"] = p2, exec_pass(p2, data["test2_meta"])
        rec["metrics"]["he_pass1"] = 100.0 * float(np.mean(rec["he_passed"]))
    del s
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def save_ep_ckpt(path, s, opt, sched, scaler, rng, ep, hist, losses, best, best_ep, best_vp, best_vb, bad, stop, elapsed, resumes, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(dict(meta=meta, weights={n: p.detach().cpu() for n, p in _trainable(s).items()}, opt=opt.state_dict(),
                    sched=sched.state_dict(), scaler=scaler.state_dict(), rng=rng.get_state(), ep=ep, hist=hist, losses=losses,
                    best=best, best_ep=best_ep, best_vp=best_vp, best_vb=best_vb, bad=bad, stop=stop, elapsed=elapsed,
                    resumes=resumes), tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# 5. Stages
# --------------------------------------------------------------------------------------
def stage_specs(stage, ds="concode_java", long_epochs=(6, 12)):
    """-> list of (spec, seeds, eval_epochs).  Used both to build jobs and to build the analysis tables."""
    P = lambda *a, **k: S(*a, ds=ds, **k)
    SM = SEEDS_MID
    out = []
    if stage == "seeds15":
        for sp in (P("codet5", "plbart", "naive"), P("codet5", "plbart", "linear"),
                   P("codet5", "plbart", "resampler"), P(None, "plbart")):
            out.append((sp, SEEDS15, False))
    elif stage == "seeds15_gpt":
        for sp in (P("codet5", "codegpt", "naive"), P("codet5", "codegpt", "linear"), P(None, "codegpt")):
            out.append((sp, SEEDS15, False))
    elif stage == "encoders":      # non-CodeT5 encoders (CodeT5+ = an encoder pretrained for composition); CodeT5 rows reuse seeds15 records
        for e in ("codet5", "codet5p", "unixcoder", "roberta", "codebert"):
            for br in ("naive", "linear"):
                out.append((P(e, "plbart", br), SM, False))
        for br in ("naive", "linear"):
            out.append((P("unixcoder", "codegpt", br), SM, False))
    elif stage == "frozen_dec":    # bridge-only: encoder AND decoder frozen
        out.append((P("codet5", "plbart", "naive", tune="frozen"), [13], False))       # zero-shot, deterministic
        out.append((P("codet5", "plbart", "linear", tune="frozen"), SM, False))
        out.append((P("codet5", "plbart", "linear", tune="frozen", epochs=10), SM, True))
        out.append((P("codet5", "plbart", "resampler", tune="frozen"), SM, False))
        out.append((P("codet5", "plbart", "calm", tune="frozen"), SM, False))          # CALM: both models frozen
        out.append((P("codet5", "codegpt", "naive", tune="frozen"), [13], False))
        out.append((P("codet5", "codegpt", "linear", tune="frozen"), SM, False))
        out.append((P("codet5", "plbart", "linear"), SM, False))                       # full-tune reference
    elif stage == "long_resampler":
        for ep in (3,) + tuple(long_epochs):
            for br in ("resampler", "linear"):
                out.append((P("codet5", "plbart", br, epochs=ep), SM, ep > 3))
    elif stage == "dimmismatch":   # encoder and decoder with different hidden sizes (512 <-> 768)
        for e, d in (("codet5_small", "plbart"), ("codet5", "codet5_small"), ("unixcoder", "codet5_small")):
            for br in ("padtrunc", "linear", "resampler"):
                out.append((P(e, d, br), SM, False))
        for br in ("naive", "linear"):                                                   # same-dimension control
            out.append((P("codet5_small", "codet5_small", br), SM, False))
        out.append((P(None, "codet5_small"), SM, False))
        out.append((P(None, "plbart"), SM, False))
    elif stage == "functional":    # execution-based pass@1 on MBPP (+ HumanEval transfer); ds must be mbpp_py
        M = lambda *a, **k: S(*a, ds="mbpp_py", epochs=5, **k)
        for sp in (M(None, "plbart"), M(None, "plbart", tune="lora"), M(None, "plbart", tune="adapter64"), M(None, "codegpt"),
                   M("codet5", "plbart", "naive"), M("codet5", "plbart", "linear"), M("codet5", "plbart", "resampler"),
                   M("codet5", "plbart", "calm"), M("plbart", "plbart", "naive"), M("plbart", "plbart", "linear"),
                   M("codet5", "codegpt", "naive"), M("codet5", "codegpt", "linear")):
            out.append((sp, SM, False))
    elif stage == "functional_best":   # chosen pair (+ matched native reference), trained under the early-stopping protocol
        Mx = lambda *a, **k: S(*a, ds="mbpp_py", epochs=FB["max_epochs"], **k)
        out.append((Mx(FB["enc"], FB["dec"], FB["bridge"]), SM, False))
        if FB["ref"]:
            out.append((Mx(None, FB["dec"]), SM, False))
    elif stage == "newdata":
        for sp in (P(None, "plbart"), P(None, "plbart", tune="lora"), P(None, "plbart", tune="adapter64"),
                   P("codet5", "plbart", "naive"), P("codet5", "plbart", "linear"),
                   P("codet5", "plbart", "resampler"), P("codet5", "plbart", "calm"), P("codet5", "plbart", "linear", tune="lora"),
                   P("plbart", "plbart", "naive"), P("plbart", "plbart", "linear"),
                   P("codet5", "codegpt", "naive"), P("codet5", "codegpt", "linear"), P(None, "codegpt")):
            out.append((sp, SM, False))
    elif stage == "baselines":     # external composition / parameter-efficient baselines (default dataset: CONCODE Java)
        for sp in (P("codet5", "plbart", "naive"), P("codet5", "plbart", "linear"), P("codet5", "plbart", "resampler"),
                   P(None, "plbart"), P(None, "plbart", tune="lora"), P("plbart", "plbart", "naive"),   # references: reuse seeds15 / newdata records
                   P("codet5", "plbart", "calm"), P("codet5", "plbart", "calm", tune="frozen"),  # CALM-style cross-attention composition
                   P("codet5", "codegpt", "naive"), P("codet5", "codegpt", "linear"), P("codet5", "codegpt", "calm"), P(None, "codegpt"),
                   P(None, "plbart", tune="lora4"), P(None, "plbart", tune="lora64"), P(None, "plbart", tune="lora256"),
                   P(None, "plbart", tune="adapter16"), P(None, "plbart", tune="adapter64"), P(None, "plbart", tune="adapter256"),
                   P("codet5", "plbart", "linear", tune="adapter64"),                           # bridge + adapter decoder
                   P("codet5p", "plbart", "naive"), P("codet5p", "plbart", "linear"), P(None, "codet5p")):   # CodeT5+ (pretrained for composition)
            out.append((sp, SM, False))
    elif stage == "scale":         # larger components; hidden sizes 1024 <-> 768 (naive coupling undefined -> pad/truncate)
        for sp in (P("codet5p770", "plbart", "padtrunc"), P("codet5p770", "plbart", "linear"), P("codet5p770", "plbart", "resampler"),
                   P("codet5p770", "plbart", "calm"), P("codet5", "gpt2m", "padtrunc"), P("codet5", "gpt2m", "linear"),
                   P("codet5", "gpt2m", "calm"), P(None, "gpt2m"), P(None, "plbart"), P("codet5", "plbart", "linear")):
            out.append((sp, SEEDS3, False))
    elif stage == "data40k":       # 4x training data (same validation / test subsets as the 10k study)
        for sp in (P("codet5", "plbart", "naive"), P("codet5", "plbart", "linear"), P("codet5", "plbart", "resampler"),
                   P(None, "plbart"), P(None, "plbart", tune="lora"), P("codet5", "plbart", "linear", tune="lora")):
            out.append((sp, SEEDS3, False))
    return out


def make_jobs(stage, args):
    jobs = []
    if stage == "bpc_trunc":
        for seed in args.seeds or SEEDS3:
            for e in ("codet5", "plbart", "codebert", "unixcoder", "roberta"):
                for d in ("codet5", "plbart", "codegpt", "gpt2"):
                    jobs.append(dict(type="screen", enc=e, dec=d, seed=seed, ds="concode_java"))
        return jobs
    ds_of = {"newdata": args.dataset or "csn_java", "baselines": args.dataset or "concode_java", "functional": "mbpp_py", "functional_best": "mbpp_py",
             "data40k": "concode_java_40k"}
    specs = stage_specs(stage, ds_of.get(stage, "concode_java"), tuple(args.long_epochs))
    seeds_over = args.seeds
    seed_union = sorted({sd for _, ss, _ in specs for sd in (seeds_over or ss)},
                        key=lambda x: (x not in SEEDS15, SEEDS15.index(x) if x in SEEDS15 else x))
    for seed in seed_union:                    # seed-major order: partial runs give complete seed blocks
        for sp, ss, ee in specs:
            if seed in (seeds_over or ss):
                jobs.append(dict(type="train", spec=sp, seed=seed, ds=sp["ds"], eval_epochs=ee, stage=stage))
    return jobs


def job_path(j):
    if j["type"] == "screen":
        return REC / f'screen_{_h(j["enc"], j["dec"], j["seed"])}.json'
    return REC / f'run_{_h(sys_id(j["spec"]), j["seed"])}.json'


# ---- crash-safe bookkeeping ------------------------------------------------------------
def atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path):
    atomic_bytes(path, json.dumps(obj).encode())


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except Exception:
        return True                                  # PermissionError etc.: it exists
    return True


def claim_stale(c):
    """A claim is stale if its owner is dead (same host, pid gone) or its heartbeat stopped STALE_S seconds ago."""
    try:
        age = time.time() - os.stat(c).st_mtime
    except FileNotFoundError:
        return True
    if age > STALE_S:
        return True
    try:
        info = json.load(open(c))
    except Exception:
        return False                                 # empty / being written by a racing worker: fresh
    pid = info.get("pid")
    return bool(info.get("host") == socket.gethostname() and pid and pid != os.getpid() and not _pid_alive(pid))


def claim(path):
    c = str(path) + ".claim"
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not claim_stale(c):
                return False
            try:                                     # re-check right before removing (another worker may just have taken over)
                if not claim_stale(c):
                    return False
                os.unlink(c)
                print(f"   took over the stale claim of {Path(c).name}", flush=True)
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname(), t=time.time()), f)
        return True
    return False


def release(path):
    try:
        os.unlink(str(path) + ".claim")
    except OSError:
        pass


class Heartbeat(threading.Thread):
    """Touches the claim file while a run is in progress so other workers can tell 'slow' from 'dead'."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.c, self.ev = str(path) + ".claim", threading.Event()

    def run(self):
        while not self.ev.wait(HEARTBEAT_S):
            try:
                os.utime(self.c)
            except OSError:
                pass

    def stop(self):
        self.ev.set()


def n_fails(path):
    try:
        return int(open(str(path) + ".fail").read().strip() or 0)
    except Exception:
        return 0


def bump_fail(path):
    n = n_fails(path) + 1                            # read BEFORE the file is opened for writing (which truncates it)
    try:
        with open(str(path) + ".fail", "w") as f:
            f.write(str(n))
    except OSError:
        pass


def clear_fails():
    for f in REC.glob("*.fail"):
        try:
            f.unlink()
        except OSError:
            pass


def _valid_record(d):
    return isinstance(d, dict) and d.get("kind") in ("train", "screen", "timing")


def verify_records():
    """Quarantine records that do not parse (e.g. written by an older version of this script and cut off by a crash), so the
    corresponding runs are redone; remove leftover temp files of dead writers."""
    bad = 0
    for f in sorted(REC.glob("*.json")):
        try:
            with open(f) as fh:
                ok = _valid_record(json.load(fh))
        except Exception:
            ok = False
        if not ok:
            try:
                f.rename(f.with_name(f.name + ".corrupt"))
                bad += 1
                print(f"   quarantined unreadable record {f.name} (its run will be redone)", flush=True)
            except OSError:
                pass
    for t in list(REC.glob("*.tmp*")) + list((ROOT / "ckpt").glob("*.tmp*")):
        try:
            if time.time() - t.stat().st_mtime > STALE_S:
                t.unlink()
        except OSError:
            pass
    return bad


def execute(j, data_cache, dev):
    if j["ds"] not in data_cache:
        data_cache[j["ds"]] = load_data(j["ds"])
    data = data_cache[j["ds"]]
    if j["type"] == "screen":
        sp = S(j["enc"], j["dec"], "resampler", tune="frozen", nq=32, ds=j["ds"])
        return run_train(sp, j["seed"], data, dev, "bpc_trunc", max_steps=400, screen_lens=BPC_LENS)
    if j["stage"] == "functional_best":
        return run_until_pass(j["spec"], j["seed"], data, dev, j["stage"], FB)
    return run_train(j["spec"], j["seed"], data, dev, j["stage"], eval_epochs=j["eval_epochs"])


def job_name(j):
    return (sys_id(j["spec"]) if j["type"] == "train" else f'screen {j["enc"]}>{j["dec"]}') + f' seed={j["seed"]}'


def _past_deadline():
    d = os.environ.get("UCF_DEADLINE")
    return bool(d) and time.time() > float(d)


_LAST_PACK = [0.0]


def _autopack(every_min=None):
    """Keep ucf_ext_records.zip fresh (records are small) so that a session that dies still leaves one file to carry over."""
    every_min = float(every_min if every_min is not None else os.environ.get("UCF_SYNC_EVERY", 20))
    if time.time() - _LAST_PACK[0] < every_min * 60:
        return
    _LAST_PACK[0] = time.time()
    try:
        pack_records(quiet=True)
        _sync_records(every_min=0)         # _autopack already applies the interval; avoids a seconds-long mismatch doubling it
    except Exception as e:
        print(f"[autopack] skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)


_LAST_SYNC = [0.0, -1, 0]          # time, record count at last successful upload, consecutive failures


def _kaggle_creds():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            os.environ[k] = us.get_secret(k)
    except Exception:
        pass


def _sync_records(force=False, every_min=None):
    """Upload ucf_ext_records.zip as a new version of a Kaggle dataset, so a timed-out session leaves its records behind.
    Never raises: a failed upload only prints one line (gives up after 3 failures in a row)."""
    h = os.environ.get("UCF_SYNC_DATASET")
    if not h or os.environ.get("UCF_NO_SYNC") or SMOKE or _LAST_SYNC[2] >= 3:
        return
    every_min = float(every_min if every_min is not None else os.environ.get("UCF_SYNC_EVERY", 60))
    if not force and time.time() - _LAST_SYNC[0] < every_min * 60:
        return
    try:
        n = len(list(REC.glob("*.json")))
        if n == 0 or n == _LAST_SYNC[1]:
            return
        pack_records(quiet=True)
        _kaggle_creds()
        user = os.environ.get("KAGGLE_USERNAME", "")
        handle = h if "/" in h else f"{user}/{h}"
        if not handle.split("/")[0]:
            raise RuntimeError("no Kaggle username (add Secrets KAGGLE_USERNAME / KAGGLE_KEY, or use 'user/slug')")
        d = ROOT / "_sync"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        shutil.copy2(ROOT / "ucf_ext_records.zip", d / "ucf_ext_records.zip")
        if (ROOT / "ucf_ext_records.csv").exists():
            shutil.copy2(ROOT / "ucf_ext_records.csv", d / "ucf_ext_records.csv")
        note = f"{n} records {time.strftime('%F %T')}"
        try:
            import kagglehub
            kagglehub.dataset_upload(handle, str(d), version_notes=note)
        except ImportError:
            slug = handle.split("/")[1]
            (d / "dataset-metadata.json").write_text(json.dumps(
                {"title": slug, "id": handle, "licenses": [{"name": "CC0-1.0"}]}))
            r = subprocess.run(["kaggle", "datasets", "version", "-p", str(d), "-m", note], capture_output=True, text=True)
            if r.returncode != 0:
                r = subprocess.run(["kaggle", "datasets", "create", "-p", str(d)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout).strip()[-200:])
        shutil.rmtree(d, ignore_errors=True)
        _LAST_SYNC[:] = [time.time(), n, 0]
        print(f"[sync] uploaded {n} records to Kaggle dataset {handle}", flush=True)
    except Exception as e:
        _LAST_SYNC[0] = time.time()
        _LAST_SYNC[2] += 1
        print(f"[sync] upload skipped ({type(e).__name__}: {str(e)[:150]}); records stay in {ROOT}", flush=True)


def run_jobs(jobs, dev, retry_failed=False):
    cache, done, t0 = {}, 0, time.time()
    verify_records()
    if retry_failed:
        clear_fails()
    try:
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))     # let `finally` / except-BaseException release the claim
    except ValueError:
        pass
    todo = sum(1 for j in jobs if not job_path(j).exists())
    print(f"{len(jobs) - todo}/{len(jobs)} runs already on disk; {todo} to do", flush=True)
    for k, j in enumerate(jobs):
        p = job_path(j)
        if p.exists():
            continue
        if n_fails(p) >= MAX_FAILS:
            print(f"[{k + 1}/{len(jobs)}] skipping {job_name(j)}: failed {n_fails(p)}x (see errors.log; --retry_failed to retry)", flush=True)
            continue
        if _past_deadline():
            print("[time budget] reached: not starting new runs in this session (finished runs are kept).", flush=True)
            break
        if not claim(p):
            continue
        name = job_name(j)
        print(f"[{k + 1}/{len(jobs)}] {name}", flush=True)
        hb = Heartbeat(p)
        hb.start()
        _failed_run = False
        try:
            rec = execute(j, cache, dev)
            atomic_json(rec, p)                  # claim file stays: it keeps a racing worker from repeating the run
            ckpt_clear(j)
            m = rec.get("metrics")
            print("   ->", (f'BLEU {m["bleu"]:.2f} EM {m["em"]:.1f} valid {m["valid"]:.1f}' if m else
                            {L: round(v["full"], 3) for L, v in rec["bpc"].items()}),
                  f'| {rec["train_s"] / 60:.1f} min' + (f' | resumed at {rec["resumes"]}' if rec.get("resumes") else ""), flush=True)
            done += 1
            _autopack()
        except Exception as e:
            traceback.print_exc()
            with open(ROOT / "errors.log", "a") as f:
                f.write(f"{time.strftime('%F %T')} {name}: {e!r}\n")
            bump_fail(p)
            release(p)                           # a later invocation may retry (up to MAX_FAILS)
            _failed_run = True
        except BaseException:                    # kernel interrupt / SIGTERM: free the claim, keep the checkpoint
            hb.stop()
            release(p)
            kept = j["type"] == "train" and _ckpt_file(j["spec"], j["seed"]).exists()
            print(f"interrupted during {name}: claim released" + ("; checkpoint kept" if kept else ""), flush=True)
            raise
        finally:
            hb.stop()
        if _failed_run:                          # here the exception (and the traceback that pins the failed model) is gone
            _failed_run = False
            if HAVE_TORCH and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
    print(f"finished {done} runs in {(time.time() - t0) / 3600:.2f} h")


# ---- previous sessions: import / pack / status -------------------------------------------
def _reload_revs():
    global REV
    if _REVF.exists():
        try:
            REV = json.load(open(_REVF))
        except Exception:
            pass


def import_records(src, with_ckpt=False):
    """Copy valid records (and revs.json) of another session's output into this session's REC.  src: a ucf_ext directory, its
    records/ directory, or a zip written by --stage pack.  Existing records are never overwritten."""
    src = Path(src)
    new = skip = bad = 0
    REC.mkdir(parents=True, exist_ok=True)

    def put(name, data):
        nonlocal new, skip, bad
        dst = REC / name
        if dst.exists():
            skip += 1
            return
        try:
            ok = _valid_record(json.loads(data))
        except Exception:
            ok = False
        if not ok:
            bad += 1
            return
        atomic_bytes(dst, data)
        new += 1

    def put_revs(data):
        if not _REVF.exists():
            _REVF.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(_REVF, data)
            _reload_revs()
        elif _REVF.read_bytes() != data:
            print("   WARNING: revs.json of the previous session differs from this one; keeping this session's pins", flush=True)

    rx = re.compile(r"^(run|screen|timing)_[0-9a-f]+\.json$")
    if src.suffix == ".zip" and src.is_file():
        import zipfile
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                b = os.path.basename(n)
                if rx.match(b):
                    put(b, z.read(n))
                elif b == "revs.json":
                    put_revs(z.read(n))
                elif with_ckpt and b.endswith(".pt") and "ckpt" in n and not (ROOT / "ckpt" / b).exists():
                    (ROOT / "ckpt").mkdir(exist_ok=True)
                    atomic_bytes(ROOT / "ckpt" / b, z.read(n))
    elif src.suffix == ".csv" and src.is_file():
        import csv
        csv.field_size_limit(2 ** 31 - 1)
        with open(src, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                b, data = os.path.basename(row.get("file") or ""), (row.get("record_json") or "").encode("utf-8")
                if rx.match(b):
                    put(b, data)
                elif b == "revs.json" and data:
                    put_revs(data)
    elif src.is_dir():
        rd = src / "records" if (src / "records").is_dir() else src
        if rd.resolve() == REC.resolve():
            return 0, 0, 0
        if "smoke" in (rd.name, rd.parent.name, src.name):
            print(f"   {src}: looks like a --smoke output (tiny data); not imported", flush=True)
            return 0, 0, 0
        for f in sorted(rd.glob("*.json")):
            if rx.match(f.name):
                put(f.name, f.read_bytes())
        for rv in (rd.parent / "revs.json", src / "revs.json"):
            if rv.exists():
                put_revs(rv.read_bytes())
                break
        if with_ckpt:
            for f in (rd.parent / "ckpt").glob("run_*.pt"):
                if not (ROOT / "ckpt" / f.name).exists():
                    (ROOT / "ckpt").mkdir(exist_ok=True)
                    shutil.copy2(f, ROOT / "ckpt" / (f.name + ".tmpimp"))
                    os.replace(ROOT / "ckpt" / (f.name + ".tmpimp"), ROOT / "ckpt" / f.name)
    return new, skip, bad


def discover_previous():
    base, out = Path("/kaggle/input"), []
    if base.is_dir():
        for pat in ("*/records", "*/*/records", "*/*/*/records"):
            out += [q.parent for q in base.glob(pat) if q.is_dir() and q.parent.name != "smoke"]   # smoke runs are not data
        for pat in ("*/ucf_ext_records.zip", "*/*/ucf_ext_records.zip", "*/*/*/ucf_ext_records.zip",
                    "*/ucf_ext_records.csv", "*/*/ucf_ext_records.csv", "*/*/*/ucf_ext_records.csv"):
            out += list(base.glob(pat))
    return out


def do_import(args, explicit=False):
    srcs = [Path(x) for x in (args.resume_from or [])] + [Path(x) for x in os.environ.get("UCF_PREV", "").split(os.pathsep) if x]
    if not (args.no_auto_import or SMOKE):
        srcs += discover_previous()
    seen, tot = set(), 0
    for sp in srcs:
        key = str(sp.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not sp.exists():
            print(f"[resume] {sp}: not found", flush=True)
            continue
        n, k, b = import_records(sp, args.with_ckpt)
        tot += n
        if n or b or explicit:
            print(f"[resume] {sp}: imported {n} records, {k} already present, {b} unreadable", flush=True)
    if explicit and not srcs:
        print("[resume] nothing to import (pass --resume_from PATH, set UCF_PREV, or attach the previous output as a Kaggle input)")
    return tot


def pack_csv():
    """ucf_ext_records.csv: readable summary columns + record_json (the full record), so the CSV is also a resume file."""
    import csv
    out = ROOT / "ucf_ext_records.csv"
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    cols = ["file", "kind", "system", "seed", "bleu", "em", "valid", "codebleu", "train_s", "record_json"]
    n = 0
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for f in sorted(REC.glob("*.json")):
            try:
                txt = f.read_text(encoding="utf-8")
                r = json.loads(txt)
                if not _valid_record(r):
                    continue
            except Exception:
                continue
            m = r.get("metrics") or {}
            try:
                sysn = sys_id(r["spec"]) if r.get("kind") == "train" and r.get("spec") else r.get("system", "")
            except Exception:
                sysn = ""
            w.writerow([f.name, r.get("kind"), sysn, r.get("seed", ""), m.get("bleu", ""), m.get("em", ""), m.get("valid", ""),
                        m.get("codebleu", ""), r.get("train_s", ""), txt])
            n += 1
        if _REVF.exists():
            w.writerow(["revs.json", "revs", "", "", "", "", "", "", "", _REVF.read_text(encoding="utf-8")])
    os.replace(tmp, out)
    return n


def pack_records(quiet=False):
    import zipfile
    out = ROOT / "ucf_ext_records.zip"
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(REC.glob("*.json")):
            z.write(f, f"records/{f.name}")
            n += 1
        for extra in (_REVF, ROOT / "errors.log"):
            if extra.exists():
                z.write(extra, extra.name)
    os.replace(tmp, out)
    try:
        pack_csv()
    except Exception as e:
        print(f"[pack] csv skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)
    if quiet:
        return
    print(f"packed {n} records -> {out} ({out.stat().st_size / 2 ** 20:.1f} MB)\n"
          "Keep it: Kaggle 'Save Version' stores /kaggle/working as the notebook output; or download it / upload it as a dataset.\n"
          "A later session imports it automatically when attached as an input, or with  --resume_from <path>.")


STATUS_ORDER = ["baselines", "encoders", "functional", "functional_best", "frozen_dec", "scale", "newdata", "data40k", "seeds15", "seeds15_gpt",
                "long_resampler", "dimmismatch", "bpc_trunc"]


def status(args):
    """Progress per stage from the files on disk (CPU only), and the commands that continue the work."""
    verify_records()
    print(f"output directory: {ROOT}   ({len(list(REC.glob('*.json')))} records)\n")
    print(f"{'stage':28s} {'done':>9s} {'running':>8s} {'stale':>6s} {'failed':>7s} {'ckpt':>5s}  next")
    todo_stages = []
    for stg in STATUS_ORDER:
        variants = [None]
        if stg == "newdata" and not args.dataset:
            variants = ["csn_java", "csn_python", "py_alpaca"]
        for dsn in variants:
            a = argparse.Namespace(**vars(args))
            a.dataset = dsn or args.dataset
            jobs = make_jobs(stg, a)
            d = run = st = fl = ck = 0
            first = None
            for j in jobs:
                p = job_path(j)
                if p.exists():
                    d += 1
                    continue
                c = Path(str(p) + ".claim")
                if c.exists():
                    if claim_stale(str(c)):
                        st += 1
                    else:
                        run += 1
                if n_fails(p) >= MAX_FAILS:
                    fl += 1
                if j["type"] == "train" and _ckpt_file(j["spec"], j["seed"]).exists():
                    ck += 1
                if first is None and n_fails(p) < MAX_FAILS:
                    first = job_name(j)
            label = stg + (f" --dataset {dsn}" if dsn else "")
            print(f"{label:28s} {d:4d}/{len(jobs):<4d} {run:8d} {st:6d} {fl:7d} {ck:5d}  {first or 'complete'}")
            if d < len(jobs):
                todo_stages.append(label)
    tim = len(list(REC.glob("timing_*.json")))
    print(f"{'timing':28s} {tim:4d}/{len(timing_specs()) * args.reps:<4d}  (run alone on one GPU: --stage timing)")
    ret = ROOT / "errors.log"
    if ret.exists():
        print(f"\nerrors.log has {len(ret.read_text().splitlines())} line(s); last: {ret.read_text().splitlines()[-1][:160]}")
    if todo_stages:
        names = []
        for lab in todo_stages:
            n_ = lab.split()[0]
            if n_ not in names:
                names.append(n_)
        print(f'\nNext: MODE = "full" and run the cell again (it continues where it stopped); left: {", ".join(names)}')
        print(f"  or from a terminal:   python ucf_extensions_kaggle_v3.py --stage {names[0]} --device cuda:0   (and cuda:1 in a second process)")
    else:
        print("\nAll stages complete: run  --stage analyze")


# --------------------------------------------------------------------------------------
# 6. Dedicated-hardware timing
# --------------------------------------------------------------------------------------
def smi(q):
    try:
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        return str(e)


def other_gpu_procs():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                      text=True).strip().splitlines()
        return [l for l in out if l and not l.startswith(str(os.getpid()))]
    except Exception:
        return []


def timing_specs():
    P = lambda *a, **k: S(*a, ds="concode_java", **k)
    return [P(None, "plbart"), P(None, "plbart", tune="lora"), P(None, "codet5"), P(None, "codegpt"),
            P("codet5", "plbart", "linear"), P("codet5", "plbart", "linear", tune="lora"),
            P("codet5", "plbart", "resampler"), P("codet5", "plbart", "linear", tune="frozen"),
            P("unixcoder", "plbart", "linear"), P("codet5", "codegpt", "linear"),
            P("codet5", "plbart", "calm"), P(None, "plbart", tune="adapter64")]


def run_timing(args):
    dev = args.device
    data = load_data("concode_java")
    assert data["train"] and data["test"], "empty CONCODE train/test split: timing needs real data"
    reps = 1 if SMOKE else args.reps
    nsteps, warm = (5, 2) if SMOKE else (args.timing_steps, 20)
    lat_n, lat_tok = (32 if SMOKE else 256), 64
    busy = other_gpu_procs()
    if busy:
        print("WARNING: other processes use the GPU -> timings are NOT dedicated-hardware:", busy)
    print("GPU:", smi("name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"))
    specs = timing_specs()
    for rep in range(reps):
        order = specs[:]
        random.Random(rep).shuffle(order)               # interleave systems -> thermal drift is not systematic
        for sp in order:
            p = REC / f"timing_{_h(sys_id(sp), rep)}.json"
            if p.exists():
                continue
            print(f"[rep {rep}] {sys_id(sp)}", flush=True)
            seed_all(0)
            s = System(sp).to(dev)
            tot_p, tr_p = s.count_params()
            tr = data["train"][:(warm + nsteps) * HP["bs"]]
            batches = [to_dev(s.tokenize([c[0] for c in tr[i:i + HP["bs"]]], [c[1] for c in tr[i:i + HP["bs"]]]), dev)
                       for i in range(0, len(tr), HP["bs"])]                     # tokenisation excluded from timing
            test_src = [c[0] for c in data["test"][:lat_n]]
            lat_b = [to_dev(s.tokenize(test_src[i:i + 32]), dev) for i in range(0, len(test_src), 32)]
            params = [q for g in s.param_groups() for q in g["params"]]
            opt, sched = make_opt(s, 1875)
            scaler = make_scaler(s.amp)
            smi0 = smi("temperature.gpu,clocks.sm,power.draw")
            s.train()
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            ts = []
            for k, b in enumerate(batches):
                torch.cuda.synchronize(dev)
                t = time.perf_counter()
                train_step(s, opt, sched, scaler, b, params)
                torch.cuda.synchronize(dev)
                if k >= warm:
                    ts.append((time.perf_counter() - t) * 1000)
            peak_tr = torch.cuda.max_memory_allocated(dev) / 2 ** 30

            def lat():
                s.eval()
                s.gen(lat_b[0], lat_tok, lat_tok)                                # warm-up
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                t = time.perf_counter()
                for b in lat_b:
                    s.gen(b, lat_tok, lat_tok)                                   # fixed-length decoding
                torch.cuda.synchronize(dev)
                return (time.perf_counter() - t) / len(test_src) * 1000, torch.cuda.max_memory_allocated(dev) / 2 ** 30

            lat_ms, peak_lat = lat()
            lat_merged = None
            if s._peft:                                                          # merged LoRA: no extra matmuls
                s._peft[0].merge_and_unload()
                lat_merged, _ = lat()
            rec = dict(kind="timing", sys_id=sys_id(sp), rep=rep, step_ms=ts, trainable_M=tr_p / 1e6, total_M=tot_p / 1e6,
                       train_peak_gb=peak_tr, lat_ms_per_ex=lat_ms, lat_merged_ms_per_ex=lat_merged,
                       lat_peak_gb=peak_lat, lat_new_tokens=lat_tok, smi_before=smi0,
                       smi_after=smi("temperature.gpu,clocks.sm,power.draw"), gpu=smi("name"),
                       other_procs=other_gpu_procs())
            atomic_json(rec, p)
            print(f"   step {np.median(ts):.0f} ms | latency {lat_ms:.1f} ms/ex | peak {peak_tr:.2f} GB", flush=True)
            del s, opt, batches, lat_b
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(20 if not SMOKE else 0)                                   # let the GPU cool between systems


# --------------------------------------------------------------------------------------
# 7. Analysis (CPU only)
# --------------------------------------------------------------------------------------
def load_records(kind):
    out = []
    for f in REC.glob("*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("kind") == kind:
            out.append(r)
    return out


def tci(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), (float("nan"), float("nan")), n
    sd = float(x.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, (m - h, m + h), n


def sys_table(recs):
    d = defaultdict(dict)
    for r in recs:
        d[r["sys_id"]][r["seed"]] = r
    return d


def _stats(rec):
    if "_bs" not in rec:
        rec["_bs"] = np.array([bleu_stats(norm(p), norm(r)) for p, r in zip(rec["preds"], rec["refs"])])
        rec["_em"] = np.array([float(norm(p) == norm(r)) for p, r in zip(rec["preds"], rec["refs"])])
    return rec["_bs"], rec["_em"]


def boot_pair(A, B, seeds, n=5000):
    """Paired bootstrap over test examples; BLEU/EM recomputed per seed on each resample, averaged over seeds."""
    SA = np.stack([_stats(A[s])[0] for s in seeds])
    SB = np.stack([_stats(B[s])[0] for s in seeds])
    EA = np.stack([_stats(A[s])[1] for s in seeds])
    EB = np.stack([_stats(B[s])[1] for s in seeds])
    ns, N = EA.shape
    rng = np.random.RandomState(0)
    db, de = [], []
    for c in range(0, n, 250):
        k = min(250, n - c)
        W = rng.multinomial(N, [1.0 / N] * N, size=k).astype(float)
        a = np.einsum("kn,snj->ksj", W, SA).reshape(-1, 10)
        b = np.einsum("kn,snj->ksj", W, SB).reshape(-1, 10)
        db.append(bleu_from_stats(a).reshape(k, ns).mean(1) - bleu_from_stats(b).reshape(k, ns).mean(1))
        de.append(100 * ((W @ EA.T) - (W @ EB.T)).mean(1) / N)
    db, de = np.concatenate(db), np.concatenate(de)
    pt_b = bleu_from_stats(SA.sum(1)).mean() - bleu_from_stats(SB.sum(1)).mean()
    pt_e = 100 * (EA.mean() - EB.mean())
    return (pt_b, *np.percentile(db, [2.5, 97.5])), (pt_e, *np.percentile(de, [2.5, 97.5]))


def boot_pass(A, B, seeds, n=5000):
    PA = np.stack([np.array(A[s]["passed"], float) for s in seeds])
    PB = np.stack([np.array(B[s]["passed"], float) for s in seeds])
    N = PA.shape[1]
    rng = np.random.RandomState(0)
    d = []
    for c in range(0, n, 500):
        k = min(500, n - c)
        W = rng.multinomial(N, [1.0 / N] * N, size=k).astype(float)
        d.append(100 * ((W @ PA.T) - (W @ PB.T)).mean(1) / N)
    return (100 * (PA.mean() - PB.mean()), *np.percentile(np.concatenate(d), [2.5, 97.5]))


def paired_line(label, A, B):
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return f"| {label} | – | – | – | – | – | – | – |"
    d = [A[s]["metrics"]["bleu"] - B[s]["metrics"]["bleu"] for s in seeds]
    m, sd, ci, n = tci(d)
    (bb, bl, bhi), (eb, el, eh) = boot_pair(A, B, seeds)
    p_perm = perm_pair(A, B, seeds)
    p_seed = float("nan")
    if n >= 3 and sd > 0:
        from scipy import stats as _st
        p_seed = float(_st.ttest_1samp(d, 0.0).pvalue)
    pas, pd_, pp = "–", float("nan"), float("nan")
    if all("passed" in A[s] and "passed" in B[s] for s in seeds):
        pd_, lo, hi = boot_pass(A, B, seeds)
        pp = perm_pass(A, B, seeds)
        pas = f"{pd_:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    if all("he_passed" in A[s] and "he_passed" in B[s] for s in seeds):          # HumanEval transfer
        hd = 100 * np.mean([np.mean(A[s]["he_passed"]) - np.mean(B[s]["he_passed"]) for s in seeds])
        pas += f" (HumanEval {hd:+.1f}, p={perm_pass(A, B, seeds, key='he_passed'):.3f})"
    PAIR_LOG[(A[seeds[0]]["sys_id"], B[seeds[0]]["sys_id"])] = dict(
        label=label, n=n, d_seed=m, sd=sd, dz=(m / sd if sd > 0 else float("nan")), bb=bb, bl=bl, bh=bhi,
        p_perm=p_perm, p_seed=p_seed, pass_d=pd_, p_pass=pp)
    return (f"| {label} | {n} | {m:+.2f} (sd {sd:.2f}) [{ci[0]:+.2f}, {ci[1]:+.2f}] ({sum(x > 0 for x in d)}/{n} seeds > 0) "
            f"| {bb:+.2f} [{bl:+.2f}, {bhi:+.2f}] | {eb:+.1f} [{el:+.1f}, {eh:+.1f}] | {pas} | {p_perm:.4f} | {_fp(p_seed)} |")


def sys_row(sid, R):
    seeds = sorted(R)
    b = [R[s]["metrics"]["bleu"] for s in seeds]
    m, sd, ci, n = tci(b)
    mean = lambda k: np.mean([R[s]["metrics"][k] for s in seeds])
    cb = [R[s]["metrics"]["codebleu"] for s in seeds if R[s]["metrics"].get("codebleu") is not None]
    ps = [R[s]["metrics"]["pass1"] for s in seeds if "pass1" in R[s]["metrics"]]
    r0 = R[seeds[0]]
    pass_txt = (f"{np.mean(ps):.1f} ± {np.std(ps, ddof=1) if len(ps) > 1 else float('nan'):.1f}") if ps else "–"
    he_ = [R[s]["metrics"]["he_pass1"] for s in seeds if R[s]["metrics"].get("he_pass1") is not None]
    if he_:
        pass_txt += f" (HumanEval {np.mean(he_):.1f})"
    return (f"| {sid} | {n} | {m:.2f} ± {sd:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {np.mean(cb) if cb else float('nan'):.1f} | "
            f"{mean('em'):.1f} | {mean('valid'):.1f} | {mean('distinct'):.1f} | "
            f"{pass_txt} | "
            f"{r0['trainable_M']:.1f} | {np.mean([R[s]['val_bpc'] for s in seeds]):.3f} | "
            f"{np.mean([R[s]['train_s'] for s in seeds]) / 60:.1f} |")


HDR_SYS = ("| system | n | test BLEU mean ± sd [95% t-CI] | CodeBLEU | EM | valid | distinct | pass@1 (executed tests) | trainable M | val BPC | train min |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|")
HDR_PAIR = ("| comparison (A − B) | n | seed-level ΔBLEU: mean (sd) [95% t-CI] | bootstrap ΔBLEU [95% CI] | bootstrap ΔEM pts [95% CI] | bootstrap Δpass@1 pts [95% CI] | p perm (test set) | p t-test (seeds) |\n"
            "|---|---|---|---|---|---|---|---|")


def analyze(figs=True):
    PAIR_LOG.clear()
    T = sys_table(load_records("train"))
    out = []
    def add(*l):
        out.extend(l)

    def get(sp):
        return T.get(sys_id(sp), {})

    # ---- 1. seeds15 ----
    sp = {k: v[0] for k, v in zip(("naive", "linear", "resampler", "native"), stage_specs("seeds15"))}
    if get(sp["linear"]):
        add("## 1. Core comparison with up to 15 seeds (CodeT5→PLBART, CONCODE)", "", HDR_SYS)
        for k in ("native", "naive", "resampler", "linear"):
            if get(sp[k]):
                add(sys_row(sys_id(sp[k]), get(sp[k])))
        add("", HDR_PAIR)
        for a, b in (("linear", "naive"), ("linear", "resampler"), ("linear", "native"), ("resampler", "naive")):
            add(paired_line(f"{a} − {b}", get(sp[a]), get(sp[b])))
        add("", "Replication check against Table X (first five seeds; differences > ~0.7 BLEU suggest the data subset or "
            "implementation differs from ucf_revision_kaggle.py):", "",
            "| system | seed | this script | Table X | diff |", "|---|---|---|---|---|")
        for k in ("native", "naive", "resampler", "linear"):
            for sd, ref in zip(SEEDS_CORE5, PAPER_TABLE_X[k]):
                r = get(sp[k]).get(sd)
                if r:
                    add(f"| {k} | {sd} | {r['metrics']['bleu']:.1f} | {ref} | {r['metrics']['bleu'] - ref:+.1f} |")
        add("", "First five seeds vs all seeds (does the 5-seed interval in the paper hold up?):", "",
            "| system | mean (5 seeds) | mean (all) | sd (5) | sd (all) |", "|---|---|---|---|---|")
        for k in ("native", "naive", "resampler", "linear"):
            R = get(sp[k])
            a = [R[s]["metrics"]["bleu"] for s in SEEDS_CORE5 if s in R]
            b = [R[s]["metrics"]["bleu"] for s in R]
            if len(a) >= 2 and len(b) >= 2:
                add(f"| {k} | {np.mean(a):.2f} | {np.mean(b):.2f} | {np.std(a, ddof=1):.2f} | {np.std(b, ddof=1):.2f} |")
        add("")
    g = {k: get(v[0]) for k, v in zip(("naive", "linear", "native"), stage_specs("seeds15_gpt"))}
    if g["linear"]:
        add("## 1b. CodeT5→CodeGPT with up to 15 seeds", "", HDR_SYS)
        for k in ("native", "naive", "linear"):
            if g[k]:
                add(sys_row(k, g[k]))
        add("", HDR_PAIR, paired_line("linear − naive", g["linear"], g["naive"]),
            paired_line("linear − native CodeGPT", g["linear"], g["native"]), "")

    # ---- generic stage tables ----
    def stage_table(title, stage, ds="concode_java", pairs=()):
        specs = stage_specs(stage, ds)
        rows = [(sys_id(s_), get(s_)) for s_, _, _ in specs if get(s_)]
        if not rows:
            return
        add(f"## {title}", "", HDR_SYS)
        for sid, R in rows:
            add(sys_row(sid, R))
        pl = []
        for la, a, lb, b in pairs:
            if get(a) and get(b):
                pl.append(paired_line(f"{la} − {lb}", get(a), get(b)))
        if pl:
            add("", HDR_PAIR, *pl)
        add("")

    P = lambda *a, **k: S(*a, ds="concode_java", **k)
    pairs_enc = []
    for e in ("unixcoder", "roberta", "codebert"):
        pairs_enc += [(f"{e}→PLBART linear", P(e, "plbart", "linear"), f"{e}→PLBART naive", P(e, "plbart", "naive")),
                      (f"{e}→PLBART linear", P(e, "plbart", "linear"), "codet5→PLBART linear", P("codet5", "plbart", "linear"))]
    pairs_enc.append(("unixcoder→CodeGPT linear", P("unixcoder", "codegpt", "linear"),
                      "unixcoder→CodeGPT naive", P("unixcoder", "codegpt", "naive")))
    stage_table("2. Non-CodeT5 encoders (fixed decoder = PLBART; CodeGPT with UniXcoder)", "encoders", pairs=pairs_enc)
    stage_table("3. Frozen-decoder (bridge-only) condition", "frozen_dec", pairs=[
        ("linear frozen", P("codet5", "plbart", "linear", tune="frozen"), "linear full-tune", P("codet5", "plbart", "linear")),
        ("linear frozen", P("codet5", "plbart", "linear", tune="frozen"), "resampler frozen", P("codet5", "plbart", "resampler", tune="frozen")),
        ("CodeGPT linear frozen", P("codet5", "codegpt", "linear", tune="frozen"), "CodeGPT linear full",
         P("codet5", "codegpt", "linear"))])
    lr_pairs = []
    for ep in (3, 6, 12):
        lr_pairs.append((f"linear ep{ep}", P("codet5", "plbart", "linear", epochs=ep), f"resampler ep{ep}",
                         P("codet5", "plbart", "resampler", epochs=ep)))
    stage_table("4. Longer resampler training (equal budget for the linear bridge)", "long_resampler", pairs=lr_pairs)
    # epoch curves
    curves = [(sid, R) for sid, R in T.items() if any(r.get("val_bpc_epochs") for r in R.values())]
    if curves:
        add("Validation BPC by epoch (mean over seeds):", "")
        for sid, R in sorted(curves):
            ne = max(len(r["val_bpc_epochs"]) for r in R.values())
            mm = [np.mean([r["val_bpc_epochs"][i][1] for r in R.values() if len(r["val_bpc_epochs"]) > i]) for i in range(ne)]
            add(f"- {sid}: " + ", ".join(f"ep{i + 1}={v:.3f}" for i, v in enumerate(mm)))
        add("")
    for dsn in ("csn_java", "csn_python", "py_alpaca"):
        N = lambda *a, **k: S(*a, ds=dsn, **k)
        stage_table(f"5. Additional dataset: {dsn}", "newdata", ds=dsn, pairs=[
            ("CodeT5→PLBART linear", N("codet5", "plbart", "linear"), "CodeT5→PLBART naive", N("codet5", "plbart", "naive")),
            ("CodeT5→PLBART linear", N("codet5", "plbart", "linear"), "CodeT5→PLBART resampler", N("codet5", "plbart", "resampler")),
            ("CodeT5→PLBART linear", N("codet5", "plbart", "linear"), "native PLBART", N(None, "plbart")),
            ("CodeT5→PLBART linear", N("codet5", "plbart", "linear"), "native PLBART+LoRA", N(None, "plbart", tune="lora")),
            ("CodeT5→PLBART linear", N("codet5", "plbart", "linear"), "PLBART→PLBART naive", N("plbart", "plbart", "naive")),
            ("CodeT5→CodeGPT linear", N("codet5", "codegpt", "linear"), "CodeT5→CodeGPT naive", N("codet5", "codegpt", "naive")),
            ("CodeT5→CodeGPT linear", N("codet5", "codegpt", "linear"), "native CodeGPT", N(None, "codegpt"))])
    dm = []
    for e, d in (("codet5_small", "plbart"), ("codet5", "codet5_small"), ("unixcoder", "codet5_small")):
        for other in ("padtrunc", "resampler"):
            dm.append((f"{e}→{d} linear", P(e, d, "linear"), other, P(e, d, other)))
        dm.append((f"{e}→{d} linear", P(e, d, "linear"), f"native {d}", P(None, d)))
    stage_table("8. Dimension-mismatched bridging (hidden sizes 512 ↔ 768; 'padtrunc' = non-learned zero-pad/truncate)",
                "dimmismatch", pairs=dm)
    M = lambda *a, **k: S(*a, ds="mbpp_py", epochs=5, **k)
    stage_table("9. Execution-based functional correctness (MBPP test, pass@1; Python only)", "functional", ds="mbpp_py", pairs=[
        ("CodeT5→PLBART linear", M("codet5", "plbart", "linear"), "CodeT5→PLBART naive", M("codet5", "plbart", "naive")),
        ("CodeT5→PLBART linear", M("codet5", "plbart", "linear"), "CodeT5→PLBART resampler", M("codet5", "plbart", "resampler")),
        ("CodeT5→PLBART linear", M("codet5", "plbart", "linear"), "native PLBART", M(None, "plbart")),
        ("CodeT5→PLBART linear", M("codet5", "plbart", "linear"), "native PLBART+LoRA", M(None, "plbart", tune="lora")),
        ("CodeT5→PLBART linear", M("codet5", "plbart", "linear"), "PLBART→PLBART linear", M("plbart", "plbart", "linear")),
        ("CodeT5→CodeGPT linear", M("codet5", "codegpt", "linear"), "CodeT5→CodeGPT naive", M("codet5", "codegpt", "naive")),
        ("CodeT5→CodeGPT linear", M("codet5", "codegpt", "linear"), "native CodeGPT", M(None, "codegpt"))])
    Mb = lambda *a, **k: S(*a, ds="mbpp_py", epochs=FB["max_epochs"], **k)
    fbn = f'{FB["enc"]}→{FB["dec"]} {FB["bridge"]}'
    stage_table("9b. Training until functional correctness (early stopping on validation pass@1; test and HumanEval scored once at the best epoch)",
                "functional_best", ds="mbpp_py",
                pairs=[(fbn, Mb(FB["enc"], FB["dec"], FB["bridge"]), f"native {FB['dec']} (same protocol)", Mb(None, FB["dec"]))] if FB["ref"] else [])
    analyze_fb(T, out)
    for dsn in ("concode_java", "csn_java", "py_alpaca", "csn_python"):
        Bd = lambda *a, _d=dsn, **k: S(*a, ds=_d, **k)
        stage_table(f"10. External composition and parameter-efficient baselines ({dsn})", "baselines", ds=dsn, pairs=[
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "CALM-style cross-attention (decoder tuned)", Bd("codet5", "plbart", "calm")),
            ("CALM-style frozen", Bd("codet5", "plbart", "calm", tune="frozen"), "linear frozen", Bd("codet5", "plbart", "linear", tune="frozen")),
            ("CodeT5→CodeGPT linear", Bd("codet5", "codegpt", "linear"), "CodeT5→CodeGPT CALM-style", Bd("codet5", "codegpt", "calm")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "native PLBART + LoRA r=16 (paper setting)", Bd(None, "plbart", tune="lora")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "PLBART→PLBART naive (same family)", Bd("plbart", "plbart", "naive")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "native PLBART + adapters (64)", Bd(None, "plbart", tune="adapter64")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "native PLBART + LoRA r=4", Bd(None, "plbart", tune="lora4")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "native PLBART + LoRA r=64", Bd(None, "plbart", tune="lora64")),
            ("CodeT5→PLBART linear", Bd("codet5", "plbart", "linear"), "native PLBART + LoRA r=256", Bd(None, "plbart", tune="lora256")),
            ("bridge + adapter decoder", Bd("codet5", "plbart", "linear", tune="adapter64"), "native PLBART + adapters (64)", Bd(None, "plbart", tune="adapter64")),
            ("CodeT5+→PLBART linear", Bd("codet5p", "plbart", "linear"), "CodeT5→PLBART linear", Bd("codet5", "plbart", "linear")),
            ("CodeT5+→PLBART linear", Bd("codet5p", "plbart", "linear"), "CodeT5+→PLBART naive", Bd("codet5p", "plbart", "naive")),
            ("CodeT5+→PLBART linear", Bd("codet5p", "plbart", "linear"), "native CodeT5+ (220M)", Bd(None, "codet5p"))])
    stage_table("11. Larger components and mismatched hidden sizes (1024 ↔ 768; 'padtrunc' = non-learned reference)", "scale", pairs=[
        ("CodeT5+770M→PLBART linear", P("codet5p770", "plbart", "linear"), "padtrunc", P("codet5p770", "plbart", "padtrunc")),
        ("CodeT5+770M→PLBART linear", P("codet5p770", "plbart", "linear"), "resampler", P("codet5p770", "plbart", "resampler")),
        ("CodeT5+770M→PLBART linear", P("codet5p770", "plbart", "linear"), "CALM-style", P("codet5p770", "plbart", "calm")),
        ("CodeT5+770M→PLBART linear", P("codet5p770", "plbart", "linear"), "CodeT5→PLBART linear (smaller encoder)", P("codet5", "plbart", "linear")),
        ("CodeT5+770M→PLBART linear", P("codet5p770", "plbart", "linear"), "native PLBART", P(None, "plbart")),
        ("CodeT5→GPT-2 medium linear", P("codet5", "gpt2m", "linear"), "padtrunc", P("codet5", "gpt2m", "padtrunc")),
        ("CodeT5→GPT-2 medium linear", P("codet5", "gpt2m", "linear"), "CALM-style", P("codet5", "gpt2m", "calm")),
        ("CodeT5→GPT-2 medium linear", P("codet5", "gpt2m", "linear"), "native GPT-2 medium", P(None, "gpt2m"))])
    stage_table("12. Four times more training data (40,000 CONCODE examples; same validation/test subsets)", "data40k", ds="concode_java_40k", pairs=[
        ("linear", S("codet5", "plbart", "linear", ds="concode_java_40k"), "naive", S("codet5", "plbart", "naive", ds="concode_java_40k")),
        ("linear", S("codet5", "plbart", "linear", ds="concode_java_40k"), "resampler", S("codet5", "plbart", "resampler", ds="concode_java_40k")),
        ("linear", S("codet5", "plbart", "linear", ds="concode_java_40k"), "native PLBART", S(None, "plbart", ds="concode_java_40k")),
        ("linear", S("codet5", "plbart", "linear", ds="concode_java_40k"), "native PLBART + LoRA", S(None, "plbart", tune="lora", ds="concode_java_40k"))])
    mc_report(out)
    analyze_errors(T, out)
    analyze_bpc(out)
    analyze_timing(out)
    export_csv(T)
    if figs:
        make_figures(T)
    txt = "\n".join(out)
    (ROOT / "analysis.md").write_text(txt)
    print(txt)
    print(f"\n[saved to {ROOT / 'analysis.md'}]")


# --------------------------------------------------------------------------------------
# 7b. Statistics added for the revision: permutation tests, Holm / BH correction, equivalence margin
# --------------------------------------------------------------------------------------
PAIR_LOG = {}          # (sys_id A, sys_id B) -> dict with every paired comparison printed by analyze()
EQUIV_BLEU = 1.0       # equivalence margin (BLEU points) used for the "equivalent" verdict


def perm_pair(A, B, seeds, n=5000):
    """Paired approximate-randomisation test over test examples (the same random swap of A/B outputs is applied to
    every seed).  Two-sided p-value for the mean per-seed corpus-BLEU difference.  Unlike the bootstrap interval this
    uses an explicit null distribution (A and B exchangeable on each example), so it can feed Holm / BH correction."""
    SA = np.stack([_stats(A[s])[0] for s in seeds])
    SB = np.stack([_stats(B[s])[0] for s in seeds])
    ns, N = SA.shape[:2]
    obs = float(bleu_from_stats(SA.sum(1)).mean() - bleu_from_stats(SB.sum(1)).mean())
    rng, cnt = np.random.RandomState(1), 0
    for c in range(0, n, 250):
        k = min(250, n - c)
        M = rng.randint(0, 2, size=(k, N)).astype(float)
        a = np.einsum("kn,snj->ksj", M, SA) + np.einsum("kn,snj->ksj", 1 - M, SB)
        b = np.einsum("kn,snj->ksj", M, SB) + np.einsum("kn,snj->ksj", 1 - M, SA)
        d = bleu_from_stats(a.reshape(-1, 10)).reshape(k, ns).mean(1) - bleu_from_stats(b.reshape(-1, 10)).reshape(k, ns).mean(1)
        cnt += int((np.abs(d) >= abs(obs) - 1e-9).sum())
    return (cnt + 1) / (n + 1)


def perm_pass(A, B, seeds, key="passed", n=5000):
    """Exact-form sign-flip test for a per-example 0/1 outcome (pass@1): swapping A/B on an example flips the sign of
    its mean paired difference."""
    D = np.mean([np.array(A[s][key], float) - np.array(B[s][key], float) for s in seeds], 0)
    obs = abs(D.mean())
    rng, cnt = np.random.RandomState(1), 0
    for c in range(0, n, 500):
        k = min(500, n - c)
        sg = rng.randint(0, 2, size=(k, len(D))) * 2.0 - 1.0
        cnt += int((np.abs((sg * D).mean(1)) >= obs - 1e-12).sum())
    return (cnt + 1) / (n + 1)


def holm(p):
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, run = len(idx), 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * p[i])
        out[i] = min(1.0, run)
    return out


def bh_adjust(p):
    """Benjamini-Hochberg q-values (NaN entries are ignored)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, run = len(idx), 1.0
    for r in range(m - 1, -1, -1):
        i = idx[r]
        run = min(run, p[i] * m / (r + 1))
        out[i] = run
    return out


def _plabel(k):
    r, ds_ = PAIR_LOG[k], k[0].split("|")[-1]
    if "→" in r["label"]:                                   # label already names the systems
        return r["label"] + ("" if ds_ == "concode_java" else f" [{ds_}]")
    return f"{r['label']} ({k[0].split('|')[0].replace('>', '→')}; {ds_})"


def _fp(x, f="{:.4f}"):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f.format(x)


def primary_pairs():
    """Confirmatory family, fixed here BEFORE the extension runs (CONCODE Java).  Everything else is exploratory."""
    c = lambda *a, **k: sys_id(S(*a, ds="concode_java", **k))
    return [(c("codet5", "plbart", "linear"), c("codet5", "plbart", "naive")),
            (c("codet5", "codegpt", "linear"), c("codet5", "codegpt", "naive")),
            (c("codet5", "plbart", "linear"), c("codet5", "plbart", "resampler")),
            (c("codet5", "plbart", "linear"), c(None, "plbart", tune="lora")),
            (c("codet5", "plbart", "linear"), c("plbart", "plbart", "naive")),
            (c("codet5", "plbart", "linear"), c("codet5", "plbart", "calm"))]


def _verdict(hp, hs, lo, hi):
    st = (not np.isnan(hp)) and hp < 0.05
    ss = (not np.isnan(hs)) and hs < 0.05
    v = "robust (test set + seeds)" if st and ss else "test-set only" if st else "seed-level only" if ss else "not significant"
    eq = "equivalent (±%.1f BLEU)" % EQUIV_BLEU if (lo > -EQUIV_BLEU and hi < EQUIV_BLEU) else "-"
    return v, eq


def mc_report(out):
    """Multiple-comparison control over every paired comparison printed by analyze()."""
    if not PAIR_LOG:
        return
    prim = [k for k in primary_pairs() if k in PAIR_LOG]
    out_rows = []

    def block(title, ks, note):
        if not ks:
            return
        pp = [PAIR_LOG[k]["p_perm"] for k in ks]
        ps = [PAIR_LOG[k]["p_seed"] for k in ks]
        H1, H2, Q = holm(pp), holm(ps), bh_adjust(pp)
        out.extend([f"## {title}", "", note, "",
                    "| comparison | n seeds | ΔBLEU (seed mean) [bootstrap 95% CI] | dz | p perm | Holm p | BH q | p seed-t | Holm p (seeds) | verdict | equivalence |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"])
        for k, h1, h2, q in zip(ks, H1, H2, Q):
            r = PAIR_LOG[k]
            v, eq = _verdict(h1, h2, r["bl"], r["bh"])
            out.append(f"| {_plabel(k)} | {r['n']} | {r['bb']:+.2f} [{r['bl']:+.2f}, {r['bh']:+.2f}] | {_fp(r['dz'], '{:+.1f}')} | "
                       f"{_fp(r['p_perm'])} | {_fp(h1)} | {_fp(q)} | {_fp(r['p_seed'])} | {_fp(h2)} | {v} | {eq} |")
            out_rows.append(dict(family=title, comparison=_plabel(k), n_seeds=r["n"], d_bleu=r["bb"], ci_lo=r["bl"], ci_hi=r["bh"],
                                 dz=r["dz"], p_perm=r["p_perm"], holm_p=h1, bh_q=q, p_seed=r["p_seed"], holm_p_seed=h2,
                                 verdict=v, equivalence=eq))
        out.append("")

    block(f"13. Multiple-comparison control: pre-specified primary family (m = {len(prim)})", prim,
          "Family fixed in `primary_pairs()` before the extension runs. Holm-Bonferroni across the family; 'p perm' = paired "
          "approximate-randomisation test over test examples, 'p seed-t' = paired t-test over seeds (few seeds: low power). "
          "'robust' = both Holm-adjusted p < 0.05. 'equivalent' = the whole bootstrap CI lies within ±1 BLEU.")
    rest = [k for k in PAIR_LOG if k not in prim]
    block(f"14. All other paired comparisons (exploratory family, m = {len(rest)})", rest,
          "Holm / BH across every remaining comparison printed above (all datasets and stages together). Read as descriptive.")
    if out_rows:
        import csv
        (ROOT / "tables").mkdir(exist_ok=True)
        with open(ROOT / "tables" / "paired_comparisons_holm.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)


def analyze_fb(T, out):
    """Stopping behaviour of the early-stopping protocol: epochs run, best epoch, stop reason, validation vs test pass@1."""
    rows = []
    for sp, _, _ in stage_specs("functional_best"):
        R = T.get(sys_id(sp), {})
        if not R:
            continue
        rs = [R[k] for k in sorted(R)]
        why = Counter(r.get("stop_reason") for r in rs)
        f = lambda key: np.mean([r[key] for r in rs if r.get(key) is not None])
        rows.append(f"| {sys_id(sp)} | {len(rs)} | {f('epochs_run'):.1f} | {f('best_epoch'):.1f} | {f('best_val_pass1'):.1f} | "
                    f"{np.mean([r['metrics']['pass1'] for r in rs]):.1f} | {', '.join(f'{k} {v}' for k, v in why.items())} |")
    if rows:
        out += ["Stopping behaviour (means over seeds). Validation pass@1 is the selection criterion and is optimistically biased; "
                "the test column is the unbiased estimate. 'max_epochs' means the cap, not convergence, ended training.", "",
                "| system | seeds | epochs run | best epoch | val pass@1 (selected) | test pass@1 | stop reasons |", "|---|---|---|---|---|---|---|", *rows, ""]


def _tok_len(s):
    return len(norm(s).split())


def analyze_errors(T, out):
    """Error analysis from the stored predictions: length behaviour and BLEU by reference-length tercile."""
    keys = [S("codet5", "plbart", "naive"), S("codet5", "plbart", "linear"), S("codet5", "plbart", "resampler"),
            S("codet5", "plbart", "calm"), S(None, "plbart"), S(None, "plbart", tune="lora"), S("codet5", "codegpt", "linear"),
            S(None, "codegpt")]
    rows = []
    for sp in keys:
        R = T.get(sys_id(sp), {})
        if not R:
            continue
        recs = [R[s] for s in sorted(R)]
        refs = recs[0]["refs"]
        rl = np.array([_tok_len(r) for r in refs])
        q = np.quantile(rl, [1 / 3, 2 / 3])
        grp = np.digitize(rl, q)
        ratio = np.mean([np.mean([_tok_len(p) / max(1, l) for p, l in zip(r["preds"], rl)]) for r in recs])
        empty = np.mean([np.mean([_tok_len(p) == 0 for p in r["preds"]]) for r in recs]) * 100
        by = []
        for g in range(3):
            idx = np.where(grp == g)[0]
            by.append(np.mean([bleu_from_stats(_stats(r)[0][idx].sum(0))[0] for r in recs]) if len(idx) else float("nan"))
        rows.append(f"| {sys_id(sp)} | {len(recs)} | {ratio:.2f} | {empty:.1f} | {by[0]:.1f} | {by[1]:.1f} | {by[2]:.1f} |")
    if rows:
        out += ["## 15. Error analysis (stored predictions)", "",
                "Length ratio = mean(predicted tokens / reference tokens); empty = share of empty outputs; BLEU by reference-length "
                "tercile (short / medium / long) shows whether a system only fails on long targets.", "",
                "| system | n seeds | length ratio | empty % | BLEU short | BLEU medium | BLEU long |", "|---|---|---|---|---|---|---|", *rows, ""]


def export_csv(T):
    import csv
    (ROOT / "tables").mkdir(exist_ok=True)
    keys = ["bleu", "chrf", "codebleu", "em", "valid", "distinct", "pass1", "he_pass1"]
    with open(ROOT / "tables" / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sys_id", "enc", "dec", "bridge", "tune", "epochs", "ds", "seed", *keys, "total_M", "trainable_M",
                    "train_s", "gen_s", "val_bpc"])
        for sid, R in sorted(T.items()):
            for sd, r in sorted(R.items()):
                sp = r["spec"]
                w.writerow([sid, sp["enc"] or "native", sp["dec"], sp["bridge"], sp["tune"], sp["epochs"], sp["ds"], sd,
                            *[r["metrics"].get(k) for k in keys], r["total_M"], r["trainable_M"], r["train_s"],
                            r.get("gen_s"), r.get("val_bpc")])
    with open(ROOT / "tables" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sys_id", "n_seeds", "bleu_mean", "bleu_sd", "bleu_ci_lo", "bleu_ci_hi", "em_mean", "valid_mean",
                    "pass1_mean", "he_pass1_mean", "trainable_M", "total_M", "train_min"])
        for sid, R in sorted(T.items()):
            b = [r["metrics"]["bleu"] for r in R.values()]
            m, sd, ci, n = tci(b)
            g = lambda k: (np.mean([r["metrics"][k] for r in R.values() if r["metrics"].get(k) is not None])
                           if any(r["metrics"].get(k) is not None for r in R.values()) else "")
            r0 = next(iter(R.values()))
            w.writerow([sid, n, m, sd, ci[0], ci[1], g("em"), g("valid"), g("pass1"), g("he_pass1"), r0["trainable_M"],
                        r0["total_M"], np.mean([r["train_s"] for r in R.values()]) / 60])


# --------------------------------------------------------------------------------------
# 7c. Publication figures (vector PDF + 300-dpi PNG; IEEE column widths; colour-blind-safe palette)
# --------------------------------------------------------------------------------------
OKABE = dict(blue="#0072B2", orange="#D55E00", green="#009E73", pink="#CC79A7", yellow="#E69F00", sky="#56B4E9", grey="#666666")
COL1, COL2 = 3.5, 7.16          # inches: one / two IEEE columns


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.7, "figure.dpi": 150, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _save(fig, name):
    d = ROOT / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  figure: figures/{name}.pdf|png", flush=True)


def _family(sp):
    if sp["enc"] is None:
        t = sp["tune"]
        return ("native full fine-tuning" if t == "full" else "native LoRA" if t.startswith("lora")
                else "native adapters" if t.startswith("adapter") else "native " + t)
    return {"naive": "naive coupling", "linear": "linear bridge", "resampler": "resampler bridge", "calm": "CALM-style cross-attention",
            "padtrunc": "pad/truncate", "ln": "LayerNorm only", "linear_noln": "linear (no LN)"}.get(sp["bridge"], sp["bridge"])


_FAMCOL = {"native full fine-tuning": OKABE["grey"], "native LoRA": OKABE["green"], "native adapters": OKABE["sky"],
           "naive coupling": OKABE["orange"], "linear bridge": OKABE["blue"], "resampler bridge": OKABE["pink"],
           "CALM-style cross-attention": OKABE["yellow"], "pad/truncate": "#999999", "LayerNorm only": "#BBBBBB"}


def _short(sp):
    a = (sp["enc"] or "native")
    s_ = f"{a}→{sp['dec']}" if sp["enc"] else sp["dec"]
    if sp["enc"]:
        s_ += f" {sp['bridge']}"
    if sp["tune"] != "full":
        s_ += f" [{sp['tune']}]"
    return s_


def _bleus(R):
    return np.array([r["metrics"]["bleu"] for r in R.values()])


def make_figures(T):
    try:
        plt = _mpl()
    except Exception as e:
        print(f"matplotlib unavailable ({e}); figures skipped")
        return
    # ---- Fig A: core comparison, all seeds shown ----
    try:
        core = [(k, sp) for k, sp in zip(("native", "naive", "resampler", "linear"),
                                         (S(None, "plbart"), S("codet5", "plbart", "naive"), S("codet5", "plbart", "resampler"),
                                          S("codet5", "plbart", "linear")))]
        gpt = [("native", S(None, "codegpt")), ("naive", S("codet5", "codegpt", "naive")), ("linear", S("codet5", "codegpt", "linear"))]
        panels = [("CodeT5 → PLBART (CONCODE)", core), ("CodeT5 → CodeGPT (CONCODE)", gpt)]
        panels = [(t, [(k, sp) for k, sp in lst if T.get(sys_id(sp))]) for t, lst in panels]
        panels = [(t, l) for t, l in panels if len(l) >= 2]
        if panels:
            fig, axs = plt.subplots(1, len(panels), figsize=(COL2 if len(panels) > 1 else COL1, 2.5), squeeze=False)
            cmap = dict(native=OKABE["grey"], naive=OKABE["orange"], resampler=OKABE["pink"], linear=OKABE["blue"])
            for ax, (title, lst) in zip(axs[0], panels):
                for i, (k, sp) in enumerate(lst):
                    b = _bleus(T[sys_id(sp)])
                    m, sd, ci, n = tci(b)
                    ax.bar(i, m, color=cmap[k], width=0.62, alpha=0.9, yerr=(m - ci[0]) if n > 1 else 0, capsize=3,
                           error_kw=dict(lw=0.8))
                    ax.scatter(i + np.linspace(-0.18, 0.18, len(b)), np.sort(b), s=5, color="k", zorder=3, lw=0)
                    ax.text(i, 0.6, f"n={n}", ha="center", va="bottom", fontsize=6.5, color="white")
                ax.set_xticks(range(len(lst)))
                ax.set_xticklabels([k for k, _ in lst])
                ax.set_ylabel("Test BLEU")
                ax.set_title(title)
            fig.text(0.5, -0.02, "Bars: mean over seeds with 95% t-interval; dots: individual seeds.", ha="center", fontsize=6.5)
            fig.tight_layout()
            _save(fig, "fig_core_comparison")
    except Exception as e:
        print("fig_core_comparison failed:", repr(e))
    # ---- Fig B: forest plot of paired differences ----
    try:
        if PAIR_LOG:
            prim = [k for k in primary_pairs() if k in PAIR_LOG]
            rest = sorted([k for k in PAIR_LOG if k not in prim], key=lambda k: -abs(PAIR_LOG[k]["bb"]))[:14]
            ks = prim + rest
            H = holm([PAIR_LOG[k]["p_perm"] for k in prim]) if prim else []
            Hs = holm([PAIR_LOG[k]["p_seed"] for k in prim]) if prim else []
            Hr = holm([PAIR_LOG[k]["p_perm"] for k in rest]) if rest else []
            Hrs = holm([PAIR_LOG[k]["p_seed"] for k in rest]) if rest else []
            hp, hsd = list(H) + list(Hr), list(Hs) + list(Hrs)
            fig, ax = plt.subplots(figsize=(COL2, 0.24 * len(ks) + 1.1))
            ax.axvspan(-EQUIV_BLEU, EQUIV_BLEU, color="#DDDDDD", alpha=0.6, lw=0, label=f"±{EQUIV_BLEU:g} BLEU equivalence band")
            for i, k in enumerate(ks):
                r = PAIR_LOG[k]
                rob = (not np.isnan(hp[i])) and hp[i] < 0.05 and (not np.isnan(hsd[i])) and hsd[i] < 0.05
                c = OKABE["blue"] if i < len(prim) else OKABE["grey"]
                ax.plot([r["bl"], r["bh"]], [i, i], color=c, lw=1.2)
                ax.plot(r["bb"], i, "o", color=c, mfc=c if rob else "white", ms=4.5, mew=1)
            ax.axvline(0, color="k", lw=0.7)
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels([_plabel(k) for k in ks], fontsize=6.0)
            ax.invert_yaxis()
            if prim:
                ax.axhline(len(prim) - 0.5, color="k", lw=0.5, ls=":")
            ax.set_xlabel("Paired ΔBLEU with 95% bootstrap CI\nfilled = Holm-adjusted p < .05 on test set and seeds; blue = pre-specified family")
            ax.legend(loc="lower right", frameon=False)
            _save(fig, "fig_forest_paired_differences")
    except Exception as e:
        print("fig_forest failed:", repr(e))
    # ---- Fig C: BLEU vs trainable parameters (Pareto) on CONCODE ----
    try:
        pts = []
        for sid, R in T.items():
            r0 = next(iter(R.values()))
            sp = r0["spec"]
            if sp["ds"] != "concode_java" or sp["epochs"] != 3 or sp["nq"] not in (0, 32):
                continue
            pts.append((max(r0["trainable_M"], 0.05), float(np.mean(_bleus(R))), _family(sp), _short(sp), len(R)))
        if len(pts) >= 4:
            fig, ax = plt.subplots(figsize=(COL2 * 0.78, 3.3))
            for fam in sorted({p[2] for p in pts}):
                q = [p for p in pts if p[2] == fam]
                ax.scatter([p[0] for p in q], [p[1] for p in q], s=22, label=fam, color=_FAMCOL.get(fam, "#444444"), edgecolor="k", lw=0.3, zorder=3)
            srt = sorted(pts)
            front, best = [], -1
            for p in srt:
                if p[1] > best:
                    front.append(p)
                    best = p[1]
            ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="k", lw=0.8, ls="--", zorder=2, label="Pareto front")
            for p in front:
                ax.annotate(p[3], (p[0], p[1]), fontsize=5.8, xytext=(3, -8), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("Trainable parameters (M, log scale)")
            ax.set_ylabel("Test BLEU (mean over seeds)")
            ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            _save(fig, "fig_pareto_bleu_vs_trainable")
    except Exception as e:
        print("fig_pareto failed:", repr(e))
    # ---- Fig D: encoder generality ----
    try:
        encs = [e for e in ("codet5", "codet5p", "unixcoder", "roberta", "codebert") if T.get(sys_id(S(e, "plbart", "linear")))]
        if len(encs) >= 2:
            fig, ax = plt.subplots(figsize=(COL1 * 1.15, 2.4))
            w = 0.38
            for j, (br, col) in enumerate((("naive", OKABE["orange"]), ("linear", OKABE["blue"]))):
                m_ = [np.mean(_bleus(T[sys_id(S(e, "plbart", br))])) if T.get(sys_id(S(e, "plbart", br))) else np.nan for e in encs]
                s_ = [np.std(_bleus(T[sys_id(S(e, "plbart", br))]), ddof=1) if T.get(sys_id(S(e, "plbart", br))) and len(T[sys_id(S(e, "plbart", br))]) > 1 else 0 for e in encs]
                ax.bar(np.arange(len(encs)) + (j - 0.5) * w, m_, w, yerr=s_, capsize=2, color=col, label=br, error_kw=dict(lw=0.7))
            nat = T.get(sys_id(S(None, "plbart")))
            if nat:
                ax.axhline(np.mean(_bleus(nat)), color=OKABE["grey"], ls="--", lw=0.8, label="native PLBART")
            ax.set_xticks(range(len(encs)))
            ax.set_xticklabels(encs, rotation=20)
            ax.set_ylabel("Test BLEU (frozen encoder → PLBART)")
            ax.legend(frameon=False)
            _save(fig, "fig_encoder_generality")
    except Exception as e:
        print("fig_encoder failed:", repr(e))
    # ---- Fig E: execution-based correctness (MBPP + HumanEval transfer) ----
    try:
        M = lambda *a, **k: S(*a, ds="mbpp_py", epochs=5, **k)
        cand = [("native PLBART", M(None, "plbart")), ("PLBART+LoRA", M(None, "plbart", tune="lora")), ("PLBART+adapters", M(None, "plbart", tune="adapter64")),
                ("naive", M("codet5", "plbart", "naive")), ("linear", M("codet5", "plbart", "linear")), ("resampler", M("codet5", "plbart", "resampler")),
                ("CALM-style", M("codet5", "plbart", "calm")), ("native CodeGPT", M(None, "codegpt")), ("CodeT5→CodeGPT linear", M("codet5", "codegpt", "linear"))]
        cand = [(n, sp) for n, sp in cand if T.get(sys_id(sp))]
        if len(cand) >= 3:
            fig, ax = plt.subplots(figsize=(COL2 * 0.8, 2.6))
            x = np.arange(len(cand))
            for j, (key, lab, col) in enumerate((("pass1", "MBPP test", OKABE["blue"]), ("he_pass1", "HumanEval (transfer)", OKABE["orange"]))):
                vals = []
                for n_, sp in cand:
                    v = [r["metrics"][key] for r in T[sys_id(sp)].values() if r["metrics"].get(key) is not None]
                    vals.append((np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0) if v else (np.nan, 0))
                ax.bar(x + (j - 0.5) * 0.38, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals], capsize=2, color=col, label=lab, error_kw=dict(lw=0.7))
            ax.set_xticks(x)
            ax.set_xticklabels([n for n, _ in cand], rotation=25, ha="right")
            ax.set_ylabel("pass@1 (%)  executed unit tests")
            ax.legend(frameon=False)
            _save(fig, "fig_functional_correctness")
    except Exception as e:
        print("fig_functional failed:", repr(e))
    # ---- Fig F: validation BPC by epoch (long resampler training) ----
    try:
        cur = {}
        for sid, R in T.items():
            if any(r.get("val_bpc_epochs") for r in R.values()):
                ne = max(len(r["val_bpc_epochs"]) for r in R.values())
                cur[sid] = [np.mean([r["val_bpc_epochs"][i][1] for r in R.values() if len(r["val_bpc_epochs"]) > i]) for i in range(ne)]
        if cur:
            fig, ax = plt.subplots(figsize=(COL1, 2.4))
            for i, (sid, v) in enumerate(sorted(cur.items())):
                ax.plot(range(1, len(v) + 1), v, marker="o", ms=2.5, lw=0.9, label=sid.split("|")[0] + "|" + sid.split("|")[1])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation BPC")
            ax.legend(frameon=False, fontsize=5.5)
            _save(fig, "fig_validation_bpc_curves")
    except Exception as e:
        print("fig_bpc failed:", repr(e))


def analyze_bpc(out):
    from scipy import stats
    recs = load_records("screen")
    if not recs:
        return
    P = defaultdict(dict)
    for r in recs:
        P[(r["enc"], r["dec"])][r["seed"]] = r["bpc"]
    seeds = sorted({s for v in P.values() for s in v})
    pairs = sorted(P)
    full = [s for s in seeds if all(s in P[p] for p in pairs)]
    out += ["## 6. BPC truncation sensitivity of the screening ranking", "",
            f"Complete screening seeds: {full} ({len(pairs)} pairs). Training truncation 128; evaluation truncation varied. "
            "'full' = characters of the untruncated reference (paper definition); 'trunc' = characters of the truncated reference.", ""]
    if not full:
        out.append("(not all 20 pairs finished for any seed yet)\n")
        return
    rk = lambda v: np.argsort(np.argsort(v)) + 1
    vec = lambda s, L, den: np.array([P[p][s][str(L)][den] for p in pairs])
    sel = [pairs.index(("codet5", "plbart")), pairs.index(("codet5", "codegpt"))]
    out += ["| eval truncation | denominator | Spearman vs L=128/full (mean, min over seeds) | Kendall τ (mean) | top-4 overlap (mean) | "
            "rank of CodeT5→PLBART / CodeT5→CodeGPT (mean) | % refs truncated (mean over pairs) |", "|---|---|---|---|---|---|---|"]
    for L in BPC_LENS:
        for den in ("full", "trunc"):
            sp_, kt, ov, r1, r2, ft = [], [], [], [], [], []
            for s in full:
                base, v = vec(s, 128, "full"), vec(s, L, den)
                sp_.append(stats.spearmanr(base, v)[0])
                kt.append(stats.kendalltau(base, v)[0])
                ov.append(len(set(np.argsort(base)[:4]) & set(np.argsort(v)[:4])))
                r1.append(rk(v)[sel[0]]); r2.append(rk(v)[sel[1]])
                ft.append(np.mean([P[p][s][str(L)]["frac_trunc"] for p in pairs]) * 100)
            out.append(f"| {L} | {den} | {np.mean(sp_):.3f}, {np.min(sp_):.3f} | {np.mean(kt):.3f} | {np.mean(ov):.1f} | "
                       f"{np.mean(r1):.1f} / {np.mean(r2):.1f} | {np.mean(ft):.1f} |")
    if len(full) >= 2:
        cc = [stats.spearmanr(vec(a, 128, "full"), vec(b, 128, "full"))[0] for i, a in enumerate(full) for b in full[i + 1:]]
        out += ["", f"Between-seed reliability of the L=128 ranking: mean pairwise Spearman = {np.mean(cc):.3f} (min {np.min(cc):.3f}).", ""]
    mean = np.mean([vec(s, 128, "full") for s in full], 0)
    order = np.argsort(mean)
    out += ["Mean screening BPC at L=128 (rank, pair):", ""]
    out += [f"{i + 1:2d}. {pairs[j][0]}→{pairs[j][1]}: {mean[j]:.3f}" for i, j in enumerate(order)]
    # screening vs final within a fixed decoder (PLBART): the fairer test suggested in Section VI-A
    T = sys_table(load_records("train"))
    xs, ys, names = [], [], []
    for e in ("codet5", "unixcoder", "roberta", "codebert"):
        R = T.get(sys_id(S(e, "plbart", "linear")), {})
        if R and (e, "plbart") in P:
            xs.append(np.mean([vec(s, 128, "full")[pairs.index((e, "plbart"))] for s in full]))
            ys.append(np.mean([r["metrics"]["bleu"] for r in R.values()]))
            names.append(e)
    if len(xs) >= 3:
        out += ["", f"Screening BPC vs final BLEU, fixed decoder PLBART, linear bridge ({', '.join(names)}; n={len(xs)}): "
                f"Spearman ρ = {stats.spearmanr(xs, ys)[0]:+.2f} (descriptive only; lower BPC should mean higher BLEU, i.e. ρ<0)."]
    out.append("")


def analyze_timing(out):
    recs = load_records("timing")
    if not recs:
        return
    G = defaultdict(list)
    for r in recs:
        G[r["sys_id"]].append(r)
    out += ["## 7. Dedicated-hardware timing", "",
            f"GPU: {recs[0]['gpu']}; reps per system: {min(len(v) for v in G.values())}–{max(len(v) for v in G.values())}. "
            "Step time = median of timed steps (tokenisation excluded, includes frozen-encoder forward); full-run estimate = "
            "median step × 1,875 steps. Latency = fixed 64 new tokens, batch 32. Median [IQR] over repetitions.", "",
            "| system | trainable M | total M | step ms | est. 3-epoch train min | latency ms/ex | merged-LoRA latency | train peak GB | latency peak GB | other GPU procs |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    q = lambda x: f"{np.median(x):.1f} [{np.percentile(x, 25):.1f}–{np.percentile(x, 75):.1f}]"
    for sid, v in sorted(G.items()):
        st = [np.median(r["step_ms"]) for r in v]
        lm = [r["lat_merged_ms_per_ex"] for r in v if r["lat_merged_ms_per_ex"]]
        out.append(f"| {sid} | {v[0]['trainable_M']:.1f} | {v[0]['total_M']:.1f} | {q(st)} | {np.median(st) * 1875 / 60000:.1f} | "
                   f"{q([r['lat_ms_per_ex'] for r in v])} | {q(lm) if lm else '–'} | {np.median([r['train_peak_gb'] for r in v]):.2f} | "
                   f"{np.median([r['lat_peak_gb'] for r in v]):.2f} | {sum(len(r['other_procs']) for r in v)} |")
    out.append("")




def backfill_codebleu():
    """Compute CodeBLEU for saved train records that lack it (preds/refs are stored in every record)."""
    n = tot = 0
    for f in sorted(REC.glob("run_*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("kind") != "train" or r["metrics"].get("codebleu") is not None:
            continue
        tot += 1
        v = codebleu_score(r["preds"], r["refs"], r["spec"]["lang"])
        if v is None:
            print("CodeBLEU still unavailable; fix the install (see --stage check) and rerun.")
            return
        r["metrics"]["codebleu"] = v
        atomic_json(r, f)
        n += 1
    print(f"CodeBLEU backfilled for {n}/{tot} records")


def check_codebleu():
    """Import success is not enough (tree-sitter grammars can be missing/incompatible): run it on both languages."""
    good = True
    for lang, code in (("java", "public int get() { return x; }"), ("python", "def f(x):\n    return x + 1")):
        try:
            from codebleu import calc_codebleu
            v = calc_codebleu([[code]], [code], lang=lang)["codebleu"]
            print(f"  ok       codebleu works for {lang} (identical pair -> {v:.2f}, expect ~1.0)")
        except Exception as e:
            good = False
            print(f"  FAIL     codebleu {lang}: {type(e).__name__}: {str(e)[:140]}")
    if not good:
        print("  fix attempts, in order:  pip install -U codebleu tree-sitter-java tree-sitter-python\n"
              "                           pip install codebleu==0.7.0 tree-sitter==0.22.3 tree-sitter-java==0.21.0 tree-sitter-python==0.21.0\n"
              "                           pip install codebleu==0.7.0 tree-sitter==0.23.2 tree-sitter-java==0.23.5 tree-sitter-python==0.23.6\n"
              "  if both fail, ignore it: every other metric is unaffected and `--stage codebleu` backfills later.")
    return good


def pin_revisions():
    """Look up the current full 40-char commit hash of every checkpoint and save it, so later runs (and other
    sessions that copy revs.json) use exactly the same weights.  Needs internet; run once, early."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print('huggingface_hub missing: pip install "transformers==4.46.3" (it brings huggingface_hub) and retry')
        return False
    api, out = HfApi(), {}
    for key, name in CK.items():
        try:
            out[name] = api.model_info(name).sha
            print(f"  {key:15s} {name}  {out[name]}")
        except Exception as e:
            print(f"  FAIL {key:15s} {name}  {type(e).__name__}: {str(e)[:100]}")
    if len(out) == len(CK):
        _REVF.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(_REVF, "w"), indent=1)
        print(f"saved {_REVF}  (loaded automatically on the next run; copy it to pin another session)")
        _reload_revs()
    else:
        print("not saved: some lookups failed")
    return len(out) == len(CK)

# --------------------------------------------------------------------------------------
# 8a. Environment check (packages, checkpoints, datasets)
# --------------------------------------------------------------------------------------
# (import name, pip name, required?)  -- codebleu is optional: score() falls back to codebleu=None without it
PACKAGES = [("torch", "torch", True), ("transformers", "transformers", True), ("datasets", "datasets", True),
            ("sacrebleu", "sacrebleu", True), ("javalang", "javalang", True), ("peft", "peft", True),
            ("sentencepiece", "sentencepiece", True), ("scipy", "scipy", True), ("numpy", "numpy", True),
            ("accelerate", "accelerate", False), ("google.protobuf", "protobuf", False),
            ("codebleu", "codebleu", False)]
EXPECT_DIM = dict(codet5p=768, codet5p770=1024, gpt2m=1024, codet5=768, codet5_small=512, plbart=768, codebert=768, unixcoder=768, roberta=768, gpt2=768,
                  codegpt_java=768, codegpt_python=768)


def check_env(check_data=True):
    import importlib
    ok = True
    print("== packages ==")
    for imp, pipn, req in PACKAGES:
        try:
            mod = importlib.import_module(imp)
            ver = getattr(mod, "__version__", "")
            want = PINNED.get(pipn)
            tag = "" if not want or ver == want else f"   <- pinned {want}"
            print(f"  ok       {pipn:14s} {ver}{tag}")
        except Exception as e:
            ok &= not req
            pin = f"=={PINNED[pipn]}" if pipn in PINNED else ""
            print(f"  {'MISSING ' if req else 'optional'} {pipn:14s} -> pip install \"{pipn}{pin}\"   ({type(e).__name__})")
    try:
        import transformers as _tf
        if int(_tf.__version__.split(".")[0]) != int(PINNED["transformers"].split(".")[0]):
            ok = False
            print(f"  FAIL     transformers {_tf.__version__} is not a 4.x release; install the pinned versions "
                  f"(see the top of this file) and restart the kernel")
    except ImportError:
        pass
    print("== codebleu (optional metric) ==")
    check_codebleu()                          # informational only: does not change the exit status
    if not HAVE_TORCH or not importlib.util.find_spec("transformers"):
        print("transformers/torch missing: skipping checkpoint and dataset checks")
        return ok
    from transformers import AutoConfig
    print("== checkpoints (config + tokenizer; hidden size vs expected) ==")
    for key, name in CK.items():
        try:
            kw = dict(revision=REV[name]) if name in REV else {}
            c = hub_call(AutoConfig.from_pretrained, name, **kw)
            dim = getattr(c, "hidden_size", None) or getattr(c, "d_model", None) or getattr(c, "n_embd", None)
            get_tok(key, "java")
            flag = "ok      " if dim == EXPECT_DIM.get(key) else "DIM?    "
            ok &= dim == EXPECT_DIM.get(key)
            print(f"  {flag} {key:15s} {name}  hidden={dim}")
        except Exception as e:
            ok = False
            print(f"  FAIL     {key:15s} {name}  {type(e).__name__}: {str(e)[:120]}")
    if check_data:
        global SMOKE
        old, SMOKE = SMOKE, True
        print("== datasets (smoke-size load; cleaning + schema) ==")
        for dsn in DS:
            try:
                d = load_data(dsn)
                assert all(len(d[k]) > 0 for k in ("train", "val", "test")), "empty split"
                assert all(a.strip() and b.strip() for k in ("train", "val", "test") for a, b in d[k]), "empty pair"
                print(f"  ok       {dsn:13s} " + str({k: len(d[k]) for k in ("train", "val", "test")}))
            except Exception as e:
                ok = False
                print(f"  FAIL     {dsn:13s} {type(e).__name__}: {str(e)[:160]}")
        SMOKE = old
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED (see above)")
    return ok

# --------------------------------------------------------------------------------------
# 8. Self-test and CLI
# --------------------------------------------------------------------------------------
def selftest():
    h = ["public int get ( ) { return x ; }", "a b c d e f", "x y"]
    r = ["public int get ( ) { return x ; }", "a b c d e g", "x y z w"]
    st = np.stack([bleu_stats(a, b) for a, b in zip(h, r)])
    mine = bleu_from_stats(st.sum(0))[0]
    print("own BLEU:", round(mine, 3))
    try:
        import sacrebleu
        ref = sacrebleu.corpus_bleu(h, [r], tokenize="none").score
        print("sacreBLEU:", round(ref, 3), "OK" if abs(ref - mine) < 1e-6 else "MISMATCH")
    except ImportError:
        print("sacrebleu not installed; skipped")
    if HAVE_TORCH:
        z = torch.arange(12.).view(1, 4, 3)
        m = torch.tensor([[1, 1, 0, 0]])
        zl, ml = left_align(z, m)
        assert ml.tolist() == [[0, 0, 1, 1]] and zl[0, 2].tolist() == z[0, 0].tolist() and zl[0, 3].tolist() == z[0, 1].tolist()
        b = Bridge("resampler", 768, 768, 32)
        print("resampler params (M):", round(sum(p.numel() for p in b.parameters()) / 1e6, 2), "(paper: ~14.8)")
        b = Bridge("linear", 768, 768, 0)
        b2 = Bridge("linear", 512, 768, 0)
        assert b2(torch.zeros(2, 5, 512), torch.ones(2, 5))[0].shape == (2, 5, 768)
        assert Bridge("padtrunc", 768, 512, 0)(torch.zeros(2, 5, 768), None)[0].shape == (2, 5, 512)
        print("linear bridge params (M):", round(sum(p.numel() for p in b.parameters()) / 1e6, 3), "(paper: 0.59)")
        print("left_align OK")
        assert Bridge("calm", 1024, 768, 0, 1.0, 3)(torch.zeros(2, 5, 1024), torch.ones(2, 5))[0].shape == (2, 5, 768)
        a_ = Adapter(16, 4)
        x_ = torch.randn(2, 3, 16)
        assert torch.allclose(a_(x_), x_), "zero-initialised adapter must be the identity"
        try:                                                  # hook plumbing on tiny random models (no download)
            from transformers import GPT2Config, GPT2LMHeadModel, PLBartConfig, PLBartForConditionalGeneration
            from transformers.modeling_outputs import BaseModelOutput
            g = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=32, n_head=2, vocab_size=50, n_positions=64)).eval()
            ids = torch.randint(0, 50, (2, 7))
            base = g(input_ids=ids).logits
            ad = nn.ModuleList([Adapter(32, 4) for _ in range(4)])
            hs = attach_adapters(list(g.transformer.h), ad)
            assert torch.allclose(base, g(input_ids=ids).logits, atol=1e-5), "adapter hooks changed the output at init"
            for h in hs:
                h.remove()
            xb = nn.ModuleList([XBlock(32, heads=2, gate0=0.5) for _ in range(2)]).eval()
            ctx = dict(z=torch.randn(2, 5, 32), m=torch.ones(2, 5, dtype=torch.long))
            hs = attach_calm(list(g.transformer.h), [1, 3], xb, ctx)
            assert not torch.allclose(base, g(input_ids=ids).logits, atol=1e-5), "CALM hooks had no effect (GPT-2)"
            for h in hs:
                h.remove()
            pl = PLBartForConditionalGeneration(PLBartConfig(vocab_size=60, d_model=32, encoder_layers=2, decoder_layers=2,
                    encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                    max_position_embeddings=64)).eval()
            stub = BaseModelOutput(last_hidden_state=torch.zeros(2, 1, 32))
            lab = torch.randint(3, 60, (2, 6))
            b0 = pl(encoder_outputs=stub, attention_mask=torch.ones(2, 1, dtype=torch.long), labels=lab).logits
            xb2 = nn.ModuleList([XBlock(32, heads=2, gate0=0.5) for _ in range(1)]).eval()
            hs = attach_calm(list(pl.model.decoder.layers), [1], xb2, ctx)
            b1 = pl(encoder_outputs=stub, attention_mask=torch.ones(2, 1, dtype=torch.long), labels=lab).logits
            assert not torch.allclose(b0, b1, atol=1e-5), "CALM hooks had no effect (PLBART)"
            for h in hs:
                h.remove()
            print("hook self-test OK (adapters = identity at init; CALM injection active in GPT-2 and PLBART)")
        except ImportError:
            print("transformers not installed: hook self-test skipped")
    r_ = np.array([0.01, 0.04, 0.03])
    assert np.allclose(holm(r_), [0.03, 0.06, 0.06]) and np.allclose(bh_adjust(r_), [0.03, 0.04, 0.04])
    print("Holm / BH OK")
    selftest_resume()


def selftest_resume():
    """Crash-safety plumbing: atomic writes, stale-claim takeover, quarantine of corrupt records, import / pack round trip,
    and (with torch) a training-checkpoint round trip.  Uses a temporary directory; touches nothing else."""
    import tempfile
    global ROOT, REC, _REVF
    old = (ROOT, REC, _REVF)
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ROOT, REC, _REVF = td / "a", td / "a" / "records", td / "a" / "revs.json"
            REC.mkdir(parents=True)
            good = dict(kind="train", sys_id="x", seed=1, metrics={})
            p = REC / "run_000000000001.json"
            atomic_json(good, p)
            assert json.load(open(p)) == good and not list(REC.glob("*.tmp*")), "atomic write"
            # claims: live owner blocks, dead owner / silent owner is taken over
            q = REC / "run_000000000002.json"
            assert claim(q) and not claim(q), "claim exclusivity"
            c = str(q) + ".claim"
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            json.dump(dict(pid=dead.pid, host=socket.gethostname(), t=time.time()), open(c, "w"))
            assert claim_stale(c) and claim(q), "dead-pid takeover"
            old_t = time.time() - 2 * STALE_S
            os.utime(c, (old_t, old_t))
            assert claim_stale(c) and claim(q), "silent-owner takeover"
            release(q)
            assert not os.path.exists(c)
            bump_fail(q)
            bump_fail(q)
            assert n_fails(q) == 2
            clear_fails()
            assert n_fails(q) == 0
            # corrupt record is quarantined
            (REC / "run_000000000003.json").write_text('{"kind": "train", "se')
            assert verify_records() == 1 and (REC / "run_000000000003.json.corrupt").exists()
            # pack -> import into a fresh session
            _REVF.write_text('{"m": "abc"}')
            pack_records()
            z = ROOT / "ucf_ext_records.zip"
            ROOT, REC, _REVF = td / "b", td / "b" / "records", td / "b" / "revs.json"
            REC.mkdir(parents=True)
            assert import_records(z) == (1, 0, 0) and (REC / p.name).exists() and _REVF.exists(), "zip import"
            assert import_records(z) == (0, 1, 0), "import is idempotent"
            ROOT, REC, _REVF = td / "c", td / "c" / "records", td / "c" / "revs.json"
            REC.mkdir(parents=True)
            assert import_records(td / "a")[0] == 1, "directory import"
            print("resume plumbing OK (atomic write, claim takeover, quarantine, pack/import)")
            if HAVE_TORCH:
                lin = nn.Linear(4, 3)
                opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
                sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0)
                sc = make_scaler(False)
                for _ in range(3):
                    opt.zero_grad()
                    lin(torch.randn(2, 4)).sum().backward()
                    opt.step()
                    sch.step()
                rg = np.random.RandomState(5)
                pm = rg.permutation(10)
                meta = dict(a=1)
                cp = td / "ck" / "run_x.pt"
                save_ckpt(cp, lin, opt, sch, sc, rg, pm, 6, 3, [(50, 1.0)], [], 12.5, [], meta)
                ck = load_ckpt(cp, meta)
                assert ck is not None and ck["step"] == 3 and ck["next"] == 6 and (ck["perm"] == pm).all()
                assert load_ckpt(cp, dict(a=2)) is None and not cp.exists(), "fingerprint mismatch must discard the checkpoint"
                lin2 = nn.Linear(4, 3)
                opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-2)
                opt2.load_state_dict(ck["opt"])
                with torch.no_grad():
                    for n_, p_ in lin2.named_parameters():
                        p_.copy_(ck["weights"][n_])
                assert all(torch.equal(a, b) for a, b in zip(lin.parameters(), lin2.parameters()))
                print("checkpoint round trip OK")
    finally:
        ROOT, REC, _REVF = old


def main(argv=None):
    global SMOKE, REC, ROOT, SEEDS_MID, CKPT_INTERVAL_S, CKPT_MIN_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="seeds15", choices=["seeds15", "seeds15_gpt", "encoders", "frozen_dec", "long_resampler",
                                                           "newdata", "dimmismatch", "functional", "baselines", "scale", "data40k", "bpc_trunc", "timing", "analyze", "selftest", "check", "codebleu", "pinrev", "status", "import", "pack", "functional_best"])
    ap.add_argument("--dataset", default=None, choices=list(DS), help="dataset for --stage newdata (default csn_java) or baselines (default concode_java)")
    ap.add_argument("--n_seeds", type=int, default=5, help="seeds (1-5, taken from 13, 42, 2024, 7, 99) for every secondary stage")
    ap.add_argument("--no_figs", action="store_true", help="analyze: skip figure generation")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override the seed list of a stage")
    ap.add_argument("--long_epochs", type=int, nargs="*", default=[6, 12])
    ap.add_argument("--reps", type=int, default=5, help="timing repetitions per system")
    ap.add_argument("--timing_steps", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force_versions", action="store_true", help="run even if transformers is not 4.x")
    ap.add_argument("--list", action="store_true", help="print the job list and exit")
    ap.add_argument("--clear_claims", action="store_true", help="remove ALL unfinished claims (only needed if no worker is alive; dead workers' claims are taken over automatically)")
    ap.add_argument("--resume_from", nargs="*", default=None, help="previous session output: ucf_ext dir, its records/ dir, or ucf_ext_records.zip")
    ap.add_argument("--with_ckpt", action="store_true", help="also import training checkpoints (large) of the previous session")
    ap.add_argument("--no_auto_import", action="store_true", help="do not scan /kaggle/input for previous records")
    ap.add_argument("--retry_failed", action="store_true", help="reset the failure counters of runs that raised before")
    ap.add_argument("--fb_enc", default="codet5", help="functional_best: encoder of the pair (the paper's primary pair by default)")
    ap.add_argument("--fb_dec", default="plbart")
    ap.add_argument("--fb_bridge", default="linear")
    ap.add_argument("--fb_max_epochs", type=int, default=20, help="functional_best: hard epoch cap")
    ap.add_argument("--fb_min_epochs", type=int, default=4)
    ap.add_argument("--fb_patience", type=int, default=4, help="stop after this many epochs without validation improvement")
    ap.add_argument("--fb_target", type=float, default=None, help="optional: stop once VALIDATION pass@1 (%%) reaches this (default: none)")
    ap.add_argument("--fb_no_ref", action="store_true", help="do not also train the native reference with the same protocol")
    ap.add_argument("--ckpt_every_min", type=float, default=15.0, help="training checkpoint interval in minutes for long runs (0 = off)")
    ap.add_argument("--ckpt_min_steps", type=int, default=3000, help="only runs with at least this many steps are checkpointed")
    args = ap.parse_args(argv)
    SMOKE = args.smoke
    SEEDS_MID = list(SEEDS_CORE5[:max(1, min(args.n_seeds, 5))])
    CKPT_INTERVAL_S, CKPT_MIN_STEPS = int(args.ckpt_every_min * 60), args.ckpt_min_steps
    FB.update(enc=args.fb_enc, dec=args.fb_dec, bridge=args.fb_bridge, max_epochs=args.fb_max_epochs,
              min_epochs=args.fb_min_epochs, patience=args.fb_patience, target=args.fb_target, ref=not args.fb_no_ref)
    ROOT = ROOT0 / "smoke" if SMOKE else ROOT0
    REC = ROOT / "records"
    REC.mkdir(parents=True, exist_ok=True)
    if args.clear_claims:
        for c in REC.glob("*.claim"):
            if not Path(str(c)[:-6]).exists():
                c.unlink()
    if args.stage == "selftest":
        return selftest()
    if args.stage not in ("check", "pinrev"):
        do_import(args, explicit=(args.stage == "import"))   # previous sessions' records (idempotent, never overwrites)
    if args.stage == "import":
        return
    if args.stage == "pack":
        return pack_records()
    if args.stage == "status":
        return status(args)
    if args.stage not in ("analyze", "codebleu", "check", "pinrev"):
        version_guard(args.force_versions)
    if args.stage == "codebleu":
        return backfill_codebleu()
    if args.stage == "pinrev":
        sys.exit(0 if pin_revisions() else 1)
    if args.stage == "check":
        sys.exit(0 if check_env() else 1)
    if args.stage == "analyze":
        return analyze(figs=not args.no_figs)
    if not HAVE_TORCH:
        sys.exit("PyTorch is required for this stage")
    HP["max_new"] = 16 if SMOKE else 128          # reset every call: a notebook may run a smoke test and then the real thing
    torch.backends.cudnn.benchmark = False
    if args.stage == "timing":
        return run_timing(args)
    jobs = make_jobs(args.stage, args)
    if args.list:
        for j in jobs:
            print(("exists " if job_path(j).exists() else "todo   ") +
                  (sys_id(j["spec"]) if j["type"] == "train" else f'screen {j["enc"]}>{j["dec"]}') + f' seed={j["seed"]}')
        print(len(jobs), "jobs")
        return
    run_jobs(jobs, args.device, args.retry_failed)


# --------------------------------------------------------------------------------------
# 9. Switch-based launcher (notebook cell / no arguments)
# --------------------------------------------------------------------------------------
JOB_STAGES = ["baselines", "encoders", "functional", "functional_best", "frozen_dec", "scale", "newdata", "data40k", "seeds15",
              "seeds15_gpt", "long_resampler", "dimmismatch", "bpc_trunc"]
SMOKE_PLAN = ["selftest", "check"] + JOB_STAGES + ["timing", "analyze"]
FULL_PLAN = ["pinrev", "status"] + JOB_STAGES + ["timing", "codebleu", "analyze", "pack", "status"]
_FLAGS = dict(smoke="--smoke", no_figs="--no_figs", force_versions="--force_versions", with_ckpt="--with_ckpt",
              retry_failed="--retry_failed", fb_no_ref="--fb_no_ref")
_VALS = dict(n_seeds="--n_seeds", reps="--reps", timing_steps="--timing_steps", ckpt_every_min="--ckpt_every_min",
             ckpt_min_steps="--ckpt_min_steps", fb_enc="--fb_enc", fb_dec="--fb_dec", fb_bridge="--fb_bridge",
             fb_max_epochs="--fb_max_epochs", fb_min_epochs="--fb_min_epochs", fb_patience="--fb_patience", fb_target="--fb_target")
_LISTS = dict(seeds="--seeds", long_epochs="--long_epochs", resume_from="--resume_from")


def _in_notebook():
    if "ipykernel" in sys.modules:
        return True
    try:
        return get_ipython() is not None          # noqa: F821
    except NameError:
        return False


def _argv(stage, device, opt, dataset=None, worker=False):
    a = ["--stage", stage, "--device", device]
    for k, f in _FLAGS.items():
        if opt.get(k):
            a.append(f)
    for k, f in _VALS.items():
        if opt.get(k) is not None:
            a += [f, str(opt[k])]
    for k, f in _LISTS.items():
        if opt.get(k):
            a += [f] + [str(x) for x in opt[k]]
    if dataset:
        a += ["--dataset", dataset]
    if not worker:                                 # only the primary invocation touches claims
        if opt.get("clear_claims"):
            a.append("--clear_claims")
    return a


def _worker_script():
    """Path of a runnable copy of this file: the file itself, or (pasted in a notebook cell) the cell's own source."""
    f = globals().get("__file__")
    if f and os.path.exists(f):
        return f
    try:
        src = get_ipython().user_ns["In"][-1]      # noqa: F821  the cell being executed
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        compile(src, "<cell>", "exec")
        ROOT0.mkdir(parents=True, exist_ok=True)
        out = ROOT0 / "ucf_worker.py"
        out.write_text(src, encoding="utf-8")
        return str(out)
    except Exception as e:
        print(f"[launcher] cannot create a worker script ({type(e).__name__}); the second GPU stays idle. "
              "Fix: save this file as ucf_extensions_kaggle_v3.py and run it with  !python  instead.", flush=True)
        return None


def _spawn(script, argv, tag):
    import subprocess as sp
    pr = sp.Popen([sys.executable, "-u", script] + argv, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1,
                  env={**os.environ, "UCF_NO_SYNC": "1"})          # only the launcher uploads

    def pump():
        for line in pr.stdout:
            print(f"[{tag}] {line.rstrip()}", flush=True)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    pr._pump = th
    return pr


def _call_main(argv, label):
    try:
        main(argv)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[launcher] {label}: stopped ({e.code})", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"[launcher] {label}: FAILED ({type(e).__name__}: {str(e)[:200]}); continuing with the next stage", flush=True)
        try:
            with open(ROOT / "errors.log", "a") as f:
                f.write(f"{time.strftime('%F %T')} stage {label}: {e!r}\n")
        except Exception:
            pass


REQ_PIP = ["transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "datasets>=2.19,<4", "sacrebleu", "javalang",
           "sentencepiece", "scipy", "protobuf"]
OPT_PIP = []                                             # CodeBLEU is handled by ensure_codebleu() (needs tree-sitter grammars)
_CB_TEST = ("from codebleu import calc_codebleu\n"
            "for l, c in (('java', 'public int get() { return x; }'), ('python', 'def f(x):\\n    return x + 1')):\n"
            "    assert calc_codebleu([[c]], [c], lang=l)['codebleu'] > 0.9\n")


def _codebleu_works():
    r = subprocess.run([sys.executable, "-c", _CB_TEST], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-300:]


def ensure_codebleu():
    """Optional metric: install codebleu + tree-sitter grammars until a java and a python test both work."""
    ok, _ = _codebleu_works()
    if ok:
        print("[install] CodeBLEU works", flush=True)
        return True
    # TypeError "an integer is required" = tree-sitter core and grammar packages from different generations (old core expects an
    # int pointer, new grammars return a capsule).  So every attempt pins core + grammars TOGETHER, in both generations.
    for pk in (["-U", "codebleu", "tree-sitter-java", "tree-sitter-python"],
               ["codebleu==0.7.0", "tree-sitter==0.22.3", "tree-sitter-java==0.21.0", "tree-sitter-python==0.21.0"],
               ["codebleu==0.7.0", "tree-sitter==0.23.2", "tree-sitter-java==0.23.5", "tree-sitter-python==0.23.6"],
               ["--no-deps", "codebleu==0.7.0", "tree-sitter==0.23.2", "tree-sitter-java==0.23.5", "tree-sitter-python==0.23.6"]):
        print(f"[install] CodeBLEU not working; pip install {' '.join(pk)}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pk, capture_output=True, text=True)
        if r.returncode != 0:
            print("[install]   pip failed: " + (r.stderr or r.stdout).strip()[-300:], flush=True)
        ok, msg = _codebleu_works()
        if ok:
            print("[install] CodeBLEU works now", flush=True)
            return True
    print(f"[install] CodeBLEU still unavailable ({msg.strip()[-120:]}); stored as None, every other metric is unaffected.", flush=True)
    return False


def _pkg_ok(spec):
    from importlib import metadata
    name = re.split(r"[=<>]", spec)[0]
    try:
        v = metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    if "==" in spec:
        return v == spec.split("==")[1]
    if name == "datasets":
        major = int(v.split(".")[0])
        return 2 <= major < 4
    return True


def ensure_packages():
    """Install what is missing, inside this cell.  Safe because nothing heavy has been imported yet in this process; if
    transformers was imported earlier in the kernel with another version, a kernel restart is the only fix (said clearly)."""
    import subprocess as sp
    need = [x for x in REQ_PIP if not _pkg_ok(x)]
    opt = [x for x in OPT_PIP if not _pkg_ok(x)]
    if not need and not opt:
        print("[install] all packages already in place", flush=True)
        return True
    ok = True
    for label, pk, fatal in (("required", need, True), ("optional", opt, False)):
        if not pk:
            continue
        print(f"[install] pip install ({label}): {' '.join(pk)}", flush=True)
        r = sp.run([sys.executable, "-m", "pip", "install", "-q"] + pk, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:], flush=True)
            print(f"[install] {label} install FAILED (Internet off?).  " + ("Stages that need these packages will fail." if fatal else
                  "Ignoring: CodeBLEU is stored as None and can be backfilled later."), flush=True)
            ok &= not fatal
    importlib.invalidate_caches()
    tf = sys.modules.get("transformers")
    if tf is not None and tf.__version__ != "4.46.3":
        print(f"[install] transformers {tf.__version__} was imported earlier in this kernel; pinned 4.46.3 is installed on disk but "
              "this process keeps the old one.  Restart the kernel (Run > Restart) and run this cell again.", flush=True)
        return False
    return ok


def set_root(path):
    global ROOT, ROOT0, REC, _REVF
    ROOT = ROOT0 = Path(path)
    REC = ROOT / "records"
    os.environ["UCF_OUT"] = str(ROOT)             # workers started later use the same directory
    if "UCF_REVS" not in os.environ:
        _REVF = ROOT / "revs.json"


def run_switches(mode=None, opt=None):
    mode = str(mode or MODE).strip().lower()
    opt = {**OPTIONS, **(opt or {})}
    if mode not in ("smoke", "full"):
        print(f'[launcher] MODE must be "smoke" or "full" (got {mode!r}).', flush=True)
        return
    if opt.get("out_dir"):
        set_root(opt["out_dir"])
    smoke = mode == "smoke"
    opt.update(smoke=smoke)
    if smoke:
        opt.update(seeds=[13], n_seeds=1, retry_failed=True, clear_claims=False)
        shutil.rmtree(ROOT0 / "smoke", ignore_errors=True)        # a smoke test is always a clean test; real results are untouched
    elif not (ROOT0 / "smoke" / "SMOKE_OK").exists():
        print('[launcher] NOTE: no passed smoke test found in this output folder.  Recommended: run MODE = "smoke" once first '
              "(about 1-2 h) so that a broken environment does not waste full-run GPU hours.  Continuing.", flush=True)
    plan = [x for x in (SMOKE_PLAN if smoke else FULL_PLAN) if x not in set(opt.get("skip_stages") or [])]
    gpus = list(opt.get("gpus") or ["cuda:0"])
    if opt.get("auto_install") and not os.environ.get("UCF_NO_INSTALL") and any(
            k in plan for k in JOB_STAGES + ["timing", "check", "codebleu"]):
        if not ensure_packages():
            print("[launcher] stopping: fix the packages first (see above).", flush=True)
            return
        ensure_codebleu()
    nrec0 = len(list(REC.glob("*.json"))) if REC.exists() else 0
    print(f"[launcher] RESUME CHECK: {nrec0} finished runs found in {ROOT0} (previous sessions' output attached as an input is "
          "imported automatically by every stage; finished runs are skipped).", flush=True)
    if opt.get("max_hours") and not smoke:
        os.environ["UCF_DEADLINE"] = str(time.time() + float(opt["max_hours"]) * 3600)
        print(f"[launcher] time budget: {opt['max_hours']} h -> no new run is started after that; records are then packed.", flush=True)
    else:
        os.environ.pop("UCF_DEADLINE", None)
    os.environ["UCF_SYNC_EVERY"] = str(opt.get("sync_every_min") or 20)
    if opt.get("sync_dataset") and not smoke:
        os.environ["UCF_SYNC_DATASET"] = str(opt["sync_dataset"])
        if opt.get("kaggle_username") and opt.get("kaggle_key"):
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = str(opt["kaggle_username"]), str(opt["kaggle_key"])
        print(f"[launcher] auto-sync of records to Kaggle dataset {opt['sync_dataset']!r} is ON.", flush=True)
    if opt.get("allow_val_fallback"):
        os.environ["UCF_ALLOW_VAL_FALLBACK"] = "1"                  # inherited by the worker processes
    if any(k in plan for k in ("baselines", "encoders", "frozen_dec", "scale", "data40k", "seeds15", "seeds15_gpt",
                               "long_resampler", "dimmismatch", "bpc_trunc", "timing")):
        try:
            if not preflight_concode():
                return
        except Exception as e:
            print(f"[launcher] CONCODE preflight could not run ({type(e).__name__}: {str(e)[:150]}); continuing.", flush=True)
    print(f"[launcher] MODE={mode} | stages: {', '.join(plan)} | gpus: {gpus}", flush=True)
    out_of_time = False
    for st in plan:
        if st == "pinrev" and _REVF.exists():
            continue                                   # already pinned
        if _past_deadline() and st not in ("pack", "status"):
            out_of_time = True
            continue                                   # no new work: go straight to pack + status
        runs = ([(st, d) for d in opt["newdata_datasets"]] if st == "newdata" else
                [(st, d) for d in opt["baselines_datasets"]] if st == "baselines" else [(st, None)])
        for stage, ds in runs:
            label = stage + (f" [{ds}]" if ds else "")
            print(f"\n[launcher] ===== {label} =====", flush=True)
            procs = []
            if stage in JOB_STAGES and len(gpus) > 1:
                script = _worker_script()
                if script:
                    for gi, g in enumerate(gpus[1:], 1):
                        procs.append(_spawn(script, _argv(stage, g, opt, ds, worker=True), f"gpu{gi}"))
            try:
                _call_main(_argv(stage, gpus[0], opt, ds), label)
            except BaseException:
                for pr in procs:
                    pr.terminate()
                raise
            for pr in procs:
                pr.wait()
                pr._pump.join(timeout=5)
    if smoke:
        sd = ROOT0 / "smoke"
        nrec = len(list((sd / "records").glob("*.json")))
        err = sd / "errors.log"
        if nrec and not err.exists():
            (sd / "SMOKE_OK").write_text(time.strftime("%F %T"))
            print(f"\n[launcher] SMOKE OK: {nrec} records, no errors.  Set MODE = \"full\" and run the cell again.", flush=True)
        else:
            print(f"\n[launcher] SMOKE FOUND PROBLEMS ({nrec} records; errors: {err if err.exists() else 'none, but nothing ran'}). "
                  "Read the messages above and errors.log, fix, and rerun MODE = \"smoke\" before the full run.", flush=True)
    if out_of_time or _past_deadline():
        print("\n[launcher] TIME BUDGET REACHED - to continue:\n"
              "   1. press Save Version (Save & Run All) so /kaggle/working/ucf_ext is stored as this notebook's output;\n"
              "   2. in the next session: Add Input -> Notebook Output Files -> this notebook;\n"
              "   3. run the same cell again (same MODE and OPTIONS).  Finished runs are skipped, interrupted ones continue.", flush=True)
    _sync_records(force=True)
    print("\n[launcher] done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not _in_notebook():
        main()                                     # command-line use: arguments override the switches
    else:
        run_switches()                             # notebook cell / no arguments: switches decide