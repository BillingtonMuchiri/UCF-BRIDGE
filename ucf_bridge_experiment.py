#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
UCF-Bridge: integrating heterogeneous pretrained code models through a
lightweight, learned bridge  (redesigned experiment)
===============================================================================

WHAT CHANGED VERSUS THE OLD "PROGRESSIVE FUNNEL" SCRIPT (and why)
-------------------------------------------------------------------------------
Old design : new shared 8k BPE tokenizer + random shared embedding + random
             cross-attention, everything trained from scratch-ish on ~8k pairs.
             Result: real-generation BLEU ~1, SVR/ESR = 0 for every pair.
New design : every model KEEPS its own tokenizer and pretrained weights. The
             only new component is a small BRIDGE that maps encoder states A
             into the space the decoder B expects.

                 source text --tokA--> [ Encoder A ] --> BRIDGE --> [ Decoder B ] --tokB--> code

    * seq2seq decoders (CodeT5, PLBART): bridge output is fed to the decoder's
      own PRETRAINED cross-attention as `encoder_outputs`.
    * decoder-only decoders (CodeGPT): bridge output becomes a soft PREFIX
      (inputs_embeds) in front of the target tokens.

MODEL POOLS (only sensible roles)
    encoders : codet5 (T5 encoder), plbart (encoder), codebert, unixcoder
    decoders : codet5, plbart, codegpt   (all real generators)
    (CodeBERT / UniXcoder are encoder-only: never used as decoders.)

BRIDGE VARIANTS  (the "compatibility components" being tested)
    naive      frozen random projection (identity when dims match); NO learned
               alignment.  This is plain "plug encoder states into the decoder".
    linear     Linear + LayerNorm                       (dimension/space alignment)
    mlp        2-layer MLP + LayerNorm                  (non-linear alignment)
    resampler  learned queries cross-attend to encoder  (latent alignment,
               states (Perceiver / Q-Former style)       fixed-length memory)

TRAIN MODES
    bridge_only  encoder + decoder frozen, only the bridge is trained
    dec_tune     encoder frozen; bridge + decoder trained
    full         everything trained

BASELINES (identical data / budget / metrics)
    native:<dec>  the decoder's own model fine-tuned as-is
                  (CodeT5 / PLBART: own encoder+decoder; CodeGPT: source as prompt)
    naive glue    bridge=naive with the SAME train mode as the learned bridges
    homogeneous   e.g. codet5->codet5 THROUGH the bridge (cost of the bridge itself)

PHASES  (python ucf_bridge_experiment.py --phase <name> ...)
    selftest tiny random models + toy tokenizer, NO downloads (~1 min, CPU is fine):
             checks every model/bridge/mode combination runs, gets gradients,
             keeps frozen parts frozen and can generate. RUN THIS VERY FIRST.
    sanity   overfit-a-tiny-set check for the pipeline. RUN THIS FIRST.
    screen   all valid (encoder, decoder) pairs, bridge_only + resampler, short
             fixed-step budget, ranked by validation BITS-PER-CHARACTER
             (comparable across different tokenizers; token loss is NOT).
    main     top-k heterogeneous pairs + homogeneous controls x bridge variants
             x seeds, full training, real generation on the test set; plus the
             native baselines.
    compat   compatibility measures per pair: tokenizer-vocab Jaccard,
             |log2 dim ratio|, linear CKA between the two models' representations.
    analyze  tables (CSV/MD/LaTeX), paired-bootstrap comparisons with Holm
             correction, seed-level CIs, taxonomy-vs-performance correlation,
             figures.
    all      screen -> main -> compat -> analyze  (sanity is separate on purpose)

RESUMABLE: every run writes one JSON; finished runs are skipped. The run key
contains a hash of the hyper-parameters, so changing settings never silently
reuses stale results. Failed runs go to failed/ and are retried next time.

KAGGLE QUICK START (GPU T4 x2, Internet ON)
    !python ucf_bridge_experiment.py --phase selftest
    !python ucf_bridge_experiment.py --phase sanity  --preset quick
    !python ucf_bridge_experiment.py --phase all     --preset medium --time_budget_h 10
    (session ended? just re-run the same command; finished runs are skipped)
    Two GPUs: run two processes with CUDA_VISIBLE_DEVICES=0/1 and
    --shard 0/2 and --shard 1/2 (run `screen` to completion before `main`).

Dataset: CodeXGLUE text-to-code (CONCODE, Java) so numbers are comparable to
published work.  Any other data: --dataset csv --csv file.csv --lang python
(columns: nl, code).

NOTE: written and syntax/unit-checked offline; the first real run happens on
your side. That is exactly what --phase sanity is for.
===============================================================================
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import re
import gc
import json
import math
import time
import copy
import random
import hashlib
import argparse
import traceback
import subprocess
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


def _ensure(pkg_import: str, pip_name: Optional[str] = None, optional: bool = False):
    try:
        __import__(pkg_import)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg_import])
            __import__(pkg_import)
            return True
        except Exception as e:  # pragma: no cover
            if optional:
                print(f"[warn] optional package {pkg_import} unavailable: {e}")
                return False
            raise


for _p, _pip, _opt in [("sacrebleu", "sacrebleu", False), ("datasets", "datasets", False),
                       ("scipy", "scipy", False), ("pandas", "pandas", False),
                       ("matplotlib", "matplotlib", False), ("sentencepiece", "sentencepiece", False)]:
    _ensure(_p, _pip, _opt)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration,
                          PLBartForConditionalGeneration, GPT2LMHeadModel,
                          get_linear_schedule_with_warmup)
from transformers.modeling_outputs import BaseModelOutput
from sacrebleu.metrics import BLEU

# =============================================================================
# 1. CONFIG
# =============================================================================
ENCODERS = {
    "codet5":    dict(family="t5",     ckpt="Salesforce/codet5-base"),
    "plbart":    dict(family="plbart", ckpt="uclanlp/plbart-base"),
    "codebert":  dict(family="bert",   ckpt="microsoft/codebert-base"),
    "unixcoder": dict(family="bert",   ckpt="microsoft/unixcoder-base"),
    # NEVER pretrained on code: a genuinely independent-pretraining encoder,
    # for a stronger test of "heterogeneous, independently pretrained models"
    # and to give the taxonomy-correlation analysis more (and more varied) pairs.
    "roberta":   dict(family="bert",   ckpt="roberta-base"),
}
DECODERS = {
    "codet5": dict(family="t5",     ckpt="Salesforce/codet5-base"),
    "plbart": dict(family="plbart", ckpt="uclanlp/plbart-base"),
    "codegpt": dict(family="gpt2",  ckpt={"java": "microsoft/CodeGPT-small-java-adaptedGPT2",
                                          "python": "microsoft/CodeGPT-small-py-adaptedGPT2"}),
    # plain GPT-2: never pretrained on code, decoder-only counterpart to "roberta" above.
    "gpt2":    dict(family="gpt2",  ckpt="gpt2"),
}
SEQ2SEQ = ("t5", "plbart")
BRIDGES = ("naive", "linear", "mlp", "resampler")
# "lora" is native-only (enc_name is None): a native decoder fine-tuned with a
# PEFT LoRA adapter instead of full fine-tuning, added as an external,
# literature-standard low-parameter-count baseline to compare against the
# bridge approach on trainable-parameter terms. See UCFModel for how it's wired.
MODES = ("bridge_only", "dec_tune", "full", "lora")

PRESETS = {
    # tiny end-to-end smoke test (minutes)
    "quick":  dict(n_train=512, n_val=64, n_test=100, epochs=1, screen_steps=30,
                   seeds=(13, 42), top_k=2, main_bridges=("naive", "resampler")),
    # realistic for 2 x T4 over a few sessions
    "medium": dict(n_train=10000, n_val=500, n_test=1000, epochs=3, screen_steps=400,
                   seeds=(13, 42, 2024), top_k=3, main_bridges=("naive", "linear", "resampler")),
    # paper-scale (many GPU hours)
    "full":   dict(n_train=100000, n_val=2000, n_test=2000, epochs=4, screen_steps=1000,
                   seeds=(13, 42, 2024, 7, 99), top_k=3, main_bridges=("naive", "linear", "mlp", "resampler")),
}


@dataclass
class Cfg:
    out_dir: str = ""
    dataset: str = "concode"            # concode | codesearchnet_python | csv
    csv_path: str = ""
    lang: str = "java"
    n_train: int = 10000
    n_val: int = 500
    n_test: int = 1000
    max_src: int = 256
    max_tgt: int = 128
    batch_size: int = 16
    grad_accum: int = 1
    epochs: int = 3
    lr: float = 5e-5                    # pretrained parts
    lr_bridge: float = 5e-4             # randomly initialised bridge
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    clip: float = 1.0
    amp: str = "auto"                   # auto | fp16 | bf16 | fp32
    num_workers: int = 2
    n_queries: int = 32                 # resampler memory length
    bridge_layers: int = 2
    seeds: Tuple[int, ...] = (13, 42, 2024)
    screen_seeds: Tuple[int, ...] = (13,)
    screen_steps: int = 400
    top_k: int = 3
    main_bridges: Tuple[str, ...] = ("naive", "linear", "resampler")
    main_modes: Tuple[str, ...] = ("dec_tune",)
    lora_baseline: bool = False         # add a native-decoder + PEFT-LoRA baseline per decoder in `main`
    num_beams: int = 1
    codebleu: bool = False
    compat_n: int = 512
    sanity_min_bleu: float = 20.0
    log_every: int = 50
    time_budget_h: float = 0.0          # 0 = unlimited
    max_runs: int = 0                   # 0 = unlimited
    shard: str = "0/1"

    def hp_hash(self, stage: str) -> str:
        d = dict(stage=stage, n_train=self.n_train, n_val=self.n_val, n_test=self.n_test,
                 max_src=self.max_src, max_tgt=self.max_tgt, bs=self.batch_size * self.grad_accum,
                 lr=self.lr, lr_bridge=self.lr_bridge, q=self.n_queries, bl=self.bridge_layers,
                 lang=self.lang, ds=self.dataset, beams=self.num_beams,
                 budget=(self.screen_steps if stage == "screen" else self.epochs),
                 amp=self.amp)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def runs_dir(self): return Path(self.out_dir) / "runs"
    @property
    def failed_dir(self): return Path(self.out_dir) / "failed"
    @property
    def per_ex_dir(self): return Path(self.out_dir) / "per_example"
    @property
    def preds_dir(self): return Path(self.out_dir) / "predictions"
    @property
    def tables_dir(self): return Path(self.out_dir) / "tables"
    @property
    def fig_dir(self): return Path(self.out_dir) / "figures"


def log(cfg: Cfg, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(Path(cfg.out_dir) / "log.txt", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. DATA
# =============================================================================
def load_data(cfg: Cfg) -> Dict[str, List[Dict[str, str]]]:
    """Fixed splits (independent of the run seed) so comparisons are paired."""
    if cfg.dataset == "csv":
        df = pd.read_csv(cfg.csv_path).dropna(subset=["nl", "code"])
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(df))
        n = len(df)
        cut1, cut2 = int(0.8 * n), int(0.9 * n)
        parts = {"train": df.iloc[idx[:cut1]], "val": df.iloc[idx[cut1:cut2]], "test": df.iloc[idx[cut2:]]}
        raw = {k: [{"nl": str(r.nl), "code": str(r.code)} for r in v.itertuples()] for k, v in parts.items()}
    elif cfg.dataset == "codesearchnet_python":
        # Cross-LANGUAGE generalization check (same task/direction as CONCODE: NL -> code,
        # scored with the same BLEU/EM metrics), on a different dataset and language, so a
        # result isn't just an artifact of CONCODE/Java. First line of the docstring is used
        # as the NL description (standard practice for this dataset); requires --lang python
        # (codegpt and plbart both have python-side checkpoints/lang codes already).
        from datasets import load_dataset
        ds = load_dataset("code_search_net", "python")

        def _rows(split):
            out = []
            for r in ds[split]:
                doc = (r.get("func_documentation_string") or "").strip().split("\n")[0].strip()
                code = r.get("func_code_string") or ""
                if doc and code:
                    out.append({"nl": doc, "code": code})
            return out
        raw = {"train": _rows("train"), "val": _rows("validation"), "test": _rows("test")}
    else:
        from datasets import load_dataset
        ds, err = None, None
        for kw in ({}, {"trust_remote_code": True}, {"revision": "refs/convert/parquet"}):
            try:
                ds = load_dataset("google/code_x_glue_tc_text_to_code", **kw)
                break
            except Exception as e:  # noqa
                err = e
        if ds is None:
            raise RuntimeError("Could not load CodeXGLUE text-to-code (CONCODE): "
                               f"{err}\nUse --dataset csv --csv your.csv (columns nl, code).")
        raw = {"train": [{"nl": r["nl"], "code": r["code"]} for r in ds["train"]],
               "val": [{"nl": r["nl"], "code": r["code"]} for r in ds["validation"]],
               "test": [{"nl": r["nl"], "code": r["code"]} for r in ds["test"]]}
    rng = np.random.RandomState(0)
    ok = lambda r: bool(str(r["nl"]).strip()) and bool(str(r["code"]).strip())
    clean = {k: [r for r in v if ok(r)] for k, v in raw.items()}
    dropped = {k: len(raw[k]) - len(clean[k]) for k in raw}
    # Hub copies of CONCODE sometimes ship a test split with empty targets -> 0 usable rows.
    # Fall back to a held-out slice of TRAIN (disjoint from the training subset) so generation
    # metrics (BLEU etc.) can still be computed; the fixed seed keeps splits identical across runs.
    perm = rng.permutation(len(clean["train"]))
    tr_all = [clean["train"][i] for i in perm]
    n_tr = min(cfg.n_train, len(tr_all)) if cfg.n_train else len(tr_all)
    train_rows, spare = tr_all[:n_tr], tr_all[n_tr:]
    out = {"train": train_rows}
    for split, n in (("val", cfg.n_val), ("test", cfg.n_test)):
        rows = clean[split]
        if n and n < len(rows):
            rows = [rows[i] for i in rng.permutation(len(rows))[:n]]
        want = n if n else 500
        if len(rows) < min(want, 50):                      # empty / unusable split
            take = spare[:want]; spare = spare[want:]
            print(f"[data] WARNING: '{split}' split has only {len(rows)} usable rows "
                  f"(dropped {dropped[split]} empty); using {len(take)} held-out train rows instead.")
            rows = take
        out[split] = rows
    if not out["test"] or not out["val"]:
        raise RuntimeError("No usable val/test data. Check the dataset or use --dataset csv.")
    return out


# =============================================================================
# 3. TOKENIZERS / MODEL LOADING (cached on CPU, deep-copied per run)
# =============================================================================
_TOK_CACHE: Dict = {}
_MODEL_CACHE: Dict = {}


def _ckpt(spec, cfg):
    c = spec["ckpt"]
    return c[cfg.lang] if isinstance(c, dict) else c


def _load_tok(ck: str, **kw):
    """AutoTokenizer with a fallback for configs whose `extra_special_tokens`
    (or `additional_special_tokens`) entry newer transformers versions reject
    (seen with Salesforce/codet5-base)."""
    try:
        return AutoTokenizer.from_pretrained(ck, **kw)
    except Exception as first:
        import json as _json, shutil, tempfile
        from pathlib import Path as _P
        from huggingface_hub import snapshot_download
        src = snapshot_download(ck, allow_patterns=["*.json", "*.txt", "*.model"])
        tmp = _P(tempfile.mkdtemp(prefix="tok_"))
        shutil.copytree(src, tmp, dirs_exist_ok=True)
        for fn in ("tokenizer_config.json", "special_tokens_map.json"):
            fp = tmp / fn
            if fp.exists():
                c = _json.load(open(fp))
                c.pop("extra_special_tokens", None)
                _json.dump(c, open(fp, "w"))
        try:
            return AutoTokenizer.from_pretrained(str(tmp), **kw)
        except Exception:
            # last resort: drop the added-token / special-token maps altogether
            for fn in ("special_tokens_map.json", "added_tokens.json"):
                (tmp / fn).unlink(missing_ok=True)
            try:
                return AutoTokenizer.from_pretrained(str(tmp), **kw)
            except Exception:
                raise first


def get_tokenizer(kind: str, name: str, cfg: Cfg):
    spec = (ENCODERS if kind == "enc" else DECODERS)[name]
    ck = _ckpt(spec, cfg)
    key = (ck, spec["family"], cfg.lang)
    if key not in _TOK_CACHE:
        if spec["family"] == "plbart":
            # NL source -> code target
            tok = _load_tok(ck, src_lang="en_XX", tgt_lang=cfg.lang)
        else:
            tok = _load_tok(ck)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
        _TOK_CACHE[key] = tok
    return _TOK_CACHE[key]


def hidden_dim(config) -> int:
    for a in ("d_model", "n_embd", "hidden_size"):
        if hasattr(config, a):
            return int(getattr(config, a))
    raise ValueError("cannot infer hidden size")


def cached_module(kind: str, name: str, cfg: Cfg) -> nn.Module:
    key = (kind, name, cfg.lang)
    if key not in _MODEL_CACHE:
        spec = (ENCODERS if kind == "enc" else DECODERS)[name]
        ck, fam = _ckpt(spec, cfg), spec["family"]
        if kind == "enc":
            if fam == "t5":
                m = T5EncoderModel.from_pretrained(ck)
            elif fam == "plbart":
                full = PLBartForConditionalGeneration.from_pretrained(ck)
                m = full.get_encoder(); m.config = full.config
            else:
                m = AutoModel.from_pretrained(ck)
        else:
            if fam == "t5":
                m = T5ForConditionalGeneration.from_pretrained(ck)
            elif fam == "plbart":
                m = PLBartForConditionalGeneration.from_pretrained(ck)
            else:
                m = GPT2LMHeadModel.from_pretrained(ck)
        _MODEL_CACHE[key] = m.eval()
    return copy.deepcopy(_MODEL_CACHE[key])


def plbart_lang_id(tok, lang: str) -> int:
    tid = tok.convert_tokens_to_ids(f"__{lang}__")
    if tid is None or tid == tok.unk_token_id:
        tid = tok.lang_code_to_id[lang]
    return int(tid)


# =============================================================================
# 4. BRIDGE
# =============================================================================
class Bridge(nn.Module):
    """Maps encoder hidden states (d_in) to the decoder-side space (d_out)."""

    def __init__(self, kind: str, d_in: int, d_out: int, n_queries: int = 32,
                 n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1,
                 out_scale: float = 1.0):
        super().__init__()
        assert kind in BRIDGES, kind
        self.kind, self.n_queries = kind, n_queries
        if kind == "naive":
            if d_in == d_out:
                self.proj = nn.Identity()
            else:
                lin = nn.Linear(d_in, d_out, bias=False)
                for p in lin.parameters():
                    p.requires_grad = False
                self.proj = lin
        elif kind == "linear":
            self.proj = nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out))
        elif kind == "mlp":
            self.proj = nn.Sequential(nn.Linear(d_in, 2 * d_out), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(2 * d_out, d_out), nn.LayerNorm(d_out))
        else:  # resampler
            self.inp = nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out))
            self.queries = nn.Parameter(torch.randn(n_queries, d_out) * 0.02)
            self.blocks = nn.ModuleList([nn.ModuleDict(dict(
                ln_q=nn.LayerNorm(d_out),
                attn=nn.MultiheadAttention(d_out, n_heads, dropout=dropout, batch_first=True),
                ln_f=nn.LayerNorm(d_out),
                ffn=nn.Sequential(nn.Linear(d_out, 4 * d_out), nn.GELU(), nn.Linear(4 * d_out, d_out)),
            )) for _ in range(n_layers)])
            self.out_ln = nn.LayerNorm(d_out)
        # match the scale the decoder is used to (e.g. GPT-2 embeddings are small)
        ln = None
        if kind in ("linear", "mlp"):
            ln = self.proj[-1]
        elif kind == "resampler":
            ln = self.out_ln
        if ln is not None:
            nn.init.constant_(ln.weight, out_scale)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        if self.kind != "resampler":
            return self.proj(h), mask
        kv = self.inp(h)
        q = self.queries.unsqueeze(0).expand(h.size(0), -1, -1)
        pad = ~mask.bool()
        for b in self.blocks:
            qn = b["ln_q"](q)
            q = q + b["attn"](qn, kv, kv, key_padding_mask=pad, need_weights=False)[0]
            q = q + b["ffn"](b["ln_f"](q))
        out = self.out_ln(q)
        return out, torch.ones(out.size()[:2], dtype=mask.dtype, device=mask.device)


# =============================================================================
# 5. THE MODEL  (bridged heterogeneous pair OR native baseline)
# =============================================================================
_LORA_TARGET_MODULES = {  # module-name suffixes PEFT should wrap, per decoder family
    "t5":     ["q", "v"],
    "plbart": ["q_proj", "v_proj"],
    "gpt2":   ["c_attn"],
}


class UCFModel(nn.Module):
    def __init__(self, enc_name: Optional[str], dec_name: str, bridge_kind: Optional[str],
                 mode: str, cfg: Cfg, dec_tok):
        super().__init__()
        assert mode in MODES
        if mode == "lora" and enc_name is not None:
            raise ValueError("'lora' is a native-only baseline mode (enc_name must be None)")
        self.cfg, self.dec_tok = cfg, dec_tok
        self.enc_name, self.dec_name = enc_name, dec_name
        self.native = enc_name is None
        self.dec_family = DECODERS[dec_name]["family"]
        self.seq2seq = self.dec_family in SEQ2SEQ
        self.mode = mode if (self.native and mode == "lora") else ("full" if self.native else mode)
        self.dec = cached_module("dec", dec_name, cfg)
        self.enc = None if self.native else cached_module("enc", enc_name, cfg)
        self.bridge = None
        self._lora = False
        if not self.native:
            d_in = hidden_dim(self.enc.config)
            d_out = hidden_dim(self.dec.config)
            scale = 1.0
            if self.dec_family == "gpt2":
                scale = float(self.dec.transformer.wte.weight.std().item())
            self.bridge = Bridge(bridge_kind, d_in, d_out, cfg.n_queries, cfg.bridge_layers, out_scale=scale)
            if bridge_kind == "naive" and self.mode == "bridge_only":
                raise ValueError("naive bridge has no trainable parameters; use dec_tune/full")
        if self.mode == "lora":
            # External, literature-standard low-parameter baseline: base decoder frozen,
            # small rank-r adapter matrices trained instead. PEFT sets requires_grad itself
            # (adapter params True, everything else False), so we must NOT re-freeze below.
            _ensure("peft", "peft")
            from peft import LoraConfig, get_peft_model, TaskType
            tm = _LORA_TARGET_MODULES.get(self.dec_family)
            if tm is None:
                raise ValueError(f"no LoRA target_modules mapping for decoder family {self.dec_family!r}")
            task = TaskType.SEQ_2_SEQ_LM if self.seq2seq else TaskType.CAUSAL_LM
            self.dec = get_peft_model(self.dec, LoraConfig(task_type=task, r=8, lora_alpha=16,
                                                            lora_dropout=0.05, target_modules=tm))
            self._lora = True
        self.enc_trainable = (not self.native) and self.mode == "full"
        self.dec_trainable = self.mode in ("dec_tune", "full", "lora")
        if self.enc is not None:
            for p in self.enc.parameters():
                p.requires_grad = self.enc_trainable
        if not self._lora:
            for p in self.dec.parameters():
                p.requires_grad = self.dec_trainable
        if self.dec_family == "plbart":
            self.start_id = plbart_lang_id(dec_tok, cfg.lang)
        elif self.dec_family == "t5":
            c = self.dec.config
            sid = getattr(c, "decoder_start_token_id", None)
            self.start_id = sid if sid is not None else c.pad_token_id
        if self.seq2seq:
            # be explicit: never rely on config/generation_config carrying these ids
            self.dec.config.is_encoder_decoder = True
            self.dec.config.decoder_start_token_id = self.start_id
            gc = getattr(self.dec, "generation_config", None)
            if gc is not None:
                gc.decoder_start_token_id = self.start_id
        self.bos = dec_tok.bos_token_id if dec_tok.bos_token_id is not None else dec_tok.eos_token_id
        self.eos = dec_tok.eos_token_id
        self.pad = dec_tok.pad_token_id

    def train(self, mode: bool = True):
        super().train(mode)
        if self.enc is not None and not self.enc_trainable:
            self.enc.eval()
        if not self.dec_trainable:
            self.dec.eval()
        return self

    # ---- parameter bookkeeping ------------------------------------------
    def param_groups(self):
        bridge = [p for p in (self.bridge.parameters() if self.bridge is not None else []) if p.requires_grad]
        backbone = [p for n, p in self.named_parameters()
                    if p.requires_grad and not n.startswith("bridge.")]
        groups = []
        if backbone:
            groups.append(dict(params=backbone, lr=self.cfg.lr))
        if bridge:
            groups.append(dict(params=bridge, lr=self.cfg.lr_bridge))
        return groups

    def count_params(self) -> Dict[str, int]:
        def used(n):  # the unused encoder half of a seq2seq decoder-model is not counted
            return not (self.seq2seq and not self.native and n.startswith(("dec.encoder.", "dec.model.encoder.")))
        total = sum(p.numel() for n, p in self.named_parameters() if used(n))
        train = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and used(n))
        return dict(total_params=total, trainable_params=train)

    # ---- memory from the (bridged) encoder ---------------------------------
    def _memory(self, src):
        ids, am = src["input_ids"], src["attention_mask"]
        if self.enc_trainable:
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        else:
            with torch.no_grad():
                h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        return self.bridge(h, am)

    # ---- decoder-only helpers ----------------------------------------------
    @staticmethod
    def _left_align(emb, mask):
        idx = torch.argsort(mask.int(), dim=1, stable=True)      # zeros first, order kept
        emb = torch.gather(emb, 1, idx.unsqueeze(-1).expand(-1, -1, emb.size(-1)))
        return emb, torch.gather(mask, 1, idx)

    def _gpt_prefix(self, src):
        if self.native:
            emb = self.dec.transformer.wte(src["input_ids"])
            mask = src["attention_mask"]
        else:
            emb, mask = self._memory(src)
        return self._left_align(emb, mask)

    # ---- loss ----------------------------------------------------------------
    def loss_and_count(self, batch):
        src = batch["src"]
        if self.seq2seq:
            labels = batch["labels"]
            if self.native:
                out = self.dec(input_ids=src["input_ids"], attention_mask=src["attention_mask"], labels=labels)
            else:
                mem, mm = self._memory(src)
                out = self.dec(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                               attention_mask=mm, labels=labels)
            return out.loss, (labels != -100).sum()
        # decoder-only: [prefix][bos code eos]
        pre, pmask = self._gpt_prefix(src)
        tids, tmask = batch["tgt_ids"], batch["tgt_mask"]
        temb = self.dec.transformer.wte(tids)
        emb = torch.cat([pre.to(temb.dtype), temb], 1)
        am = torch.cat([pmask, tmask], 1)
        pos = (am.long().cumsum(-1) - 1).clamp(min=0)
        lab = tids.masked_fill(tmask == 0, -100).clone()
        lab[:, 0] = -100                                    # bos is given, not predicted
        lab = torch.cat([torch.full(pmask.shape, -100, dtype=lab.dtype, device=lab.device), lab], 1)
        out = self.dec(inputs_embeds=emb, attention_mask=am, position_ids=pos, labels=lab)
        return out.loss, (lab[:, 1:] != -100).sum()

    # ---- generation -------------------------------------------------------------
    @torch.no_grad()
    def generate(self, batch, max_new_tokens: int, num_beams: int = 1):
        src = batch["src"]
        kw = dict(max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
        if self.seq2seq:
            kw["decoder_start_token_id"] = self.start_id
            if self.native:
                return self.dec.generate(input_ids=src["input_ids"], attention_mask=src["attention_mask"], **kw)
            mem, mm = self._memory(src)
            return self.dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                                     attention_mask=mm, **kw)
        pre, pmask = self._gpt_prefix(src)
        b = pre.size(0)
        bos = torch.full((b, 1), self.bos, dtype=torch.long, device=pre.device)
        emb = torch.cat([pre, self.dec.transformer.wte(bos).to(pre.dtype)], 1)
        am = torch.cat([pmask, torch.ones(b, 1, dtype=pmask.dtype, device=pmask.device)], 1)
        if num_beams == 1:
            return self._greedy_gpt(emb, am, max_new_tokens)
        return self.dec.generate(inputs_embeds=emb, attention_mask=am, pad_token_id=self.pad,
                                 eos_token_id=self.eos, **kw)

    @torch.no_grad()
    def _greedy_gpt(self, emb, am, max_new_tokens: int):
        """Greedy decoding with explicit position_ids identical to the ones used
        in loss_and_count (cumsum of the attention mask), so train and inference
        cannot diverge. Returns only the newly generated ids."""
        b = emb.size(0)
        pos = (am.long().cumsum(-1) - 1).clamp(min=0)
        out = self.dec(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=True)
        past, last_pos = out.past_key_values, pos[:, -1]
        nxt = out.logits[:, -1].argmax(-1)
        gen = [nxt]
        done = nxt == self.eos
        for _ in range(max_new_tokens - 1):
            if bool(done.all()):
                break
            am = torch.cat([am, torch.ones(b, 1, dtype=am.dtype, device=am.device)], 1)
            last_pos = last_pos + 1
            e = self.dec.transformer.wte(nxt).unsqueeze(1).to(emb.dtype)
            out = self.dec(inputs_embeds=e, attention_mask=am, position_ids=last_pos.unsqueeze(1),
                           past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            gen.append(nxt)
            done = done | (nxt == self.eos)
        return torch.stack(gen, 1)


# =============================================================================
# 6. COLLATION
# =============================================================================
class Collator:
    def __init__(self, cfg: Cfg, enc_tok, dec_tok, dec_family: str, native: bool):
        self.cfg, self.enc_tok, self.dec_tok = cfg, enc_tok, dec_tok
        self.fam, self.native = dec_family, native
        self.bos = dec_tok.bos_token_id if dec_tok.bos_token_id is not None else dec_tok.eos_token_id
        self.eos, self.pad = dec_tok.eos_token_id, dec_tok.pad_token_id

    def __call__(self, batch):
        nl = [b["nl"] for b in batch]
        code = [b["code"] for b in batch]
        st = self.dec_tok if self.native else self.enc_tok
        s = st(nl, max_length=self.cfg.max_src, truncation=True, padding=True, return_tensors="pt")
        out = {"src": {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]},
               "chars": int(sum(len(c) for c in code))}
        if self.fam in SEQ2SEQ:
            lab = self.dec_tok(text_target=code, max_length=self.cfg.max_tgt, truncation=True,
                               padding=True, return_tensors="pt")
            out["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
        else:
            seqs = []
            for c in code:
                ids = self.dec_tok(c, add_special_tokens=False)["input_ids"][: self.cfg.max_tgt - 2]
                seqs.append([self.bos] + ids + [self.eos])
            L = max(len(x) for x in seqs)
            t = torch.full((len(seqs), L), self.pad, dtype=torch.long)
            m = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, x in enumerate(seqs):
                t[i, :len(x)] = torch.tensor(x); m[i, :len(x)] = 1
            out["tgt_ids"], out["tgt_mask"] = t, m
        return out


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(dev) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(dev)
        else:
            out[k] = v
    return out


# =============================================================================
# 7. METRICS
# =============================================================================
_CODE_TOK = re.compile(r"\w+|[^\w\s]")


def ctok(s: str) -> str:
    return " ".join(_CODE_TOK.findall(s))


def per_example_scores(preds: List[str], refs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    sb = BLEU(tokenize="none", effective_order=True)
    bleu = np.array([sb.sentence_score(ctok(p), [ctok(r)]).score if p.strip() else 0.0
                     for p, r in zip(preds, refs)], dtype=np.float64)
    em = np.array([float(ctok(p) == ctok(r)) for p, r in zip(preds, refs)], dtype=np.float64)
    return bleu, em


def corpus_bleu(preds: List[str], refs: List[str]) -> float:
    return float(BLEU(tokenize="none").corpus_score([ctok(p) for p in preds], [[ctok(r) for r in refs]]).score)


def maybe_codebleu(preds, refs, lang) -> Optional[float]:
    try:
        _ensure("codebleu", "codebleu", optional=True)
        from codebleu import calc_codebleu
        return float(calc_codebleu([[r] for r in refs], preds, lang=lang)["codebleu"]) * 100.0
    except Exception as e:  # noqa
        print(f"[warn] CodeBLEU unavailable: {e}")
        return None


def degeneracy_stats(preds: List[str], refs: List[str]) -> Dict[str, float]:
    empty = float(np.mean([not p.strip() for p in preds]))
    ratio = float(np.mean([len(ctok(p).split()) / max(1, len(ctok(r).split())) for p, r in zip(preds, refs)]))
    uniq = len(set(preds)) / max(1, len(preds))
    return dict(empty_rate=empty, len_ratio=ratio, unique_pred_rate=uniq)


# =============================================================================
# 8. TRAIN / EVAL
# =============================================================================
class TrainingStalled(RuntimeError):
    pass


def amp_setup(cfg: Cfg):
    if not torch.cuda.is_available() or cfg.amp == "fp32":
        return None, False
    cap = torch.cuda.get_device_capability()
    if cfg.amp == "bf16" or (cfg.amp == "auto" and cap[0] >= 8):
        return torch.bfloat16, False
    return torch.float16, True


def train_model(model: UCFModel, train_rows, cfg: Cfg, collate, seed: int,
                epochs: Optional[int] = None, max_steps: Optional[int] = None, tag: str = ""):
    dev = next(model.parameters()).device
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(train_rows, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    collate_fn=collate, num_workers=cfg.num_workers, generator=g)
    if len(dl) == 0:
        raise ValueError(f"only {len(train_rows)} training rows for batch_size {cfg.batch_size}")
    steps_per_epoch = max(1, len(dl) // cfg.grad_accum)
    total = max_steps if max_steps else steps_per_epoch * (epochs or cfg.epochs)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)
    dtype, use_scaler = amp_setup(cfg)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):              # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    model.train()
    curve, recent, skipped, step, micro, t0 = [], [], 0, 0, 0, time.time()
    first, done = [], False
    while not done:
        for batch in dl:
            batch = to_device(batch, dev)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                loss, _ = model.loss_and_count(batch)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True); micro = 0
                if skipped > max(20, 0.05 * (step + skipped)):
                    raise TrainingStalled(f"{skipped} non-finite losses (try --amp fp32)")
                continue
            scaler.scale(loss / cfg.grad_accum).backward()
            micro += 1
            lv = float(loss.detach())
            recent.append(lv)
            if len(first) < 10:
                first.append(lv)
            if micro % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups for p in gr["params"]], cfg.clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0 or step == total:
                    m = float(np.mean(recent[-cfg.log_every * cfg.grad_accum:]))
                    curve.append((step, m))
                    print(f"    {tag} step {step}/{total} loss {m:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if step >= total:
                    done = True
                    break
    return dict(steps=step, skipped_steps=skipped, first_loss=float(np.mean(first)),
                last_loss=float(np.mean(recent[-10:])), curve=curve, train_time_s=time.time() - t0)


@torch.no_grad()
def eval_val(model: UCFModel, rows, cfg: Cfg, collate) -> Dict[str, float]:
    dev = next(model.parameters()).device
    model.eval()
    dl = DataLoader(rows, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate,
                    num_workers=cfg.num_workers)
    dtype, _ = amp_setup(cfg)
    nll, ntok, chars = 0.0, 0, 0
    for batch in dl:
        batch = to_device(batch, dev)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            loss, n = model.loss_and_count(batch)
        nll += float(loss) * int(n); ntok += int(n); chars += batch["chars"]
    return dict(val_loss_tok=nll / max(1, ntok), val_bpc=nll / (math.log(2) * max(1, chars)))


@torch.no_grad()
def generate_all(model: UCFModel, rows, cfg: Cfg, collate, dec_tok) -> List[str]:
    dev = next(model.parameters()).device
    model.eval()
    dl = DataLoader(rows, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate,
                    num_workers=cfg.num_workers)
    dtype, _ = amp_setup(cfg)
    preds = []
    for batch in dl:
        batch = to_device(batch, dev)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            out = model.generate(batch, cfg.max_tgt, cfg.num_beams)
        seqs = out.tolist()
        if model.dec_family == "gpt2":
            seqs = [s[:s.index(model.eos)] if model.eos in s else s for s in seqs]
        preds += [t.strip() for t in dec_tok.batch_decode(seqs, skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=False)]
    return preds


# =============================================================================
# 9. RUN SPECS / RUNNER
# =============================================================================
@dataclass
class RunSpec:
    stage: str                    # sanity | screen | main
    enc: Optional[str]            # None = native baseline
    dec: str
    bridge: Optional[str]
    mode: str
    seed: int

    def key(self, cfg: Cfg) -> str:
        return (f"{self.stage}__{self.enc or 'native'}__{self.dec}__{self.bridge or 'none'}__"
                f"{self.mode}__s{self.seed}__{cfg.hp_hash(self.stage)}")


def build_model_and_data(spec: RunSpec, cfg: Cfg):
    dec_tok = get_tokenizer("dec", spec.dec, cfg)
    enc_tok = None if spec.enc is None else get_tokenizer("enc", spec.enc, cfg)
    fam = DECODERS[spec.dec]["family"]
    collate = Collator(cfg, enc_tok, dec_tok, fam, spec.enc is None)
    set_seed(spec.seed)
    model = UCFModel(spec.enc, spec.dec, spec.bridge, spec.mode, cfg, dec_tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev), collate, dec_tok


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def run_one(spec: RunSpec, cfg: Cfg, data) -> Optional[Dict]:
    key = spec.key(cfg)
    out_path = cfg.runs_dir / f"{key}.json"
    if out_path.exists():
        return json.load(open(out_path))
    log(cfg, f"RUN {key}")
    model = None
    try:
        model, collate, dec_tok = build_model_and_data(spec, cfg)
        pc = model.count_params()
        if spec.stage == "screen":
            tr = train_model(model, data["train"], cfg, collate, spec.seed, max_steps=cfg.screen_steps, tag="screen")
            metrics = eval_val(model, data["val"], cfg, collate)
        else:
            tr = train_model(model, data["train"], cfg, collate, spec.seed, tag=spec.stage)
            metrics = eval_val(model, data["val"], cfg, collate)
            preds = generate_all(model, data["test"], cfg, collate, dec_tok)
            refs = [r["code"] for r in data["test"]]
            b, e = per_example_scores(preds, refs)
            metrics.update(bleu=corpus_bleu(preds, refs), sent_bleu=float(b.mean()), em=float(e.mean() * 100),
                           **degeneracy_stats(preds, refs))
            if cfg.codebleu:
                metrics["codebleu"] = maybe_codebleu(preds, refs, cfg.lang)
            cfg.per_ex_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cfg.per_ex_dir / f"{key}.npz", bleu=b, em=e)
            cfg.preds_dir.mkdir(parents=True, exist_ok=True)
            with open(cfg.preds_dir / f"{key}.jsonl", "w") as f:
                for p, r in zip(preds, refs):
                    f.write(json.dumps(dict(pred=p, ref=r)) + "\n")
        res = dict(spec=asdict(spec), key=key, hp=cfg.hp_hash(spec.stage), metrics=metrics, train=tr, **pc)
        save_json(out_path, res)
        log(cfg, f"  done: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))
        return res
    except Exception as ex:  # noqa
        save_json(cfg.failed_dir / f"{key}.json", dict(spec=asdict(spec), error=str(ex),
                                                       trace=traceback.format_exc()))
        log(cfg, f"  FAILED: {ex}")
        return None
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_list(specs: List[RunSpec], cfg: Cfg, data, t_start: float):
    i_sh, n_sh = map(int, cfg.shard.split("/"))
    specs = [s for j, s in enumerate(specs) if j % n_sh == i_sh]
    todo = [s for s in specs if not (cfg.runs_dir / f"{s.key(cfg)}.json").exists()]
    log(cfg, f"{len(specs)} runs in this shard, {len(todo)} still to do")
    launched = 0
    for s in todo:
        if cfg.time_budget_h and (time.time() - t_start) / 3600 > cfg.time_budget_h:
            log(cfg, "time budget reached; re-run the same command to continue"); return False
        if cfg.max_runs and launched >= cfg.max_runs:
            log(cfg, "max_runs reached"); return False
        run_one(s, cfg, data); launched += 1
    return True


# =============================================================================
# 10. PHASES
# =============================================================================
def valid_pairs() -> List[Tuple[str, str]]:
    return [(e, d) for e in ENCODERS for d in DECODERS]


def is_hetero(e: str, d: str) -> bool:
    return ENCODERS[e]["family"] != DECODERS[d]["family"]


def phase_sanity(cfg: Cfg, data) -> bool:
    """Overfit 64 examples. If this fails, do NOT spend compute on the real runs."""
    tiny = data["train"][:64]
    small = copy.deepcopy(cfg); small.log_every = 20; small.num_beams = 1
    small.lr, small.lr_bridge = 1e-4, 1e-3          # memorisation needs a bigger step than fine-tuning
    tests = [(None, "codet5", None, "full"),
             ("unixcoder", "codet5", "resampler", "dec_tune"),
             ("codebert", "codegpt", "resampler", "dec_tune"),
             ("codet5", "plbart", "linear", "dec_tune"),
             # genuinely independent pretraining (neither model ever saw code):
             ("roberta", "gpt2", "linear", "dec_tune")]
    if cfg.lora_baseline:
        tests.append((None, "codegpt", None, "lora"))  # smallest decoder: cheapest place to catch a PEFT bug
    ok_all = True
    for enc, dec, br, mode in tests:
        spec = RunSpec("sanity", enc, dec, br, mode, 13)
        model = None
        try:
            model, collate, dec_tok = build_model_and_data(spec, small)
            # a randomly initialised bridge (or LoRA adapter) needs longer than a fully fine-tuned
            # native model to memorise 64 pairs
            tr = train_model(model, tiny, small, collate, 13,
                              max_steps=(500 if (enc is not None or mode == "lora") else 150), tag="sanity")
            preds = generate_all(model, tiny[:16], small, collate, dec_tok)
            refs = [r["code"] for r in tiny[:16]]
            b = corpus_bleu(preds, refs)
            ok = tr["last_loss"] < 0.5 * tr["first_loss"] and b >= cfg.sanity_min_bleu and tr["skipped_steps"] == 0
            log(cfg, f"SANITY {enc or 'native'}->{dec} [{br}/{mode}] loss {tr['first_loss']:.2f}->{tr['last_loss']:.2f} "
                     f"train-set BLEU {b:.1f} skipped {tr['skipped_steps']} => {'PASS' if ok else 'FAIL'}")
            # Prefix-dependence probe: with the source replaced by another example's source, the
            # loss must go UP. If it does not, the decoder memorised targets from its own previous
            # tokens (posterior collapse) and ignores the bridge -> low loss but generic generation.
            if br is not None:
                try:
                    model.eval()
                    dev = next(model.parameters()).device
                    bt = to_device(collate(tiny[:16]), dev)
                    sh = {k: v.roll(1, 0) for k, v in bt["src"].items()}
                    bt2 = dict(bt); bt2["src"] = sh
                    with torch.no_grad():
                        l_true = float(model.loss_and_count(bt)[0]); l_shuf = float(model.loss_and_count(bt2)[0])
                    gap = l_shuf - l_true
                    log(cfg, f"   prefix-dependence: loss true-src {l_true:.3f} vs shuffled-src {l_shuf:.3f} (gap {gap:+.3f})")
                    if gap < 0.3:
                        # The teacher-forced loss gap is a WEAK probe once 64 targets are memorised
                        # (each memorised continuation is fixed by its own previous tokens, so only the
                        # first few tokens depend on the source). Decide with a generation test instead:
                        # decode the same 16 targets with each row's source swapped for another row's.
                        # A decoder that really ignores the bridge gives the same output either way.
                        n16 = min(16, len(tiny))
                        swapped = [dict(tiny[i], nl=tiny[(i + 1) % n16]["nl"]) for i in range(n16)]
                        preds_sh = generate_all(model, swapped, small, collate, dec_tok)
                        b_sh = corpus_bleu(preds_sh, [r["code"] for r in tiny[:n16]])
                        log(cfg, f"   swapped-source generation BLEU {b_sh:.1f} vs true-source {b:.1f}")
                        if b_sh >= 0.8 * b:
                            ok = False
                            log(cfg, "   -> decoder ignores the bridge output (generation unchanged by swapping the source)")
                        else:
                            log(cfg, "   -> loss gap small but generation depends on the source: bridge IS used, not a failure")
                except Exception as ex3:  # noqa
                    log(cfg, f"   (prefix probe failed: {ex3})")
            ok_all &= ok
            if not ok:
                log(cfg, f"   example pred: {preds[0][:120]!r}\n   example ref : {refs[0][:120]!r}")
                try:  # eval-mode teacher-forced loss vs train-mode loss: separates dropout/AMP from decode bugs
                    model.eval()
                    dev = next(model.parameters()).device
                    bt = to_device(collate(tiny[:16]), dev)
                    dtype, _ = amp_setup(small)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                        l, _n = model.loss_and_count(bt)
                    log(cfg, f"   eval-mode teacher-forced loss on train subset: {float(l):.3f}")
                except Exception as ex2:  # noqa
                    log(cfg, f"   (diagnostic failed: {ex2})")
        except Exception as ex:  # noqa
            ok_all = False
            log(cfg, f"SANITY {enc or 'native'}->{dec} CRASHED: {ex}\n{traceback.format_exc()}")
        finally:
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log(cfg, "SANITY OVERALL: " + ("PASS - safe to run screen/main" if ok_all else
                                   "FAIL - fix the failing configuration(s) first (see hints in README)"))
    return ok_all


def phase_screen(cfg: Cfg, data, t0):
    specs = [RunSpec("screen", e, d, "resampler", "bridge_only", s)
             for (e, d) in valid_pairs() for s in cfg.screen_seeds]
    return run_list(specs, cfg, data, t0)


def load_runs(cfg: Cfg, stage: str) -> pd.DataFrame:
    rows = []
    for p in sorted(cfg.runs_dir.glob(f"{stage}__*.json")):
        r = json.load(open(p))
        if r["hp"] != cfg.hp_hash(stage):
            continue
        row = dict(r["spec"]); row.update(r["metrics"]); row.update(
            trainable_params=r["trainable_params"], total_params=r["total_params"],
            skipped_steps=r["train"]["skipped_steps"], train_time_s=r["train"]["train_time_s"], key=r["key"])
        rows.append(row)
    return pd.DataFrame(rows)


def screen_ranking(cfg: Cfg) -> pd.DataFrame:
    df = load_runs(cfg, "screen")
    if df.empty:
        raise RuntimeError("No screening results found; run --phase screen first.")
    g = df.groupby(["enc", "dec"]).agg(val_bpc=("val_bpc", "mean"), n=("seed", "count")).reset_index()
    g["hetero"] = [is_hetero(e, d) for e, d in zip(g.enc, g.dec)]
    return g.sort_values("val_bpc").reset_index(drop=True)


def main_pairs(cfg: Cfg) -> List[Tuple[str, str]]:
    rk = screen_ranking(cfg)
    het = [(r.enc, r.dec) for r in rk.itertuples() if r.hetero][: cfg.top_k]
    controls = [(e, d) for (e, d) in valid_pairs() if e == d]        # homogeneous controls
    return het + [c for c in controls if c not in het]


def phase_main(cfg: Cfg, data, t0):
    pairs = main_pairs(cfg)
    log(cfg, f"main pairs (top-{cfg.top_k} hetero + homogeneous controls): {pairs}")
    specs = []
    for s in cfg.seeds:
        for d in sorted({d for _, d in pairs}):
            specs.append(RunSpec("main", None, d, None, "full", s))            # native baselines
            if cfg.lora_baseline:
                specs.append(RunSpec("main", None, d, None, "lora", s))        # native + PEFT LoRA (external baseline)
        for (e, d) in pairs:
            for br in cfg.main_bridges:
                for mode in cfg.main_modes:
                    if br == "naive" and mode == "bridge_only":
                        continue
                    specs.append(RunSpec("main", e, d, br, mode, s))
    log(cfg, f"planned main runs: {len(specs)}")
    return run_list(specs, cfg, data, t0)


# ---------------------------------------------------------------- compatibility
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


def norm_vocab(tok) -> set:
    return {t for t in (re.sub(r"^[Ġ▁]+", "", k) for k in tok.get_vocab().keys()) if t}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


@torch.no_grad()
def pooled_reps(module, tok, texts, cfg: Cfg, reader: str, bs: int = 32) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(dev).eval()
    outs = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], max_length=cfg.max_tgt, truncation=True, padding=True, return_tensors="pt").to(dev)
        ids, am = b["input_ids"], b["attention_mask"]
        if reader == "gpt2":
            h = module.transformer(input_ids=ids, attention_mask=am).last_hidden_state
        elif reader == "seq2seq":
            h = module.get_encoder()(input_ids=ids, attention_mask=am).last_hidden_state
        else:
            h = module(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        outs.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    module.cpu()
    return np.concatenate(outs, 0)


def phase_compat(cfg: Cfg, data):
    texts = [r["code"] for r in data["train"][: cfg.compat_n]]
    reps, vocabs, dims = {}, {}, {}
    for n in ENCODERS:
        tok = get_tokenizer("enc", n, cfg); m = cached_module("enc", n, cfg)
        reps[("enc", n)] = pooled_reps(m, tok, texts, cfg, "encoder")
        vocabs[("enc", n)] = norm_vocab(tok); dims[("enc", n)] = hidden_dim(m.config); del m
    for n in DECODERS:
        tok = get_tokenizer("dec", n, cfg); m = cached_module("dec", n, cfg)
        reps[("dec", n)] = pooled_reps(m, tok, texts, cfg, "gpt2" if DECODERS[n]["family"] == "gpt2" else "seq2seq")
        vocabs[("dec", n)] = norm_vocab(tok); dims[("dec", n)] = hidden_dim(m.config); del m
    rows = []
    for e, d in valid_pairs():
        rows.append(dict(enc=e, dec=d, hetero=is_hetero(e, d),
                         cka=linear_cka(reps[("enc", e)], reps[("dec", d)]),
                         vocab_jaccard=jaccard(vocabs[("enc", e)], vocabs[("dec", d)]),
                         dim_gap=abs(math.log2(dims[("enc", e)] / dims[("dec", d)]))))
    df = pd.DataFrame(rows)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.tables_dir / "compat_measures.csv", index=False)
    log(cfg, f"compat measures written ({len(df)} pairs)")
    return df


# ---------------------------------------------------------------- statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    d = a - b
    N = len(d)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    chunk = 500
    for i in range(0, n_boot, chunk):
        idx = rng.randint(0, N, size=(min(chunk, n_boot - i), N))
        means[i:i + chunk] = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n_boot)))


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def ci95(x) -> Tuple[float, float]:
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean() - h), float(x.mean() + h)


def cfg_label(r) -> str:
    return f"native:{r['dec']}" if r["enc"] is None or (isinstance(r["enc"], float) and math.isnan(r["enc"])) \
        else f"{r['enc']}->{r['dec']} | {r['bridge']} | {r['mode']}"


def _md_cell(v) -> str:
    # Markdown table cells split on "|"; escape any pipe already inside a value
    # (e.g. cfg_label()'s "enc->dec | bridge | mode" strings) or every later
    # column in that row silently shifts left.
    s = f"{v:.3f}" if isinstance(v, float) else str(v)
    return s.replace("|", "\\|")


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(_md_cell(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(v) for v in r.values) + " |")
    return "\n".join(lines)


def export_table(df: pd.DataFrame, name: str, cfg: Cfg):
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.tables_dir / f"{name}.csv", index=False)
    (cfg.tables_dir / f"{name}.md").write_text(df_to_md(df))
    try:
        (cfg.tables_dir / f"{name}.tex").write_text(df.to_latex(index=False, float_format="%.2f"))
    except Exception:
        pass


def _code_parses(code: str, lang: str) -> Optional[bool]:
    """Best-effort syntactic-validity check (a functional-correctness PROXY, not real
    compilation/execution -- see check_syntax_validity.py for why the latter isn't
    well-defined for these single-method snippets). Returns None if no checker is
    available for `lang` (never raises)."""
    if lang == "python":
        import ast, textwrap
        for src in (code, "def _f():\n" + textwrap.indent(code, "    ")):
            try:
                ast.parse(src)
                return True
            except SyntaxError:
                continue
        return False
    if lang == "java":
        if not _ensure("javalang", "javalang", optional=True):
            return None
        import javalang
        try:
            javalang.parse.parse(f"class Dummy {{ {code} }}")
            return True
        except Exception:
            return False
    return None


def phase_analyze(cfg: Cfg):
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_runs(cfg, "main")
    if df.empty:
        log(cfg, "no main-stage results yet"); return
    df["config"] = df.apply(cfg_label, axis=1)

    # ---- syntax validity (functional-correctness proxy, see _code_parses docstring) ----
    sv_by_config: Dict[str, float] = {}
    sv_rows = []
    for c, g in df.groupby("config"):
        n = pv = rv = 0
        usable = True
        for k in g["key"]:
            fp = cfg.preds_dir / f"{k}.jsonl"
            if not fp.exists():
                continue
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pok, rok = _code_parses(row["pred"], cfg.lang), _code_parses(row["ref"], cfg.lang)
                if pok is None:
                    usable = False; break
                n += 1; pv += bool(pok); rv += bool(rok)
            if not usable:
                break
        if usable and n:
            rate = pv / n
            sv_by_config[c] = rate
            sv_rows.append(dict(config=c, n=n, pred_syntax_valid_rate=rate, ref_syntax_valid_rate=rv / n))
    if sv_rows:
        export_table(pd.DataFrame(sv_rows).sort_values("pred_syntax_valid_rate", ascending=False),
                     "syntax_validity", cfg)
        log(cfg, "\n" + df_to_md(pd.DataFrame(sv_rows).sort_values("pred_syntax_valid_rate", ascending=False).round(3)))
    else:
        log(cfg, f"[syntax-validity] no checker available for lang={cfg.lang!r} (or no predictions yet) -- skipped; "
                 f"for java, `pip install javalang` and re-run --phase analyze")

    # ---- per-config summary (seed level) ---------------------------------------
    rows = []
    for c, g in df.groupby("config"):
        lo, hi = ci95(g["bleu"].values)
        rows.append(dict(config=c, n_seeds=len(g), BLEU=g["bleu"].mean(), BLEU_sd=g["bleu"].std(ddof=1) if len(g) > 1 else float("nan"),
                         CI_lo=lo, CI_hi=hi, EM=g["em"].mean(), val_bpc=g["val_bpc"].mean(),
                         empty_rate=g["empty_rate"].mean(), trainable_M=g["trainable_params"].mean() / 1e6,
                         stalled_steps=g["skipped_steps"].sum(),
                         **({"pred_syntax_valid_rate": sv_by_config[c]} if c in sv_by_config else {}),
                         **({"CodeBLEU": g["codebleu"].mean()} if "codebleu" in g and g["codebleu"].notna().any() else {})))
    summ = pd.DataFrame(rows).sort_values("BLEU", ascending=False)
    export_table(summ, "main_results", cfg)
    log(cfg, "\n" + df_to_md(summ.round(3)))

    # ---- per-example vectors (mean over seeds, paired across configs) --------------
    vec = {}
    for c, g in df.groupby("config"):
        arrs = []
        for k in g["key"]:
            p = cfg.per_ex_dir / f"{k}.npz"
            if p.exists():
                arrs.append(np.load(p)["bleu"])
        if arrs:
            vec[c] = np.mean(arrs, axis=0)

    # ---- comparisons -------------------------------------------------------------------
    comps = []
    meta = df.drop_duplicates("config").set_index("config")
    for c in vec:
        r = meta.loc[c]
        if c.startswith("native:"):
            continue
        targets = [f"native:{r['dec']}"]
        if r["bridge"] != "naive":
            targets.append(f"{r['enc']}->{r['dec']} | naive | {r['mode']}")
        for t in targets:
            if t not in vec or len(vec[t]) != len(vec[c]):
                continue
            d, lo, hi, p = paired_bootstrap(vec[c], vec[t])
            sa = df[df.config == c]["bleu"].values; sb = df[df.config == t]["bleu"].values
            w = stats.ttest_ind(sa, sb, equal_var=False).pvalue if len(sa) > 1 and len(sb) > 1 else float("nan")
            comps.append(dict(system=c, baseline=t, delta_sentBLEU=d, CI_lo=lo, CI_hi=hi, p_boot=p,
                              cohens_d_seeds=cohens_d(sa, sb), welch_p_seeds=w))
    if comps:
        cdf = pd.DataFrame(comps)
        cdf["p_holm"] = holm(cdf["p_boot"].tolist())
        cdf["significant_holm_0.05"] = cdf["p_holm"] < 0.05
        export_table(cdf.sort_values("p_holm"), "comparisons", cfg)
        log(cfg, "\n" + df_to_md(cdf.sort_values("p_holm").round(4)))

    # ---- taxonomy validation: compatibility measures vs screening BPC -------------------
    comp_path = cfg.tables_dir / "compat_measures.csv"
    try:
        sc = screen_ranking(cfg)
        if comp_path.exists():
            cm = pd.read_csv(comp_path)
            m = sc.merge(cm, on=["enc", "dec"], suffixes=("", "_c"))
            export_table(m[["enc", "dec", "hetero", "val_bpc", "cka", "vocab_jaccard", "dim_gap"]], "compat_vs_screen", cfg)
            crow = []
            for scope, sub in (("all pairs", m), ("hetero only", m[m.hetero])):
                for meas in ("cka", "vocab_jaccard", "dim_gap"):
                    if len(sub) >= 4:
                        rho, p = stats.spearmanr(sub[meas], sub["val_bpc"])
                        crow.append(dict(scope=scope, measure=meas, n_pairs=len(sub), spearman_rho=rho, p=p))
            if crow:
                export_table(pd.DataFrame(crow), "taxonomy_correlation", cfg)
                log(cfg, "\n" + df_to_md(pd.DataFrame(crow).round(4)))
            fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
            for a, meas, lab in zip(ax, ("cka", "vocab_jaccard", "dim_gap"),
                                    ("linear CKA", "tokenizer Jaccard", "|log2 dim ratio|")):
                a.scatter(m[meas], m["val_bpc"], c=np.where(m.hetero, "tab:blue", "tab:orange"))
                a.set_xlabel(lab); a.set_ylabel("val bits/char (lower = better)")
            fig.tight_layout(); cfg.fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(cfg.fig_dir / "compat_vs_bpc.png", dpi=200); fig.savefig(cfg.fig_dir / "compat_vs_bpc.pdf")
            plt.close(fig)
    except RuntimeError:
        pass

    # ---- bar figure ---------------------------------------------------------------------------
    top = summ.head(14).iloc[::-1]
    err = np.vstack([top["BLEU"] - top["CI_lo"], top["CI_hi"] - top["BLEU"]]).clip(min=0)
    err = np.nan_to_num(err)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(top) + 1.2))
    ax.barh(top["config"], top["BLEU"], xerr=err, color=["tab:gray" if c.startswith("native") else "tab:blue" for c in top["config"]])
    ax.set_xlabel("test BLEU (mean, 95% CI over seeds)")
    fig.tight_layout(); cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.fig_dir / "main_bleu.png", dpi=200); fig.savefig(cfg.fig_dir / "main_bleu.pdf")
    plt.close(fig)

    # ---- efficiency: quality vs. trainable-parameter cost -----------------------------
    e = summ.copy().sort_values("trainable_M")
    e["BLEU_per_M_trainable"] = e["BLEU"] / e["trainable_M"].replace(0, np.nan)
    best_so_far, pareto = -np.inf, []
    for _, r in e.iterrows():                    # Pareto frontier: best BLEU at-or-below each param budget
        is_front = r["BLEU"] > best_so_far
        pareto.append(is_front)
        best_so_far = max(best_so_far, r["BLEU"])
    e["pareto_optimal"] = pareto
    eff_cols = ["config", "BLEU", "trainable_M", "BLEU_per_M_trainable", "pareto_optimal"]
    export_table(e[eff_cols].sort_values("BLEU", ascending=False), "efficiency", cfg)
    log(cfg, "\n" + df_to_md(e[eff_cols].sort_values("BLEU", ascending=False).round(3)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(e["trainable_M"], e["BLEU"], c=["tab:gray" if c.startswith("native") else "tab:blue" for c in e["config"]])
    front = e[e["pareto_optimal"]].sort_values("trainable_M")
    ax.plot(front["trainable_M"], front["BLEU"], "--", color="tab:red", linewidth=1, alpha=0.7)
    for _, r in front.iterrows():
        ax.annotate(r["config"], (r["trainable_M"], r["BLEU"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("trainable parameters (M)"); ax.set_ylabel("test BLEU")
    ax.set_title("quality vs. trainable-parameter cost (dashed = Pareto frontier)")
    fig.tight_layout(); fig.savefig(cfg.fig_dir / "efficiency.png", dpi=200); fig.savefig(cfg.fig_dir / "efficiency.pdf")
    plt.close(fig)

    # ---- qualitative failure analysis: worst-N examples (for manual reading) + ----------
    # ---- automated, heuristic failure categorization (not a substitute for human eval) --
    N_WORST = 15
    fail_dir = Path(cfg.out_dir) / "failure_examples"; fail_dir.mkdir(parents=True, exist_ok=True)
    cat_rows = []
    for c, g in df.groupby("config"):
        if c not in vec:
            continue
        bvec = vec[c]
        fp = cfg.preds_dir / f"{g['key'].iloc[0]}.jsonl"
        if not fp.exists():
            continue
        txt = [json.loads(l) for l in open(fp) if l.strip()]
        if len(txt) != len(bvec):
            continue  # guard: predictions file doesn't line up with the per-example vector
        worst = np.argsort(bvec)[:N_WORST]
        md = [f"# worst {N_WORST} of {len(txt)} test examples by sentence BLEU -- {c}\n"]
        for i in worst:
            md.append(f"### sentBLEU={bvec[i]:.1f}\n**pred:** `{txt[i]['pred'][:300]}`\n\n"
                      f"**ref:**  `{txt[i]['ref'][:300]}`\n")
        n_empty = sum(1 for r in txt if not r["pred"].strip())
        n_near_cap = sum(1 for r in txt if len(r["pred"]) >= cfg.max_tgt * 3)   # heuristic maxlen proxy
        n_len_out = sum(1 for r in txt if r["ref"].split() and
                        abs(len(r["pred"].split()) - len(r["ref"].split())) > 3 * len(r["ref"].split()))
        (fail_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", c) + ".md")).write_text("\n".join(md))
        cat_rows.append(dict(config=c, n=len(txt), empty_rate=n_empty / len(txt),
                             near_maxlen_rate=n_near_cap / len(txt), len_outlier_rate=n_len_out / len(txt)))
    if cat_rows:
        export_table(pd.DataFrame(cat_rows).sort_values("config"), "failure_categories", cfg)
        log(cfg, f"failure examples -> {fail_dir}  (worst {N_WORST}/config; read these yourself -- "
                 f"the categorization columns are crude heuristics, not a replacement for that)")

    log(cfg, f"tables -> {cfg.tables_dir}   figures -> {cfg.fig_dir}")


# =============================================================================
# 10b. SELFTEST  (no downloads: tiny random models + toy tokenizer, ~1 min on CPU)
# =============================================================================
class _ToyTok:
    """Char-level stand-in for a real tokenizer; mimics the bits the pipeline uses."""
    V, LANG = 80, 79
    pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3
    pad_token, eos_token, unk_token = "<pad>", "</s>", "<unk>"

    def __init__(self, style: str):
        self.style = style                                 # gpt2 | seq2seq | plbart | plain

    def _ids(self, s, max_length, truncation, add_special_tokens):
        ids = [4 + (ord(c) % 60) for c in s]
        if max_length and truncation:
            ids = ids[: max(1, max_length - 3)]
        if add_special_tokens:
            if self.style == "seq2seq":
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            elif self.style == "plbart":
                ids = ids + [self.eos_token_id, self.LANG]
        return ids

    def __call__(self, texts=None, max_length=None, truncation=False, padding=False, return_tensors=None,
                 text_target=None, add_special_tokens=True):
        from transformers import BatchEncoding
        if texts is None and text_target is None:
            raise ValueError("_ToyTok.__call__ needs `texts` or `text_target`")
        texts = text_target if text_target is not None else texts
        single = isinstance(texts, str)
        seqs = [self._ids(t, max_length, truncation, add_special_tokens) for t in ([texts] if single else texts)]
        if return_tensors != "pt":
            return {"input_ids": seqs[0] if single else seqs}
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), self.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": am})

    def convert_tokens_to_ids(self, t):
        return self.LANG if t == "__java__" else self.unk_token_id

    def batch_decode(self, seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["".join(chr(97 + (i - 4) % 26) for i in row if i >= 4 and i != self.LANG) for row in seqs]


def _toy_models():
    from transformers import (T5Config, PLBartConfig, GPT2Config, BertConfig, BertModel)
    V, D = _ToyTok.V, 64
    t5 = T5Config(vocab_size=V, d_model=D, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
                  pad_token_id=0, eos_token_id=2, decoder_start_token_id=0)
    pl = PLBartConfig(vocab_size=V, d_model=D, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                      decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128,
                      max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                      decoder_start_token_id=2)
    gp = GPT2Config(vocab_size=V, n_embd=D, n_layer=2, n_head=4, n_positions=256, bos_token_id=1, eos_token_id=2)
    bt = BertConfig(vocab_size=V, hidden_size=D, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=128, max_position_embeddings=256, pad_token_id=0)
    import copy as _copy
    # NOTE: every model gets its OWN deep-copied config. T5EncoderModel flips
    # is_encoder_decoder=False on the config it is given (in place on newer transformers),
    # which silently turned the toy T5 *decoder* into a decoder-only model and made every
    # ->codet5 selftest configuration crash in generate().
    enc = {}
    enc["codet5"] = T5EncoderModel(_copy.deepcopy(t5))
    full = PLBartForConditionalGeneration(_copy.deepcopy(pl)); enc["plbart"] = full.get_encoder(); enc["plbart"].config = full.config
    enc["codebert"] = BertModel(_copy.deepcopy(bt)); enc["unixcoder"] = BertModel(_copy.deepcopy(bt)); enc["roberta"] = BertModel(_copy.deepcopy(bt))
    dec = {"codet5": T5ForConditionalGeneration(_copy.deepcopy(t5)), "plbart": PLBartForConditionalGeneration(_copy.deepcopy(pl)),
           "codegpt": GPT2LMHeadModel(_copy.deepcopy(gp)), "gpt2": GPT2LMHeadModel(_copy.deepcopy(gp))}
    missing_enc = set(ENCODERS) - set(enc); missing_dec = set(DECODERS) - set(dec)
    assert not missing_enc and not missing_dec, (
        f"_toy_models() is missing entries for new pool members: enc={missing_enc} dec={missing_dec} "
        "-- add a toy model for each new name in ENCODERS/DECODERS here.")
    return enc, dec


def phase_selftest(cfg: Cfg) -> bool:
    tc = copy.deepcopy(cfg)
    tc.max_src, tc.max_tgt, tc.batch_size, tc.n_queries, tc.bridge_layers = 32, 24, 4, 8, 1
    tc.num_workers, tc.amp, tc.lang, tc.log_every = 0, "fp32", "java", 1000
    tc.lr, tc.lr_bridge, tc.epochs = 2e-3, 3e-3, 1
    _TOK_CACHE.clear(); _MODEL_CACHE.clear()
    enc_m, dec_m = _toy_models()
    style = {"t5": "seq2seq", "plbart": "plbart", "bert": "plain", "gpt2": "gpt2"}
    for n, sp in ENCODERS.items():
        _TOK_CACHE[(_ckpt(sp, tc), sp["family"], tc.lang)] = _ToyTok(style[sp["family"]])
        _MODEL_CACHE[("enc", n, tc.lang)] = enc_m[n]
    for n, sp in DECODERS.items():
        _TOK_CACHE[(_ckpt(sp, tc), sp["family"], tc.lang)] = _ToyTok(style[sp["family"]])
        _MODEL_CACHE[("dec", n, tc.lang)] = dec_m[n]
    rng = np.random.RandomState(0)
    rows = [{"nl": "".join(rng.choice(list("abcdefgh "), 20)), "code": "".join(rng.choice(list("xyz(){};"), 14))}
            for _ in range(8)]
    fails = []

    def check(name, cond, extra=""):
        if not cond:
            fails.append(name); log(cfg, f"  SELFTEST FAIL: {name} {extra}")

    # left-alignment helper
    e = torch.arange(6, dtype=torch.float).view(1, 3, 2); m = torch.tensor([[1, 1, 0]])
    le, lm = UCFModel._left_align(e, m)
    check("left_align mask", lm.tolist() == [[0, 1, 1]], str(lm.tolist()))
    check("left_align order", torch.equal(le[0, 1:], e[0, :2]) and le.shape == e.shape)

    combos = [(None, d, None, "full") for d in DECODERS]
    for en in ENCODERS:
        for d in DECODERS:
            for br in BRIDGES:
                for mode in (("dec_tune", "full") if br == "naive" else ("bridge_only", "dec_tune")):
                    combos.append((en, d, br, mode))
    for en, d, br, mode in combos:
        name = f"{en or 'native'}->{d} [{br}/{mode}]"
        try:
            spec = RunSpec("selftest", en, d, br, mode, 1)
            model, collate, dec_tok = build_model_and_data(spec, tc)
            model = model.cpu()
            batch = collate(rows[:4])
            model.train()
            loss, n = model.loss_and_count(batch)
            check(name + " finite loss", bool(torch.isfinite(loss)))
            check(name + " token count", int(n) > 0)
            loss.backward()
            groups = model.param_groups()
            check(name + " has trainable params", len(groups) > 0)
            train_params = [p for g in groups for p in g["params"]]
            n_grad = sum(p.grad is not None for p in train_params)
            check(name + " gradients reach params", n_grad > 0)
            if br not in (None, "naive"):
                bp = [p for p in model.bridge.parameters() if p.requires_grad]
                check(name + " bridge gets grads", all(p.grad is not None for p in bp))
            if mode == "bridge_only":
                check(name + " decoder frozen", all(p.grad is None for p in model.dec.parameters()))
            model.eval()
            out = model.generate(batch, 6, 1)
            check(name + " generate shape", out.dim() == 2 and out.size(0) == 4)
            txt = dec_tok.batch_decode(out.tolist())
            check(name + " decode", len(txt) == 4)
        except Exception as ex:  # noqa
            fails.append(name); log(cfg, f"  SELFTEST CRASH {name}: {ex}\n{traceback.format_exc()}")
    log(cfg, f"selftest interface checks done over {len(combos)} configurations")

    # frozen stays frozen / trainable moves after real optimiser steps; loss goes down
    for en, d in (("unixcoder", "codet5"), ("codet5", "plbart"), ("codebert", "codegpt")):
        for mode in ("bridge_only", "dec_tune"):
            name = f"overfit {en}->{d} [resampler/{mode}]"
            try:
                spec = RunSpec("selftest", en, d, "resampler", mode, 1)
                model, collate, _ = build_model_and_data(spec, tc)
                model = model.cpu()
                dec0 = [p.detach().clone() for p in model.dec.parameters() if p.requires_grad or True][:3]
                tr = train_model(model, rows, tc, collate, 1, max_steps=120, tag="toy")
                dec1 = [p.detach() for p in model.dec.parameters()][:3]
                moved = any(not torch.equal(a, b) for a, b in zip(dec0, dec1))
                check(name + " decoder moved iff dec_tune", moved == (mode == "dec_tune"))
                ratio = tr["last_loss"] / tr["first_loss"]
                log(cfg, f"  {name}: loss {tr['first_loss']:.2f} -> {tr['last_loss']:.2f} (ratio {ratio:.2f})")
                # frozen random decoder: the bridge alone can only nudge the loss, so require any decrease
                check(name + " loss decreases", ratio < (0.85 if mode == "dec_tune" else 1.0), f"ratio {ratio:.2f}")
            except Exception as ex:  # noqa
                fails.append(name); log(cfg, f"  SELFTEST CRASH {name}: {ex}\n{traceback.format_exc()}")

    # metrics pipeline
    b, em = per_example_scores(["int x = 1 ;", ""], ["int x = 1 ;", "foo"])
    check("bleu perfect", abs(b[0] - 100) < 1e-6 and b[1] == 0 and em.tolist() == [1.0, 0.0])
    _TOK_CACHE.clear(); _MODEL_CACHE.clear()
    log(cfg, "SELFTEST OVERALL: " + ("PASS" if not fails else f"FAIL ({len(fails)}): {fails[:8]}"))
    return not fails


# =============================================================================
# 11. CLI
# =============================================================================
def parse_args() -> Cfg:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sanity", choices=["selftest", "sanity", "screen", "main", "compat", "analyze", "all"])
    ap.add_argument("--preset", default="medium", choices=list(PRESETS))
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--dataset", default="concode", choices=["concode", "codesearchnet_python", "csv"])
    ap.add_argument("--csv", default="")
    ap.add_argument("--lang", default="java")
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--modes", default="dec_tune", help="comma list from bridge_only,dec_tune,full")
    ap.add_argument("--codebleu", action="store_true")
    ap.add_argument("--lora_baseline", action="store_true",
                     help="also train a native decoder + PEFT LoRA baseline per decoder in `main` "
                          "(external, literature-standard low-param baseline; UNTESTED end-to-end -- "
                          "run --phase sanity or a --preset quick main first)")
    ap.add_argument("--time_budget_h", type=float, default=0.0)
    ap.add_argument("--max_runs", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a, _ = ap.parse_known_args()
    p = PRESETS[a.preset]
    out = a.out_dir or ("/kaggle/working/ucf_bridge_results" if os.path.isdir("/kaggle/working") else "./ucf_bridge_results")
    cfg = Cfg(out_dir=out, dataset=a.dataset, csv_path=a.csv, lang=a.lang, amp=a.amp,
              batch_size=a.batch_size, grad_accum=a.grad_accum, num_beams=a.num_beams,
              num_workers=a.num_workers, main_modes=tuple(a.modes.split(",")), codebleu=a.codebleu,
              lora_baseline=a.lora_baseline,
              time_budget_h=a.time_budget_h, max_runs=a.max_runs, shard=a.shard, **p)
    cfg.phase = a.phase  # type: ignore[attr-defined]
    return cfg


def main():
    cfg = parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    for d in (cfg.runs_dir, cfg.failed_dir, cfg.tables_dir):
        d.mkdir(parents=True, exist_ok=True)
    phase = cfg.phase  # type: ignore[attr-defined]
    save_json(Path(cfg.out_dir) / "config.json", asdict(cfg))
    log(cfg, f"phase={phase} out={cfg.out_dir} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    t0 = time.time()
    if phase == "analyze":
        phase_analyze(cfg); return
    if phase == "selftest":
        sys.exit(0 if phase_selftest(cfg) else 1)
    data = load_data(cfg)
    log(cfg, f"data: train {len(data['train'])} / val {len(data['val'])} / test {len(data['test'])}")
    if phase == "sanity":
        okk = phase_sanity(cfg, data)
        if okk:
            save_json(Path(cfg.out_dir) / "sanity_ok.json", {"time": time.time(), "hp": cfg.hp_hash("sanity")})
        sys.exit(0 if okk else 1)
    if phase in ("screen", "all"):
        if not phase_screen(cfg, data, t0):
            sys.exit(3)                       # stopped early (budget / max_runs): re-run to resume
    if phase in ("compat", "all"):
        phase_compat(cfg, data)
    if phase in ("main", "all"):
        if not phase_main(cfg, data, t0):
            sys.exit(3)
    if phase == "all":
        phase_analyze(cfg)


if __name__ == "__main__":
    main()
