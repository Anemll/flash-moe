
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from astc_encoder import (
    ASTCConfig,
    ASTCContext,
    ASTCImage,
    ASTCProfile,
    ASTCQualityPreset,
    ASTCSwizzle,
    ASTCType,
)

# -----------------------------
# ASTC helpers
# -----------------------------

_ASTC_CTX_CACHE = {}

def get_astc_context(
    block_x: int = 6,
    block_y: int = 6,
    quality: int = ASTCQualityPreset.THOROUGH,
    profile: int = ASTCProfile.HDR_RGB_LDR_A,
) -> ASTCContext:
    key = (block_x, block_y, int(quality), int(profile))
    if key not in _ASTC_CTX_CACHE:
        cfg = ASTCConfig(profile, block_x, block_y, quality=quality)
        _ASTC_CTX_CACHE[key] = ASTCContext(cfg)
    return _ASTC_CTX_CACHE[key]


def astc_compress_reconstruct_matrix(
    W: np.ndarray,
    block_x: int = 6,
    block_y: int = 6,
    quality: int = ASTCQualityPreset.THOROUGH,
    min_shift: bool = True,
    profile: int = ASTCProfile.HDR_RGB_LDR_A,
) -> dict:
    """
    Public approximation of Apple's published 6x6 + HDR-ch + per-block-min trick.

    W is interpreted as a 2D scalar field. One weight maps to one texel.
    We store the scalar only in the R channel and use an encoder swizzle of RRR1
    so the compressor sees grayscale HDR RGB + constant alpha.
    """
    W = np.asarray(W, dtype=np.float32)
    h, w = W.shape
    ph = ((h + block_y - 1) // block_y) * block_y
    pw = ((w + block_x - 1) // block_x) * block_x

    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = W

    mins = np.zeros((ph // block_y, pw // block_x), dtype=np.float32)
    if min_shift:
        for by in range(0, ph, block_y):
            for bx in range(0, pw, block_x):
                m = padded[by:by + block_y, bx:bx + block_x].min()
                padded[by:by + block_y, bx:bx + block_x] -= m
                mins[by // block_y, bx // block_x] = np.float16(m).astype(np.float32)

    rgba = np.zeros((ph, pw, 4), dtype=np.float16)
    rgba[..., 0] = padded.astype(np.float16)

    img = ASTCImage(ASTCType.F16, pw, ph, data=rgba.tobytes())
    ctx = get_astc_context(block_x, block_y, quality, profile)
    sw_enc = ASTCSwizzle.from_str("RRR1")
    comp = ctx.compress(img, sw_enc)

    image_dec = ASTCImage(ASTCType.F16, pw, ph)
    ctx.decompress(comp, image_dec, ASTCSwizzle.from_str("RGBA"))
    rec = np.frombuffer(image_dec.data, dtype=np.float16).reshape(ph, pw, 4)[..., 0].astype(np.float32)

    if min_shift:
        for by in range(0, ph, block_y):
            for bx in range(0, pw, block_x):
                rec[by:by + block_y, bx:bx + block_x] += mins[by // block_y, bx // block_x]

    blocks = (ph // block_y) * (pw // block_x)
    astc_bytes = len(comp)
    min_bytes = blocks * 2 if min_shift else 0
    total_bytes = astc_bytes + min_bytes

    return {
        "reconstructed": rec[:h, :w],
        "h": h,
        "w": w,
        "ph": ph,
        "pw": pw,
        "blocks": blocks,
        "astc_bytes": astc_bytes,
        "min_bytes": min_bytes,
        "total_bytes": total_bytes,
        "astc_bpw_nominal": (astc_bytes * 8) / (ph * pw),
        "effective_bpw_on_original": (total_bytes * 8) / (h * w),
    }


def per_block_int4_reconstruct(W: np.ndarray, block_x: int = 6, block_y: int = 6) -> dict:
    W = np.asarray(W, dtype=np.float32)
    h, w = W.shape
    ph = ((h + block_y - 1) // block_y) * block_y
    pw = ((w + block_x - 1) // block_x) * block_x
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = W
    rec = np.zeros_like(padded)
    blocks = 0
    for by in range(0, ph, block_y):
        for bx in range(0, pw, block_x):
            blk = padded[by:by + block_y, bx:bx + block_x]
            mn = blk.min()
            mx = blk.max()
            if mx <= mn + 1e-12:
                rec_blk = np.full_like(blk, mn)
            else:
                scale = (mx - mn) / 15.0
                q = np.round((blk - mn) / scale).clip(0, 15)
                rec_blk = q * scale + mn
            rec[by:by + block_y, bx:bx + block_x] = rec_blk
            blocks += 1
    total_bytes = math.ceil(ph * pw / 2) + blocks * 4  # 4b/weight + fp16 min + fp16 scale
    return {
        "reconstructed": rec[:h, :w],
        "h": h,
        "w": w,
        "ph": ph,
        "pw": pw,
        "blocks": blocks,
        "total_bytes": total_bytes,
        "effective_bpw_on_original": (total_bytes * 8) / (h * w),
    }


def fp16_reconstruct(W: np.ndarray) -> dict:
    W = np.asarray(W, dtype=np.float32)
    rec = W.astype(np.float16).astype(np.float32)
    h, w = W.shape
    return {
        "reconstructed": rec,
        "h": h,
        "w": w,
        "total_bytes": h * w * 2,
        "effective_bpw_on_original": 16.0,
    }


def matrix_metrics(orig: np.ndarray, rec: np.ndarray) -> dict:
    orig = np.asarray(orig, dtype=np.float32)
    rec = np.asarray(rec, dtype=np.float32)
    diff = rec - orig
    denom = np.linalg.norm(orig.ravel()) + 1e-12
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.ravel()) / denom),
        "cosine": float(np.dot(orig.ravel(), rec.ravel()) / ((np.linalg.norm(orig.ravel()) * np.linalg.norm(rec.ravel())) + 1e-12)),
    }


def mlp_logits_from_weights(X_scaled: np.ndarray, coefs: list[np.ndarray], intercepts: list[np.ndarray]) -> np.ndarray:
    h = X_scaled
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        h = h @ W + b
        if i < len(coefs) - 1:
            h = np.maximum(h, 0.0)
    return h


def mlp_predict_from_weights(X_scaled: np.ndarray, coefs: list[np.ndarray], intercepts: list[np.ndarray]):
    logits = mlp_logits_from_weights(X_scaled, coefs, intercepts)
    return logits.argmax(axis=1), logits


def low_rank_plus_astc_reconstruct(
    W: np.ndarray,
    rank: int = 4,
    block_x: int = 6,
    block_y: int = 6,
    quality: int = ASTCQualityPreset.THOROUGH,
) -> dict:
    """
    Simple stand-in for Apple's 'pull out significant singular vectors' idea:
    keep a rank-r SVD side path in fp16, ASTC-compress the residual.
    """
    W = np.asarray(W, dtype=np.float32)
    m, n = W.shape
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(rank, len(S))
    low_rank = (U[:, :r] * S[:r]) @ Vt[:r, :]
    residual = W - low_rank
    comp = astc_compress_reconstruct_matrix(residual, block_x, block_y, quality=quality, min_shift=True)
    rec = low_rank.astype(np.float32) + comp["reconstructed"]
    low_rank_bytes = (m * r + r * n) * 2
    total_bytes = comp["total_bytes"] + low_rank_bytes
    return {
        "reconstructed": rec,
        "rank": r,
        "low_rank_bytes": low_rank_bytes,
        "astc_total_bytes": comp["total_bytes"],
        "total_bytes": total_bytes,
        "effective_bpw_on_original": (total_bytes * 8) / (m * n),
    }


# -----------------------------
# Experiment harness
# -----------------------------

def train_model(random_state: int = 42) -> Pipeline:
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe, (X_train, X_test, y_train, y_test)


def evaluate_model_with_new_coefs(
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    base_logits: np.ndarray,
    new_coefs: list[np.ndarray],
    intercepts: list[np.ndarray],
    base_test_acc: float,
) -> dict:
    pred, logits = mlp_predict_from_weights(X_test_s, new_coefs, intercepts)
    acc = accuracy_score(y_test, pred)
    return {
        "test_acc": float(acc),
        "acc_drop_pp": float((base_test_acc - acc) * 100),
        "logit_rmse": float(np.sqrt(np.mean((logits - base_logits) ** 2))),
        "logit_mae": float(np.mean(np.abs(logits - base_logits))),
    }


def main():
    pipe, data = train_model()
    X_train, X_test, y_train, y_test = data
    scaler = pipe.named_steps["scaler"]
    mlp = pipe.named_steps["mlp"]

    X_test_s = scaler.transform(X_test).astype(np.float32)
    base_coefs = [w.astype(np.float32).copy() for w in mlp.coefs_]
    base_intercepts = [b.astype(np.float32).copy() for b in mlp.intercepts_]

    base_pred, base_logits = mlp_predict_from_weights(X_test_s, base_coefs, base_intercepts)
    base_test_acc = accuracy_score(y_test, base_pred)

    schemes = {}

    def run_scheme(name: str):
        new_coefs = []
        layer_infos = []
        for W in base_coefs:
            if name == "fp16":
                comp = fp16_reconstruct(W)
            elif name == "int4_6x6":
                comp = per_block_int4_reconstruct(W, 6, 6)
            elif name == "astc_4x4_minshift":
                comp = astc_compress_reconstruct_matrix(W, 4, 4, quality=ASTCQualityPreset.THOROUGH, min_shift=True)
            elif name == "astc_6x6_minshift":
                comp = astc_compress_reconstruct_matrix(W, 6, 6, quality=ASTCQualityPreset.THOROUGH, min_shift=True)
            elif name == "astc_8x8_minshift":
                comp = astc_compress_reconstruct_matrix(W, 8, 8, quality=ASTCQualityPreset.THOROUGH, min_shift=True)
            elif name == "astc_6x6_raw":
                comp = astc_compress_reconstruct_matrix(W, 6, 6, quality=ASTCQualityPreset.THOROUGH, min_shift=False)
            else:
                raise ValueError(name)
            new_coefs.append(comp["reconstructed"])
            info = {k: v for k, v in comp.items() if k != "reconstructed"}
            info.update(matrix_metrics(W, comp["reconstructed"]))
            layer_infos.append(info)

        total_bytes = sum(info["total_bytes"] for info in layer_infos)
        total_weights = sum(W.size for W in base_coefs)
        evalm = evaluate_model_with_new_coefs(
            X_test_s, y_test, base_logits, new_coefs, base_intercepts, base_test_acc
        )
        return {
            "scheme": name,
            "layer_infos": layer_infos,
            "total_bytes": total_bytes,
            "total_effective_bpw": (total_bytes * 8) / total_weights,
            **evalm,
        }

    for scheme in ["fp16", "int4_6x6", "astc_4x4_minshift", "astc_6x6_minshift", "astc_8x8_minshift", "astc_6x6_raw"]:
        schemes[scheme] = run_scheme(scheme)

    svd_astc = {}
    for rank in [1, 2, 4, 8, 16]:
        new_coefs = []
        layer_infos = []
        for W in base_coefs:
            comp = low_rank_plus_astc_reconstruct(W, rank=rank, block_x=6, block_y=6, quality=ASTCQualityPreset.THOROUGH)
            new_coefs.append(comp["reconstructed"])
            info = {k: v for k, v in comp.items() if k != "reconstructed"}
            info.update(matrix_metrics(W, comp["reconstructed"]))
            layer_infos.append(info)
        total_bytes = sum(info["total_bytes"] for info in layer_infos)
        total_weights = sum(W.size for W in base_coefs)
        evalm = evaluate_model_with_new_coefs(
            X_test_s, y_test, base_logits, new_coefs, base_intercepts, base_test_acc
        )
        svd_astc[f"rank_{rank}"] = {
            "rank": rank,
            "layer_infos": layer_infos,
            "total_bytes": total_bytes,
            "total_effective_bpw": (total_bytes * 8) / total_weights,
            **evalm,
        }

    out = {
        "base_model": {
            "test_acc": float(base_test_acc),
            "layer_shapes": [list(w.shape) for w in base_coefs],
            "total_weights": int(sum(W.size for W in base_coefs)),
        },
        "schemes": schemes,
        "svd_astc": svd_astc,
    }

    out_path = Path("astc_weight_probe_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
