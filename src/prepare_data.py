"""Build beat-by-beat AAMI data for the Holter model.

Recursively discovers WFDB records with beat annotations. Challenge 2021
records without beat annotations are reported and skipped: their record-level
Dx codes cannot be used as labels for individual beats.
"""
import argparse
import json
import os
import zlib

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEARN = os.path.abspath(os.path.join(HERE, "../LearnDataset"))
DEFAULT_CHALLENGE = os.path.abspath(os.path.join(
    HERE, "../../../../Dataset/physionet.org/files/challenge-2021/1.0.3/training"))
DEFAULT_OUT = os.path.join(HERE, "dataset")
FS_TARGET, BEFORE, AFTER, WINDOW = 360, 90, 270, 360
MAX_BEATS_PER_RECORD = 1200
SEED = 42
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S", "V": "V",
        "E": "V", "F": "F"}
CLASSES = np.array(["F", "N", "S", "V"])

parser = argparse.ArgumentParser(description="Prepare annotated ECG beats for Holter AAMI classification")
parser.add_argument("--data", default=DEFAULT_LEARN, help="annotated WFDB dataset root")
parser.add_argument("--challenge-data", default=DEFAULT_CHALLENGE,
                    help="optional Challenge 2021 root; records without beat annotations are skipped")
parser.add_argument("--out", default=DEFAULT_OUT, help="prepared dataset output directory")
parser.add_argument("--max-beats-per-record", type=int, default=MAX_BEATS_PER_RECORD,
                    help="reproducible per-record cap; use 0 for no cap")
parser.add_argument("--balance", choices=("median", "max", "none"), default="max",
                    help="balance only the training fit split to the largest class (default), median class size, or none")
args = parser.parse_args()


def discover(root):
    """Find complete WFDB records having an .atr annotation file."""
    if not root or not os.path.isdir(root):
        return []
    found = []
    for directory, _, files in os.walk(root):
        for header_name in files:
            if not header_name.lower().endswith(".hea"):
                continue
            base = os.path.splitext(header_name)[0]
            record = os.path.join(directory, base)
            if os.path.isfile(record + ".atr"):
                found.append(record)
    return sorted(set(found))


def choose_leads(signal, sig_names):
    """Select Lead II/MLII and V1 where present; duplicate a mono lead."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    names = [str(n).strip().upper().replace(" ", "") for n in (sig_names or [])]
    def find(candidates):
        return next((i for i, name in enumerate(names) if name in candidates), None)
    lead_ii = find({"II", "MLII", "MCL1"})
    lead_v1 = find({"V1", "ECG V1"})
    if lead_ii is None:
        lead_ii = 0
    if lead_v1 is None:
        lead_v1 = next((i for i in range(signal.shape[1]) if i != lead_ii), lead_ii)
    return signal[:, [lead_ii, lead_v1]]


def normalize_beat(beat):
    if not np.isfinite(beat).all():
        return None
    mean = beat.mean(axis=0, keepdims=True)
    std = beat.std(axis=0, keepdims=True)
    if np.any(std < 1e-6):
        return None
    return ((beat - mean) / (std + 1e-8)).astype(np.float32)


roots = [os.path.abspath(args.data)]
if args.challenge_data and os.path.abspath(args.challenge_data) not in roots:
    roots.append(os.path.abspath(args.challenge_data))
records = []
for root in roots:
    if not os.path.isdir(root):
        print(f"Warning: data root not found, skipping: {root}")
        continue
    discovered = discover(root)
    records.extend((record, root) for record in discovered)
    print(f"Found {len(discovered)} beat-annotated records under {root}")
records.sort(key=lambda item: item[0])
if not records:
    raise SystemExit("No WFDB records with .atr beat annotations found")

X_parts, y_parts, group_parts = [], [], []
used, skipped = [], []
for ix, (record_path, root) in enumerate(records, 1):
    name = os.path.relpath(record_path, root)
    try:
        header = wfdb.rdheader(record_path)
        record = wfdb.rdrecord(record_path, physical=True)
        annot = wfdb.rdann(record_path, "atr")
        signal = choose_leads(record.p_signal, header.sig_name)
        fs = float(header.fs)
        samples = np.asarray(annot.sample, dtype=int)
        if fs != FS_TARGET:
            from math import gcd
            ratio = FS_TARGET / fs
            up, down = int(round(ratio * 10000)), 10000
            common = gcd(up, down)
            up, down = up // common, down // common
            signal = resample_poly(signal, up, down, axis=0).astype(np.float32)
            samples = np.round(samples * (FS_TARGET / fs)).astype(int)
        symbols = np.asarray(annot.symbol)
        valid = [(int(s), AAMI[sym]) for s, sym in zip(samples, symbols)
                 if sym in AAMI and AAMI[sym] in CLASSES
                 and s >= BEFORE and s + AFTER <= len(signal)]
        if args.max_beats_per_record and len(valid) > args.max_beats_per_record:
            rng = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 1_000_000)
            selected = sorted(rng.choice(len(valid), args.max_beats_per_record, replace=False))
            valid = [valid[j] for j in selected]
        beats, labels = [], []
        for peak, label in valid:
            beat = normalize_beat(signal[peak - BEFORE:peak + AFTER])
            if beat is not None:
                beats.append(beat)
                labels.append(label)
        if beats:
            X_parts.append(np.stack(beats))
            y_parts.append(np.asarray(labels))
            group_parts.extend([name] * len(beats))
            used.append({"record": name, "beats": len(beats), "source_fs": fs})
    except Exception as exc:
        skipped.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
    if ix % 100 == 0 or ix == len(records):
        print(f"Processed {ix}/{len(records)} annotated records")

if not X_parts:
    raise SystemExit("Beat annotation files were found, but no usable beats were extracted")
X = np.concatenate(X_parts)
labels = np.concatenate(y_parts)
groups = np.asarray(group_parts)
y = np.searchsorted(CLASSES, labels).astype(np.int64)
if len(np.unique(groups)) < 2:
    raise SystemExit("At least two usable records are required for a record-level train/test split")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
# Make a record-level validation split before any resampling. This keeps all
# beats from each record together and prevents oversampled copies leaking into
# validation or test data.
val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
fit_rel, val_rel = next(val_splitter.split(X[train_idx], y[train_idx], groups[train_idx]))
fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]

fit_counts = np.bincount(y[fit_idx], minlength=len(CLASSES))
present_counts = fit_counts[fit_counts > 0]
if len(present_counts) != len(CLASSES):
    missing = CLASSES[fit_counts == 0].tolist()
    raise SystemExit(f"Training fit split is missing classes {missing}; use more annotated records or change the split seed.")
if args.balance == "median":
    target_count = int(np.median(present_counts))
elif args.balance == "max":
    target_count = int(present_counts.max())
else:
    target_count = None

if target_count is not None:
    rng = np.random.default_rng(SEED)
    sampled_parts = []
    for label in range(len(CLASSES)):
        class_idx = fit_idx[y[fit_idx] == label]
        replace = len(class_idx) < target_count
        chosen = rng.choice(class_idx, size=target_count, replace=replace)
        sampled_parts.append(chosen)
    fit_idx = np.concatenate(sampled_parts)
    rng.shuffle(fit_idx)

os.makedirs(args.out, exist_ok=True)
for name, arr in (("X_train", X[fit_idx]), ("X_val", X[val_idx]), ("X_test", X[test_idx]),
                  ("y_train", y[fit_idx]), ("y_val", y[val_idx]), ("y_test", y[test_idx]),
                  ("classes", CLASSES), ("records_train", groups[train_idx]),
                  ("records_fit", groups[fit_idx]), ("records_val", groups[val_idx]),
                  ("records_test", groups[test_idx])):
    np.save(os.path.join(args.out, name + ".npy"), arr)
summary = {"sources": roots, "records_found_with_atr": len(records),
           "records_used": used, "records_skipped": skipped,
           "beats_total": int(len(y)), "window_samples": WINDOW,
           "target_fs": FS_TARGET, "leads": ["II/MLII", "V1"],
           "classes": CLASSES.tolist(), "train_shape": list(X[train_idx].shape),
           "fit_shape_after_rebalancing": list(X[fit_idx].shape),
           "validation_shape": list(X[val_idx].shape), "balance": args.balance,
           "fit_class_counts_before": {str(CLASSES[i]): int(fit_counts[i]) for i in range(len(CLASSES))},
           "fit_class_counts_after": {str(CLASSES[i]): int((y[fit_idx] == i).sum()) for i in range(len(CLASSES))},
           "test_shape": list(X[test_idx].shape),
           "class_counts": {c: int((labels == c).sum()) for c in CLASSES}}
with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {len(train_idx):,} train beats and {len(test_idx):,} test beats to {args.out}")
print("Challenge 2021 records without .atr beat annotations were automatically skipped.")
