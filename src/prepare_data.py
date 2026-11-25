"""Prepare Challenge 2021 ECG records for multilabel classification.

Records are discovered recursively from the PhysioNet Challenge training
directory. Each .hea/.mat pair becomes one example; ``# Dx`` header codes are
the multilabel targets. Signals are resampled to a fixed length for batching.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

DEFAULT_DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../project/Dataset/physionet.org/files/challenge-2021/1.0.3/training"
))
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "dataset")
TARGET_LENGTH = 5000
TARGET_LEADS = 12
SEED = 42

parser = argparse.ArgumentParser(description="Prepare Challenge 2021 ECG dataset")
parser.add_argument("--data", default=DEFAULT_DATA, help="Challenge 2021 training directory")
parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output directory")
parser.add_argument("--length", type=int, default=TARGET_LENGTH,
                    help="samples per record after resampling (default: 5000)")
parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="parallel record readers (default: up to 4)")
args = parser.parse_args()
if args.workers < 1:
    parser.error("--workers must be at least 1")


def discover_records(root):
    """Return sorted record paths whose header and referenced signal exist."""
    result = []
    for directory, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".hea"):
                continue
            path = os.path.join(directory, os.path.splitext(name)[0])
            try:
                header = wfdb.rdheader(path)
            except Exception:
                continue
            if header.n_sig and all(os.path.isfile(os.path.join(directory, f))
                                    for f in header.file_name):
                result.append(path)
    return sorted(result)


def diagnosis_codes(comments):
    codes = []
    for line in comments or []:
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "dx":
            codes.extend(x.strip() for x in value.split(",") if x.strip())
    return sorted(set(codes))


def resize_signal(signal, length):
    """Linearly resample time and normalize to a consistent 12-lead input."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, copy=False)
    old_x = np.linspace(0.0, 1.0, signal.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    resized = np.stack([np.interp(new_x, old_x, signal[:, lead])
                        for lead in range(signal.shape[1])], axis=-1)
    if resized.shape[1] < TARGET_LEADS:
        resized = np.pad(resized, ((0, 0), (0, TARGET_LEADS - resized.shape[1])))
    else:
        resized = resized[:, :TARGET_LEADS]
    mean = resized.mean(axis=0, keepdims=True)
    std = resized.std(axis=0, keepdims=True)
    return ((resized - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def prepare_record(index_path):
    """Read and resize one ECG; return a compact failure reason on errors."""
    index, path = index_path
    try:
        header = wfdb.rdheader(path)
        codes = diagnosis_codes(header.comments)
        if not codes:
            return index, None, None, "no Dx labels"
        signal = wfdb.rdrecord(path, physical=True).p_signal
        if signal is None or len(signal) < 2:
            raise ValueError("empty signal")
        return index, codes, resize_signal(signal, args.length), None
    except Exception as exc:
        return index, None, None, f"{type(exc).__name__}: {exc}"


if not os.path.isdir(args.data):
    raise SystemExit(f"Dataset directory not found: {args.data}")
record_paths = discover_records(args.data)
if not record_paths:
    raise SystemExit(f"No readable .hea/.mat records found under {args.data}")
print(f"Discovered {len(record_paths)} records under {args.data}")

os.makedirs(args.out, exist_ok=True)
# Keep the large signal matrix on disk. Building a Python list and then calling
# np.stack duplicates the full dataset in RAM, which can take down VS Code.
signals_path = os.path.join(args.out, ".signals.tmp.npy")
signals = None
successful = {}
skipped = []
signals = np.lib.format.open_memmap(
    signals_path, mode="w+", dtype=np.float32,
    shape=(len(record_paths), args.length, TARGET_LEADS))

# Keep only a small number of futures in flight. Each completed result contains
# one resized ECG, so unbounded submission could otherwise consume substantial RAM.
items = iter(enumerate(record_paths))
completed_count = 0
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    pending = set()
    for _ in range(min(len(record_paths), args.workers * 2)):
        index_path = next(items, None)
        if index_path is not None:
            pending.add(executor.submit(prepare_record, index_path))
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            index, codes, resized, reason = future.result()
            path = record_paths[index]
            if reason is not None:
                skipped.append({"record": os.path.relpath(path, args.data), "reason": reason})
            else:
                signals[index] = resized
                successful[index] = codes
            completed_count += 1
            if completed_count % 1000 == 0 or completed_count == len(record_paths):
                print(f"  Processed {completed_count}/{len(record_paths)}")
        for _ in range(len(done)):
            index_path = next(items, None)
            if index_path is not None:
                pending.add(executor.submit(prepare_record, index_path))

labels = [successful[i] for i in sorted(successful)]
record_names = [os.path.relpath(record_paths[i], args.data) for i in sorted(successful)]
successful_indices = np.asarray(sorted(successful), dtype=np.int64)

if len(labels) < 2:
    raise SystemExit("Fewer than two labeled records could be read")
classes = np.asarray(sorted({code for row in labels for code in row}))
class_index = {code: i for i, code in enumerate(classes)}
y = np.zeros((len(labels), len(classes)), dtype=np.float32)
for row, codes in enumerate(labels):
    y[row, [class_index[c] for c in codes]] = 1.0

# Split records once. Stratification is unsuitable for multilabel targets;
# seeded shuffle gives a reproducible record-level split without leakage.
indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, shuffle=True)
signals.flush()
del signals
signals = np.load(signals_path, mmap_mode="r")

def save_signal_split(filename, split_indices):
    destination = os.path.join(args.out, filename)
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32,
        shape=(len(split_indices), args.length, TARGET_LEADS))
    batch_size = 64
    for start in range(0, len(split_indices), batch_size):
        batch_indices = split_indices[start:start + batch_size]
        source_indices = successful_indices[batch_indices]
        output[start:start + len(batch_indices)] = signals[source_indices]
    output.flush()
    del output

save_signal_split("X_train.npy", train_idx)
save_signal_split("X_test.npy", test_idx)
np.save(os.path.join(args.out, "y_train.npy"), y[train_idx])
np.save(os.path.join(args.out, "y_test.npy"), y[test_idx])
np.save(os.path.join(args.out, "classes.npy"), classes)
np.save(os.path.join(args.out, "records_train.npy"), np.asarray(record_names)[train_idx])
np.save(os.path.join(args.out, "records_test.npy"), np.asarray(record_names)[test_idx])
del signals
os.remove(signals_path)
meta = {"source_dir": os.path.abspath(args.data), "records_discovered": len(record_paths),
        "records_used": len(labels), "records_skipped": skipped, "length": args.length,
        "leads": TARGET_LEADS, "classes": classes.tolist(),
        "train_shape": [len(train_idx), args.length, TARGET_LEADS],
        "test_shape": [len(test_idx), args.length, TARGET_LEADS],
        "labels_per_class": {c: int(y[:, j].sum()) for j, c in enumerate(classes)}}
with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"Saved {len(train_idx)} train and {len(test_idx)} test records to {args.out}")
print(f"Classes: {len(classes)} | leads: {TARGET_LEADS} | shape: ({args.length}, {TARGET_LEADS})")
