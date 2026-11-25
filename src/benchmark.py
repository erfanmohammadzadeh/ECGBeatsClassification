"""
benchmark.py
============
Compare les performances de classification ECG en fonction
du temps d'acquisition (TOTAL_SAMPLES).

UTILISATION :
    1. Faire 4 acquisitions avec ces valeurs dans le main STM32 :
       - TOTAL_SAMPLES = 600u  → ~15s
       - TOTAL_SAMPLES = 1200u → ~30s
       - TOTAL_SAMPLES = 1600u → ~40s
       - TOTAL_SAMPLES = 2000u → ~50s
    2. Nommer les CSV : ECG_15s.CSV, ECG_30s.CSV, ECG_40s.CSV, ECG_50s.CSV
    3. Lancer : python benchmark.py

SORTIE :
    - benchmark_results.png → courbe performances vs durée
    - benchmark_results.csv → tableau des résultats
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "C:/ENSISA/S8/2A ASE/Projet 2A Electrocardio/IA/model/ecg_cnn.h5"
CLASSES_PATH = "C:/ENSISA/S8/2A ASE/Projet 2A Electrocardio/IA/dataset/classes.npy"
STM32_FS = 1000
MITBIH_FS = 360
WINDOW_MITBIH = 36
ADC_BASELINE = 2000
CONF_THRESHOLD = 0.60

# Fichiers CSV à tester → (durée réelle en secondes, chemin CSV)
CSV_FILES = [
    (15, "src/ECG_15s.CSV"),
    (30, "src/ECG_30s.CSV"),
    (40, "src/ECG_40s.CSV"),
    (50, "src/ECG_50s.CSV"),
]


# ─── CHARGEMENT CSV ───────────────────────────────────────────────────────────
def load_csv(csv_path):
    df = pd.read_csv(csv_path, sep=";", skiprows=2, header=0)
    return df["filtered"].values.astype(np.float32)


# ─── PRÉTRAITEMENT ────────────────────────────────────────────────────────────
def preprocess(signal):
    signal = signal - ADC_BASELINE
    n = int(len(signal) * MITBIH_FS / STM32_FS)
    signal_360 = resample(signal, n)
    step = WINDOW_MITBIH // 2
    windows = []
    for i in range(0, len(signal_360) - WINDOW_MITBIH, step):
        w = signal_360[i : i + WINDOW_MITBIH].astype(np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        windows.append(w)
    return np.array(windows)[..., np.newaxis] if windows else np.array([])


# ─── CLASSIFICATION D'UN FICHIER ──────────────────────────────────────────────
def classify_file(csv_path, model, classes):
    signal = load_csv(csv_path)
    X = preprocess(signal)

    if len(X) == 0:
        return None

    preds = model.predict(X, verbose=0)
    labels = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)

    # Filtre par seuil de confiance
    mask = confs >= CONF_THRESHOLD
    labels_f = labels[mask]
    confs_f = confs[mask]

    if len(labels_f) == 0:
        return None

    counts = Counter(labels_f)
    total = len(labels_f)

    dominant_idx = max(counts, key=counts.get)
    dominant_cls = classes[dominant_idx]
    pct_normal = counts.get(list(classes).index("N"), 0) / total * 100
    mean_conf = confs_f.mean() * 100
    n_windows = len(X)
    n_retained = int(mask.sum())

    return {
        "n_windows_total": n_windows,
        "n_windows_retained": n_retained,
        "pct_retained": mask.mean() * 100,
        "dominant_cls": dominant_cls,
        "pct_normal": pct_normal,
        "mean_conf": mean_conf,
        "counts": counts,
        "total": total,
        "n_samples": len(signal),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

results = []
durations = []

print("\nClassification des fichiers...")
for dur, csv_path in CSV_FILES:
    if not os.path.exists(csv_path):
        print(f"  ⚠ Manquant : {csv_path} — ignoré")
        continue

    print(f"  {csv_path} ({dur}s)...", end=" ")
    r = classify_file(csv_path, model, classes)

    if r is None:
        print("Pas assez de données, ignoré")
        continue

    r["duration"] = dur
    r["csv"] = csv_path
    results.append(r)
    durations.append(dur)

    print(
        f"OK — {r['n_windows_retained']} fenêtres retenues, "
        f"confiance={r['mean_conf']:.1f}%, "
        f"N={r['pct_normal']:.1f}%"
    )

if len(results) < 2:
    print("\nERREUR : il faut au moins 2 fichiers CSV pour tracer la courbe.")
    exit(1)

# ─── TABLEAU RÉSULTATS ────────────────────────────────────────────────────────
df_res = pd.DataFrame(
    [
        {
            "Durée (s)": r["duration"],
            "Fichier": r["csv"],
            "Samples": r["n_samples"],
            "Fenêtres totales": r["n_windows_total"],
            "Fenêtres retenues": r["n_windows_retained"],
            "% retenues": f"{r['pct_retained']:.1f}%",
            "% Normal": f"{r['pct_normal']:.1f}%",
            "Confiance moy. (%)": f"{r['mean_conf']:.1f}%",
            "Diagnostic": r["dominant_cls"],
        }
        for r in results
    ]
)

print("\n" + "=" * 70)
print(df_res.to_string(index=False))
print("=" * 70)
df_res.to_csv("benchmark_results.csv", index=False, sep=";")
print("\nTableau sauvegardé : benchmark_results.csv")

# ─── GRAPHIQUES ───────────────────────────────────────────────────────────────
durs = [r["duration"] for r in results]
confs = [r["mean_conf"] for r in results]
normals = [r["pct_normal"] for r in results]
wins = [r["n_windows_retained"] for r in results]

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Performances de l'IA ECG en fonction du temps d'acquisition",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Graphique 1 : Confiance moyenne ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(
    durs,
    confs,
    "o-",
    color="#1D9E75",
    linewidth=2.5,
    markersize=10,
    markerfacecolor="white",
    markeredgewidth=2.5,
)
for x, y in zip(durs, confs):
    ax1.annotate(
        f"{y:.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="500",
    )
ax1.set_xlabel("Durée d'acquisition (s)", fontsize=11)
ax1.set_ylabel("Confiance moyenne (%)", fontsize=11)
ax1.set_title("Confiance moyenne de l'IA", fontsize=12, fontweight="500")
ax1.set_ylim(max(0, min(confs) - 10), 105)
ax1.set_xticks(durs)
ax1.grid(True, alpha=0.3)
ax1.fill_between(durs, confs, alpha=0.08, color="#1D9E75")

# ── Graphique 2 : % Normal ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    durs,
    normals,
    "o-",
    color="#378ADD",
    linewidth=2.5,
    markersize=10,
    markerfacecolor="white",
    markeredgewidth=2.5,
)
for x, y in zip(durs, normals):
    ax2.annotate(
        f"{y:.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="500",
    )
ax2.set_xlabel("Durée d'acquisition (s)", fontsize=11)
ax2.set_ylabel("% fenêtres classées Normal (%)", fontsize=11)
ax2.set_title("Stabilité du diagnostic Normal", fontsize=12, fontweight="500")
ax2.set_ylim(0, 110)
ax2.set_xticks(durs)
ax2.grid(True, alpha=0.3)
ax2.fill_between(durs, normals, alpha=0.08, color="#378ADD")

# ── Graphique 3 : Nombre de fenêtres retenues ──
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(
    durs, wins, color="#534AB7", alpha=0.75, width=6, edgecolor="#3C3489", linewidth=0.8
)
for bar, val in zip(bars, wins):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        str(val),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="500",
    )
ax3.set_xlabel("Durée d'acquisition (s)", fontsize=11)
ax3.set_ylabel("Nombre de fenêtres retenues", fontsize=11)
ax3.set_title("Fenêtres analysées (confiance ≥ 60%)", fontsize=12, fontweight="500")
ax3.set_xticks(durs)
ax3.grid(True, alpha=0.3, axis="y")

# ── Graphique 4 : Score composite ──
ax4 = fig.add_subplot(gs[1, 1])
# Score composite = moyenne normalisée confiance + % Normal
conf_norm = [(c - min(confs)) / (max(confs) - min(confs) + 1e-8) * 100 for c in confs]
norm_norm = [
    (n - min(normals)) / (max(normals) - min(normals) + 1e-8) * 100 for n in normals
]
composite = [(c + n) / 2 for c, n in zip(conf_norm, norm_norm)]

ax4.plot(
    durs,
    composite,
    "o-",
    color="#E24B4A",
    linewidth=2.5,
    markersize=10,
    markerfacecolor="white",
    markeredgewidth=2.5,
)
for x, y in zip(durs, composite):
    ax4.annotate(
        f"{y:.0f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="500",
    )
ax4.set_xlabel("Durée d'acquisition (s)", fontsize=11)
ax4.set_ylabel("Score composite (0–100)", fontsize=11)
ax4.set_title("Score global de fiabilité", fontsize=12, fontweight="500")
ax4.set_ylim(0, 115)
ax4.set_xticks(durs)
ax4.grid(True, alpha=0.3)
ax4.fill_between(durs, composite, alpha=0.08, color="#E24B4A")

plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
print("Graphique sauvegardé : benchmark_results.png")
plt.show()

print("\n✅ Benchmark terminé !")
