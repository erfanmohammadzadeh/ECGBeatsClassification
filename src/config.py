"""
config.py
=========
Configuration centrale — à modifier quand tu changes le sample time dans CubeIDE.

Règle :  WINDOW_MITBIH = BEFORE_PEAK + AFTER_PEAK
"""

# ─── STM32 acquisition ────────────────────────────────────────────────────────
STM32_FS     = 1000   # Fréquence d'échantillonnage STM32 (Hz) ← changer ici si tu modifies le timer CubeIDE
ADC_BASELINE = 2000   # Valeur de repos de l'ADC

# ─── MIT-BIH / Modèle ─────────────────────────────────────────────────────────
MITBIH_FS    = 360    # Fréquence MIT-BIH — NE PAS CHANGER (base d'entraînement)

# ─── Fenêtre de battement ─────────────────────────────────────────────────────
# Ces 3 valeurs doivent rester cohérentes : WINDOW_MITBIH = BEFORE_PEAK + AFTER_PEAK
BEFORE_PEAK   = 90    # samples avant le pic R  (à 360Hz MIT-BIH)
AFTER_PEAK    = 270   # samples après le pic R  (à 360Hz MIT-BIH)
WINDOW_MITBIH = BEFORE_PEAK + AFTER_PEAK   # = 360 — calculé automatiquement

# ─── Inférence ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.60  # Seuil de confiance minimum pour retenir une fenêtre

# ─── Chemins ──────────────────────────────────────────────────────────────────
MODEL_PATH   = 'model/ecg_cnn.h5'
CLASSES_PATH = 'dataset/classes.npy'
