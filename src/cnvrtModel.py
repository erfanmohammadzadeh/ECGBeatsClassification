"""
cnvrtModel.py
=============
Convertit le modèle Keras entraîné (model/ecg_cnn.h5) au format ONNX
(model/ecg_cnn.onnx), puis vérifie que le graphe est valide et que ses
prédictions correspondent à celles de Keras.

UTILISATION (depuis la racine du projet) :
    python src/cnvrtModel.py
    python src/cnvrtModel.py --fixed-batch      # entrée (1, 360, 1) au lieu de (?, 360, 1)

SORTIE :
    - model/ecg_cnn.onnx    → modèle ONNX (opset 13)
    - affiche un résumé (opset, entrées/sorties, taille) et un contrôle numérique
      sur les battements de dataset/X_test.npy
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
import tensorflow as tf
from tf2onnx import convert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "ecg_cnn.h5")
ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "ecg_cnn.onnx")
TEST_DATA = os.path.join(PROJECT_ROOT, "dataset", "X_test.npy")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "dataset", "classes.npy")

OPSET = 13          # 13 = large compatibilité (onnxruntime, X-CUBE-AI, tf2onnx)
WINDOW = 360
INPUT_NAME = "ecg_window"
N_CHECK = 256       # nombre de battements comparés Keras ↔ ONNX


def export_onnx(model, onnx_path, fixed_batch=False):
    """Exporte le modèle Keras en ONNX et renvoie (ModelProto, méthode)."""
    in_shape = list(model.input_shape)          # (None, 360, n_leads)
    batch = 1 if fixed_batch else None
    signature = [tf.TensorSpec((batch, in_shape[1], in_shape[2]), tf.float32,
                               name=INPUT_NAME)]

    # TF 2.15 / Keras 2 : chemin standard
    try:
        model_proto, _ = convert.from_keras(
            model, input_signature=signature, opset=OPSET)
        onnx.save(model_proto, onnx_path)
        return model_proto, "tf2onnx.from_keras"
    except Exception as err_keras2:
        # Keras 3 (TF >= 2.16) : export natif
        try:
            model.export(onnx_path, format="onnx")
            return onnx.load(onnx_path), "keras.Model.export"
        except Exception as err_keras3:
            raise SystemExit(
                "Échec de l'export ONNX.\n"
                f"  tf2onnx.from_keras : {err_keras2}\n"
                f"  Model.export      : {err_keras3}")


def save_classes_txt(model, model_path, onnx_path):
    """Write output-label order beside ONNX using training metadata."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    results_path = os.path.join(model_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, encoding="utf-8") as f:
            classes = json.load(f).get("classes")
        source = results_path
    elif os.path.isfile(CLASSES_PATH):
        classes = np.load(CLASSES_PATH, allow_pickle=True).astype(str).tolist()
        source = CLASSES_PATH
    else:
        raise SystemExit("Classes introuvables : attendu results.json à côté du modèle "
                         f"ou {CLASSES_PATH}")

    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise SystemExit(f"Liste 'classes' invalide dans {source}")
    output_count = int(model.output_shape[-1])
    if len(classes) != output_count:
        raise SystemExit(f"Nombre de classes incohérent : {len(classes)} dans {source}, "
                         f"mais le modèle produit {output_count} sorties")

    classes_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "classes.txt")
    with open(classes_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(classes) + "\n")
    print(f"  Classes      : {classes_path} ({len(classes)} depuis {source})")
    return classes_path


def describe(model_proto):
    def shape_of(value_info):
        dims = value_info.type.tensor_type.shape.dim
        return [d.dim_param if d.dim_param else d.dim_value for d in dims]

    opsets = {o.domain or "ai.onnx": o.version for o in model_proto.opset_import}
    print(f"  Opset        : {opsets}")
    print(f"  Entrées      : {[(i.name, shape_of(i)) for i in model_proto.graph.input]}")
    print(f"  Sorties      : {[(o.name, shape_of(o)) for o in model_proto.graph.output]}")
    print(f"  Noeuds       : {len(model_proto.graph.node)}")
    print(f"  Taille       : {os.path.getsize(ONNX_PATH) / 1024:.1f} Ko")


def check_numerics(model, model_proto):
    """Compare les prédictions Keras et ONNX sur de vrais battements."""
    if not os.path.exists(TEST_DATA):
        print(f"\n  (contrôle numérique ignoré : {TEST_DATA} absent)")
        return

    try:
        from onnx.reference import ReferenceEvaluator
    except ImportError:
        print("\n  (contrôle numérique ignoré : onnx.reference indisponible)")
        return

    x = np.load(TEST_DATA, mmap_mode="r")[:N_CHECK].astype(np.float32)
    if x.ndim == 2:                       # dataset 1 dérivation → (N, 360, 1)
        x = x[..., np.newaxis]
    y_k = model.predict(x, verbose=0)

    sess = ReferenceEvaluator(model_proto)
    input_name = model_proto.graph.input[0].name
    output_name = model_proto.graph.output[0].name

    if model_proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1:
        y_o = np.concatenate([sess.run([output_name], {input_name: xi[np.newaxis]})[0]
                              for xi in x])
    else:
        y_o = sess.run([output_name], {input_name: x})[0]

    max_diff = float(np.max(np.abs(y_k - y_o)))
    agreement = float(np.mean(np.argmax(y_k, axis=1) == np.argmax(y_o, axis=1)))
    print(f"\n  Contrôle numérique sur {len(x)} battements de X_test :")
    print(f"    écart max Keras/ONNX : {max_diff:.3e}")
    print(f"    accord des classes   : {agreement:.2%}")

    if agreement < 1.0 or max_diff > 1e-3:
        raise SystemExit("Le modèle ONNX ne reproduit pas fidèlement Keras.")
    print("    [OK] le modèle ONNX est équivalent au modèle Keras")


def main():
    global MODEL_PATH, ONNX_PATH, TEST_DATA

    parser = argparse.ArgumentParser(description="Export Keras → ONNX")
    parser.add_argument("--fixed-batch", action="store_true",
                        help="entrée (1, 360, n) au lieu de (?, 360, n)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"modèle Keras (défaut : {MODEL_PATH})")
    parser.add_argument("--out", default=ONNX_PATH,
                        help=f"fichier ONNX de sortie (défaut : {ONNX_PATH})")
    parser.add_argument("--test-data", default=TEST_DATA,
                        help="X_test.npy utilisé pour le contrôle numérique")
    args = parser.parse_args()

    MODEL_PATH, ONNX_PATH, TEST_DATA = args.model, args.out, args.test_data

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Modèle introuvable : {MODEL_PATH}\n"
                         "Lance d'abord : python src/train.py")

    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    print(f"Chargement du modèle : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  input_shape : {model.input_shape}, sortie : {model.output_shape}")

    print(f"\nExport ONNX -> {ONNX_PATH}")
    model_proto, method = export_onnx(model, ONNX_PATH, args.fixed_batch)
    print(f"  Méthode      : {method}")
    save_classes_txt(model, MODEL_PATH, ONNX_PATH)

    onnx.checker.check_model(model_proto)
    describe(model_proto)
    check_numerics(model, model_proto)

    print(f"\nModèle ONNX prêt : {ONNX_PATH}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
