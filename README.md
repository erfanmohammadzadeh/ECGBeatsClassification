# ECG Beats Classification

A Python workflow for classifying ECG beats with a one-dimensional convolutional neural network (CNN). The primary pipeline prepares beat-annotated WFDB records, trains a four-class AAMI classifier, and can export the trained model to ONNX. Additional scripts support STM32 CSV conversion and inference.

> **Research and educational use:** This project is not a medical device and its predictions are not a diagnosis. The Challenge-record conversion script writes model-generated pseudo-labels, not expert annotations.

## Project structure

```text
ECGBeatsClassification/
├── README.md
├── LICENSE
└── src/
    ├── prepare_data.py       # WFDB records → normalized beat dataset and splits
    ├── train.py              # Train/evaluate the CNN and save model artifacts
    ├── classify_stm32.py     # Classify a STM32 acquisition CSV
    ├── make_stm32.py         # Convert numeric signals into the expected STM32 CSV format
    ├── cnvrtModel.py         # Export Keras model to ONNX and compare predictions
    ├── convertChallenge.py   # Detect and pseudo-label beats in Challenge 2021 records
    ├── benchmark.py          # Experimental acquisition-duration benchmark
    ├── config.py             # STM32 and beat-window settings (legacy utility config)
    └── Testgpu.py            # Minimal TensorFlow/GPU availability check
```

Generated files are not included in the repository. By default, data preparation and training put them under `src/dataset/` and `src/model/` respectively.

## Requirements

Use Python 3.10 or newer and install the libraries used by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn wfdb tensorflow matplotlib pandas onnx tf2onnx
```

TensorFlow installation and GPU support depend on your operating system and hardware. The scripts do not currently ship with a pinned dependency file; record compatible versions in a requirements file when establishing a reproducible environment.

## Data and labels

`prepare_data.py` searches recursively for WFDB records with `.hea` headers and `.atr` beat annotations. It resamples signals to 360 Hz, selects Lead II/MLII and V1 when available (falling back to available leads), extracts 360-sample windows (90 before and 270 after the annotated beat), and normalizes each lead. It groups train, validation, and test splits by source record to avoid putting beats from one record in multiple splits.

The classifier uses these output labels and order:

| Model label | AAMI grouping |
| --- | --- |
| `F` | Fusion beat |
| `N` | Normal beat group |
| `S` | Supraventricular ectopic beat group |
| `V` | Ventricular ectopic beat group |

Provide a local WFDB dataset with `.atr` annotations. By default, the script looks for a `LearnDataset/` directory next to the repository and also checks a configured PhysioNet Challenge 2021 location; override these paths explicitly for your machine. Challenge records without beat annotations are skipped during preparation because record-level diagnoses cannot provide beat-level labels.

## Train the classifier

Run commands from the repository root:

```bash
python src/prepare_data.py --data /path/to/annotated-wfdb --out src/dataset
python src/train.py --data src/dataset --model-dir src/model --epochs 200 --batch-size 128
```

Optional preparation arguments include `--challenge-data`, `--max-beats-per-record`, and `--balance` (`max`, `median`, or `none`). Training writes the Keras model, class order, training log, plots, and test metrics to the model directory. The main artifacts are `ecg_cnn.h5`, `classes.txt`, `classes.npy`, `results.json`, and `training_log.csv`.

## Run inference on STM32 CSV

The CSV format expected by `classify_stm32.py` contains two metadata lines followed by a semicolon-separated table with `index`, `raw`, and `filtered` columns. It expects `model/ecg_cnn.h5` and `dataset/classes.npy` relative to the current working directory. Since training defaults to `src/model` and `src/dataset`, either copy/symlink those output folders to the project root or update the constants near the top of `src/classify_stm32.py` to point to the generated files.

```bash
python src/make_stm32.py --in raw_filtered.csv --out acquisition.csv
python src/classify_stm32.py --csv acquisition.csv
```

`make_stm32.py` can also combine separate raw and filtered files:

```bash
python src/make_stm32.py --raw raw.csv --filtered filtered.csv --out acquisition.csv
```

The classifier assumes a 1000 Hz STM32 sampling rate and ADC baseline near 2000, then resamples to 360 Hz. Check `src/classify_stm32.py` and `src/config.py` if your device configuration differs. The latter central config is not currently imported by every script, so changing it alone may not update all processing paths.

## Export to ONNX

After training, export and numerically check the model:

```bash
python src/cnvrtModel.py --model src/model/ecg_cnn.h5 --out src/model/ecg_cnn.onnx \
  --test-data src/dataset/X_test.npy
```

The converter uses `tf2onnx` (opset 13) and checks agreement with Keras on held-out test windows. Use `--fixed-batch` for a fixed batch-size input when required by a deployment target.

## Challenge 2021 pseudo-annotations

Once the model is trained, `convertChallenge.py` can detect R peaks in Challenge 2021 records and write model-predicted beat annotations:

```bash
python src/convertChallenge.py --data /path/to/challenge/training \
  --model src/model/ecg_cnn.h5 --classes src/model/classes.txt --threshold 0.80
```

It writes `.atr` annotations beside the source records. These are pseudo-labels generated by this classifier; preserve the originals and use `--overwrite` only when intentionally replacing annotations.

## Other utilities

- `src/benchmark.py` is an experimental acquisition-duration benchmark. Its model, class, and CSV paths are currently hard-coded for a particular setup, so edit those constants before use.
- `src/Testgpu.py` imports TensorFlow and is a minimal environment check.
- `src/config.py` documents the device sampling rate, baseline, and beat window; some scripts retain their own constants.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
