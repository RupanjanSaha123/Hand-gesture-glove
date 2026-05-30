# Zyro ASL Gesture Classification

A gesture recognition project for American Sign Language (ASL) using sensor data from a glove. This repository includes data analysis, classifier training, evaluation, and model export scripts.

## Project Structure

- `asl_glove_dataset.csv` - Main dataset containing flex sensor, accelerometer, gyroscope, hand, and label data.
- `train_asl_model.py` - Train a Random Forest classifier, evaluate performance, plot a confusion matrix, and optionally export a C header for microcontroller use.
- `classify_test.py` - Compare a sample observation against label/hand means from the dataset using Euclidean distance.
- `predict_probs.py` - Train a classifier and print the top 3 predicted label probabilities for a sample observation.
- `check_accuracy.py` - Train a classifier and compute per-label recall values, printing labels below an 85% recall threshold.
- `analysis_asl.py` - Compute mean sensor values per label/hand and rank the closest label-hand combinations for a sample observation.
- `evaluate_noise_A.py` - Test model robustness by adding noise to selected `A` label samples and evaluating noisy predictions.

## Requirements

Recommended Python dependencies:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- Optional: `micromlgen` for exporting the model to C

Install with pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

If you want the Arduino/C export path:

```bash
pip install micromlgen
```

## Usage

Run the training and evaluation script:

```bash
python train_asl_model.py
```

This will:

- load `asl_glove_dataset.csv`
- encode labels and hand data
- train a Random Forest classifier
- print accuracy and a classification report
- show a confusion matrix plot
- export `asl_model.h` if `micromlgen` is installed
- print scaler parameters and feature/label mappings

### Other scripts

- `python classify_test.py`
  - computes closest label-hand means to a sample observation using Euclidean distance.

- `python predict_probs.py`
  - trains a classifier and prints the top 3 label probabilities for a hard-coded observation.

- `python check_accuracy.py`
  - trains a classifier and reports recall by label, highlighting labels under 85% recall.

- `python analyze_confusable_pairs.py`
  - analyze the dataset for label/hand confusion patterns and compute centralized statistics.

- `python evaluate_noise_A.py`
  - evaluates model robustness by adding noise to samples from label `A` and reporting misclassifications.

## Notes

- The dataset features include:
  - flex sensor values: `flex_thumb`, `flex_index`, `flex_middle`, `flex_ring`, `flex_pinky`
  - accelerometer: `accel_x`, `accel_y`, `accel_z`
  - gyroscope: `gyro_x`, `gyro_y`, `gyro_z`
- The model also uses an encoded `hand` feature.

## Extending the Project

- Add new sensor observations or labels to `asl_glove_dataset.csv`.
- Improve model performance with hyperparameter tuning or additional feature engineering.
- Use exported `asl_model.h` for embedded deployment on microcontrollers.
- Add automated tests or a notebook for interactive model exploration.
