# Cat vs Dog Image Classification: A Comparative Deep Learning Study

An end-to-end image classification project comparing a from-scratch CNN (built up through five controlled regularization/architecture experiments) against transfer learning with MobileNetV2, evaluated with a strict train/validation/test protocol and a final manual error analysis.

**Final result: 99.00% accuracy on a held-out test set that played no role in training, validation, or model selection — the model was fully chosen beforehand, then evaluated on the test set exactly once (Notebook 06), followed only by a post-hoc, non-model-altering error analysis (Notebook 07). Validation→test generalization gap: only 0.08 percentage points.**

---

## Overview

| | |
|---|---|
| **Task** | Binary image classification (cat vs. dog) |
| **Dataset** | 24,998 images (Kaggle "Dogs vs. Cats" / PetImages), cleaned to 24,966 usable images |
| **Approach** | 5 from-scratch CNN variants → best combined model → MobileNetV2 transfer learning (frozen head → fine-tuned) |
| **Final model** | Fine-Tuned MobileNetV2 |
| **Test accuracy** | **99.00%** (F1 0.9900, ROC-AUC 0.9994) |
| **Framework** | TensorFlow / Keras 2.21.0, Python 3.13 |

## Why this project is more than "train a model and report accuracy"

Every stage was designed around one rule: **the test set plays no role in training, validation, or model/architecture selection — the model is fully chosen before the test set is ever evaluated.** All model selection happens on the validation set only. Beyond that:

- Duplicate and corrupted images are detected and handled explicitly: 26 within-class duplicates are automatically excluded from the manifest, and 2 cross-class duplicate groups (6 files total) are flagged for manual review rather than auto-resolved. Nothing is deleted from disk — exclusions are tracked in a manifest, and the original dataset folder is left untouched.
- The train/val/test split is verified programmatically to have **zero file overlap** between sets.
- Data augmentation is implemented as Keras preprocessing *layers* (not `.map()` on the `tf.data` pipeline), so it only ever touches the training pass and cannot leak into validation/test.
- Each from-scratch experiment changes **one variable at a time** relative to the baseline — except the final Combined Regularized CNN, which deliberately stacks the strongest techniques together on purpose.
- The final model is chosen against criteria fixed *before* looking at test results — not selected post hoc to maximize a headline number.
- A dedicated error-analysis pass on the 25 test misclassifications caught a genuine **data quality issue**: two of the test images are not real photographs (a garden statue and a cartoon/clipart image), flagged for manual dataset review.

### Test set usage policy

To be precise about what "the test set" was and wasn't used for, the project follows this strict sequence:

1. **Training** (Notebooks 03–05) — the test set is never seen by any model during training.
2. **Validation / model selection** (Notebooks 03–05) — every architecture choice, hyperparameter, and the final model decision are made using validation metrics only. The test set has no influence here.
3. **Final test evaluation** (Notebook 06) — the already-selected model is scored on the test set exactly once, producing the final, authoritative quantitative numbers (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
4. **Post-hoc error analysis** (Notebook 07) — after that one evaluation, the same 25 misclassified test images are inspected visually to understand *what* the model gets wrong. This step is read-only with respect to the model: it does not retrain, re-tune, re-threshold, or change which model was selected. The 99.00% figure from Notebook 06 remains the final result.

So the test set is evaluated quantitatively exactly once, and inspected qualitatively (post-hoc, without feedback into the model) exactly once after that.

## Pipeline

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Audit & Cleaning | Corruption/duplicate detection, RGB/JPEG normalization, decode verification |
| 02 | Data Pipeline & Preprocessing | Stratified 80/10/10 split with leakage checks, `tf.data` pipeline |
| 03 | Baseline CNN | Unregularized reference model |
| 04 | Augmentation & CNN Experiments | 5 controlled experiments (Augmentation, BatchNorm, Dropout, added depth, combined) |
| 05 | Transfer Learning & Fine-Tuning | MobileNetV2: frozen head, then fine-tuned last 30 layers |
| 06 | Final Evaluation on Test Set | One-time, final test-set scoring of the selected model |
| 07 | Error Analysis | Manual inspection of every test-set misclassification |

## Setup and How to Run

This section is for anyone cloning the repo for the first time.

### Requirements

- **Python 3.13** (the exact interpreter version recorded in the notebooks' own metadata).
- Install the Python packages with:
  ```
  pip install -r requirements.txt
  ```
  `requirements.txt` pins `tensorflow==2.21.0`; `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `Pillow` are listed without a pinned version. These are the only third-party libraries imported anywhere in the notebooks.
- No GPU is required — every notebook in this repository was trained and run on CPU (see [Reproducibility note](#reproducibility-note)).

### Dataset

The raw image dataset is **not included in this repository** (it's tens of thousands of image files — too large and not appropriate for git).

1. Download the dataset from the official source: [Dogs vs. Cats — Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).
2. Extract it so you end up with a folder named exactly **`PetImages/`**, containing two subfolders: **`PetImages/Cat/`** and **`PetImages/Dog/`**.
3. Place `PetImages/` in the same directory as the notebooks (the project root) — Notebook 01 expects it at that relative path.
4. Run **Notebook 01**. It reads every file under `PetImages/`, checks it for corruption and duplicates, normalizes it to RGB JPEG, and writes the results to a **new** folder (`PetImages_RGB/`) and manifest files (see "Files generated" below). It never modifies or deletes anything inside the original `PetImages/` folder.

### Notebook run order

The notebooks are numbered and **must be run in order**, top to bottom, because each one depends on files produced by the one(s) before it (the manifest from Notebook 01, the saved models from Notebooks 03–05, etc.). See the [Pipeline](#pipeline) table above for what each notebook does. In short: `01 → 02 → 03 → 04 → 05 → 06 → 07`.

### Files generated while running

Running the notebooks in order produces these files/folders (none of them are included in this repository — see the next section):

| File / folder | Produced by | Contents |
|---|---|---|
| `PetImages_RGB/` | Notebook 01 | Cleaned, normalized (RGB JPEG) copy of every usable image |
| `normalization_report.csv` | Notebook 01 | Per-file diagnostic log of the normalization step |
| `clean_manifest.csv` | Notebook 01 (created, then updated) | The official list of usable images + labels that every later notebook reads |
| `baseline_cnn.keras` | Notebook 03 | Saved baseline CNN model |
| `augmentation_cnn.keras`, `batchnorm_cnn.keras`, `dropout_cnn.keras`, `deeper_cnn.keras`, `combined_cnn.keras` | Notebook 04 | Saved models for each of the 5 controlled experiments |
| `frozen_mobilenetv2.keras`, `finetuned_mobilenetv2.keras` | Notebook 05 | Saved MobileNetV2 models (frozen-head stage and fine-tuned stage) |
| `final_test_results.csv` | Notebook 06 | The one-time final test-set metrics |

### Re-running the project

If you clone this repository, the dataset and every file in the table above will be **missing** — that's expected, not a bug. This project is not a "run one script and it's all set up" pipeline: it needs the dataset downloaded first, and it needs Notebooks 01–07 run in order to regenerate the manifest and model files from scratch. Budget real time and CPU for this (see the [Reproducibility note](#reproducibility-note) below on training time and non-determinism).

## Dataset

- Source: 24,998 labeled images (12,499 Cat / 12,499 Dog)
- After cleaning: **24,966 usable images** (12,480 Cat / 12,486 Dog) — 32 files excluded (0 corrupted, 26 within-class duplicates, 6 files from cross-class duplicate groups)
- Split: stratified 80% train / 10% validation / 10% test (seed 42) → 19,972 / 2,497 / 2,497 images, with 1,248 Cat / 1,249 Dog in the test set

### Dataset Source

This project uses the [Dogs vs. Cats — Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data) dataset. Per Kaggle's official description, the training set consists of 25,000 labeled cat and dog images; the raw folder used as the starting point for this project (see Notebook 01) contained 24,998 images (12,499 Cat / 12,499 Dog) — 2 fewer than Kaggle's stated count, a gap that predates this project's own cleaning process and is unrelated to the 32 files excluded during it. All cleaning, deduplication, normalization, and splitting described above (Notebooks 01 and 02) were performed internally as part of this project, which is why the final dataset counts differ from Kaggle's original figures. No licensing terms are stated here beyond what Kaggle's page itself specifies.

## Results

| Model | Val Accuracy | Val Loss | Val F1 | Params |
|---|---|---|---|---|
| Baseline CNN | 82.30%¹ | 0.3936 | — (not computed) | 6.65M |
| + Data Augmentation | 87.91% | 0.2673 | 88.44% | 6.65M |
| + Batch Normalization | 84.42% | 0.4284 | 79.78% | 6.65M |
| + Dropout | 83.54% | 0.3762 | 84.09% | 6.65M |
| + Slightly Deeper CNN | 89.03% | 0.2994 | 87.56% | 3.04M |
| **Combined Regularized CNN** (best from-scratch) | **91.55%** | 0.2018 | 91.44% | 3.04M |
| MobileNetV2 — frozen head | 98.92% | 0.0419 | 98.92% | 2.26M (1,281 trainable) |
| **MobileNetV2 — fine-tuned** (selected model) | **99.08%** | 0.0323 | 98.84% | 2.26M (1.51M trainable) |

¹ The saved baseline model (restored to its best-validation-*loss* checkpoint, epoch 3) scored 81.70% validation accuracy. A different epoch (epoch 6) reached a higher 82.30% validation accuracy, but that checkpoint was never the one actually saved or carried forward — it's an independently-tracked peak, not the deployed model's score. 82.30% is the figure used as the baseline reference throughout the rest of this project (Notebooks 04–05), so it's worth knowing both numbers exist.

*Note: for every row, "Val Accuracy" and "Val Loss/F1" are tracked independently and can come from different epochs — the epoch with the lowest loss isn't always the epoch with the highest accuracy. Precision/Recall/F1 always correspond to the best-validation-loss checkpoint (the one actually restored and saved).*

### Final test-set evaluation (Fine-Tuned MobileNetV2, one-time run)

| Metric | Score |
|---|---|
| Accuracy | 99.00% |
| Precision | 98.65% |
| Recall | 99.36% |
| F1 | 99.00% |
| ROC-AUC | 0.9994 |
| Validation → Test gap | −0.08 pp |

**Confusion matrix** (2,497 test images, 25 total errors — 1.00% error rate):

| | Predicted Cat | Predicted Dog |
|---|---|---|
| **Actual Cat** | 1,231 | 17 |
| **Actual Dog** | 8 | 1,241 |

### Error analysis highlights

Based on a single manual visual pass over all 25 misclassified images (not a rigorous coded analysis — categories are not mutually exclusive and a second look is warranted before treating this as a firm conclusion), observed patterns include: image-quality issues such as blur or low light (6 cases), occlusion by cage bars, wire mesh, or a person's hands (5 cases), multiple animals appearing in the same frame (4 cases), and visual similarity in coat pattern or body proportions between the classes (3 cases). Two errors were not genuine photographs at all — a stone garden statue of a dog and a cartoon/clipart illustration — flagged as a data-quality issue worth a manual review of the source dataset, though it's a single image out of 24,966 and doesn't change the project's conclusions.

## Key engineering decisions

- **Independent best-epoch tracking**: best validation accuracy and best validation loss are tracked and reported separately per run, since they don't always occur at the same epoch.
- **Model selection by fixed criteria**: accuracy, F1, loss curves, parameter efficiency, and training cost were all compared before a model was chosen — the frozen-head model actually edged out the fine-tuned one on precision/F1, but fine-tuning was selected on the criteria set in advance.
- **No silent data mutation**: cleaning never auto-deletes or auto-relabels; ambiguous cases (e.g. cross-class duplicates) are surfaced for manual review, not resolved automatically.

## Tech stack

Python 3.13 · TensorFlow / Keras 2.21.0 · scikit-learn · pandas · NumPy · Matplotlib · Pillow

## Repository structure

```
├── 01_Data_Audit_and_Cleaning.ipynb
├── 02_Data_Pipeline_and_Preprocessing.ipynb
├── 03_Baseline_CNN.ipynb
├── 04_Data_Augmentation_and_CNN_Experiments.ipynb
├── 05_Transfer_Learning_and_Fine_Tuning.ipynb
├── 06_Final_Evaluation_on_Test_Set.ipynb
├── 07_Error_Analysis.ipynb
├── requirements.txt
└── README.md
```

## Files Not Included in This Repository

The structure above is everything actually tracked in this repository. The following are **not** included, because they're either the raw dataset or large/binary files generated by running the notebooks — all of them can be regenerated by following [Setup and How to Run](#setup-and-how-to-run):

- **Original dataset** (`PetImages/`) — the raw Kaggle images; download separately (see [Dataset](#dataset) above).
- **Cleaned/normalized dataset** (`PetImages_RGB/`) — generated by Notebook 01.
- **Manifest and report files** (`clean_manifest.csv`, `normalization_report.csv`) — generated by Notebook 01.
- **Trained model artifacts** (`baseline_cnn.keras`, `augmentation_cnn.keras`, `batchnorm_cnn.keras`, `dropout_cnn.keras`, `deeper_cnn.keras`, `combined_cnn.keras`, `frozen_mobilenetv2.keras`, `finetuned_mobilenetv2.keras`) — generated by Notebooks 03, 04, and 05 respectively.
- **Final test results file** (`final_test_results.csv`) — generated by Notebook 06.

None of these are hidden or missing by mistake — they're excluded intentionally because of their size, and every one of them is fully reproducible by running the notebooks in order.

## Reproducibility note

Results were produced with TensorFlow 2.21.0 and Python 3.13, trained on **CPU** (no GPU was available in this environment), with a fixed random seed (42) applied to Python's `random`, NumPy, and TensorFlow. A fixed seed does not by itself guarantee bit-for-bit identical results on every re-run — exact determinism would additionally require `tf.config.experimental.enable_op_determinism()`, which was not used here — so re-running the notebooks may produce slightly different numbers. The figures in this README are from the single, final, saved execution in this repository and are the authoritative reference for this project.

## AI Usage & Transparency

AI tools were used at several points throughout this project — always in a supporting or reviewing role, never as the one building the project, training the models, or making final decisions.

- **Data cleaning and initial analysis.** In the early stages, AI tools were used to help think through the image auditing and cleaning approach (duplicate detection, corruption checks). Every resulting decision was understood and reviewed before being applied — nothing was accepted from AI output blindly.
- **Learning and discussion during experimentation.** While building and running the experiments, AI was used as a learning aid: to understand concepts, interpret results, and discuss the reasoning behind methodological choices. The goal was understanding the work, not copying solutions.
- **Coding assistance during model building.** While building the models, pipelines, and evaluation code, AI helped write and review parts of the code and check its logic and methodology. Understanding the decisions, reviewing the results, and choosing what to keep or discard remained the author's own work throughout.
- **Post-completion audit.** After the notebooks were completed, AI was used as a reviewer/auditor to check for methodological issues, verify numerical consistency across notebooks, and cross-check this README against the actual executed results. Any finding or suggestion raised during this review was independently verified before being accepted or applied.
- **Writing and formatting assistance.** AI also helped draft and organize the Markdown/README text for clarity and professionalism. The results, numbers, and decisions described throughout this document reflect the project's actual execution, not AI-generated content.

In short: the design, experiments, model training, and final decisions in this project were carried out and owned by the author. AI was used as a supporting tool at various stages — not as a co-author, and not as a substitute for understanding the work.

## Author

Built as an independent deep learning project to practice a full, leakage-free ML workflow: data auditing, controlled experimentation, transfer learning, and honest, one-time evaluation.
