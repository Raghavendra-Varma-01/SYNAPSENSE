# SYNAPSENSE research results (sanitized summary)

This note preserves high-level dataset and evaluation findings without local file paths, clip identifiers, archive names, or external Drive links. Detailed evaluation artifacts that do not contain machine-specific paths are listed in [`github-file-manifest.md`](github-file-manifest.md) and remain available under `models/`.

## DAiSEE data preparation

The local recovery report records 5,357 usable labelled training pairs, 1,429 validation pairs, and 1,784 test pairs (8,570 total). The processed full-split frame inventory contains 137,120 frames at 16 frames per usable pair. A separate 144-frame pilot extraction is present locally and is not included in the 137,120 full-split figure. The recovery report flags one missing or corrupt official training video; the reported usable-pair counts should not be read as proof that every official source video was recovered.

The local `datasets/` directory contains approximately 2.15 GB of source or generated data and is intentionally excluded from GitHub. This summary does not include any original videos, frames, feature arrays, clip IDs, or participant IDs.

## Champion validation result

The Temporal Attention BiLSTM training history records a best validation Macro-F1 of **0.298212** (epoch 6; rounded to **0.2982** in the project README). The `best_attention_lstm.pth` checkpoint is the live champion; its saved feature normalization and the MediaPipe task asset are required for inference.

The validation comparison report records mean Macro-F1 values of 0.177900 for a majority baseline, 0.222269 for logistic regression, 0.284193 for a BiLSTM baseline, and 0.298212 for the temporal-attention champion. These are validation results and should not be described as test scores.

Separate experiment artifacts report test mean Macro-F1 values of 0.272633 for the visual LSTM variant and 0.217347 for the visual focal variant. These are different experiments from the champion and should not be compared to the champion's validation result as though they were measured on the same split.

## Interpretation limits

The labels are four-class estimates from visual cues. The live classifier uses MobileNetV3-Small visual features; the eight MediaPipe-derived geometry values are displayed as metrics but are not classifier inputs. These outputs are not diagnoses or measurements of neural activity. The modest validation score and the acknowledged data recovery issue limit claims about generalization.
