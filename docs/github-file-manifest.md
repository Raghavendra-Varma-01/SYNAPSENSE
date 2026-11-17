# Proposed SYNAPSENSE GitHub file manifest

**Status:** Phase 2 recommendation only. No files have been staged, committed, or pushed. This manifest does not change the target repository's visibility.

## Recommended upload files

Paths below are relative to the SYNAPSENSE repository root. This list includes the existing source files and the new sanitized documentation/template files. Local datasets, local recovery settings, and non-champion model binaries are not included.

### Root, app, config, docs, reports, and tests

```text
.gitignore
README.md
app/app.js
app/assets/synapsense-emblem.png
app/index.html
app/styles.css
configs/daisee_recovery.example.json
docs/github-file-manifest.md
docs/research-results.md
reports/IMPLEMENTATION_AUDIT.md
tests/test_environment.py
```

### Required inference assets and sanitized model results

```text
models/baseline_comparison/validation_baseline_results.json
models/error_analysis/validation_error_analysis.json
models/mediapipe/face_landmarker.task
models/ordinal_lstm/training_history.json
models/visual_attention/attention_error_analysis.json
models/visual_attention/best_attention_lstm.pth
models/visual_attention/feature_normalization.npz
models/visual_attention/training_history.json
models/visual_attention_focal/training_history.json
models/visual_attention_sampler/training_history.json
models/visual_focal/test_metrics.json
models/visual_focal/training_history.json
models/visual_lstm/boredom_confusion_matrix.csv
models/visual_lstm/confusion_confusion_matrix.csv
models/visual_lstm/engagement_confusion_matrix.csv
models/visual_lstm/frustration_confusion_matrix.csv
models/visual_lstm/test_metrics.json
models/visual_lstm/training_history.json
```

The JSON/CSV evaluation and training-history files above preserve the recorded research results without including the alternative trained weights. The local path scan found no machine-specific paths, Drive links, or credential patterns in these proposed model result files.

### Source files

```text
src/data/audit_annotations.py
src/data/build_manifest.py
src/data/check_dataset_structure.py
src/data/check_missing_participants.py
src/data/daisee_recovery.py
src/data/dataset_quality_report.py
src/data/deep_dataset_check.py
src/data/inspect_daisee.py
src/data/prepare_final_manifests.py
src/features/extract_facial_features.py
src/features/prepare_fusion_features.py
src/features/test_face_landmarker.py
src/inference/live_server.py
src/preprocessing/check_model_environment.py
src/preprocessing/extract_cnn_features.py
src/preprocessing/extract_facial_features.py
src/preprocessing/extract_frames_full.py
src/preprocessing/extract_frames_pilot.py
src/preprocessing/verify_extracted_frames.py
src/training/analyze_attention.py
src/training/analyze_visual_errors.py
src/training/compare_visual_baselines.py
src/training/fix_clip_ids.py
src/training/focal_loss.py
src/training/train_synapsense_attention.py
src/training/train_synapsense_attention_focal.py
src/training/train_synapsense_focal.py
src/training/train_synapsense_fusion.py
src/training/train_synapsense_late_fusion.py
src/training/train_synapsense_lstm.py
src/training/train_synapsense_ordinal.py
src/training/train_synapsense_sampler.py
src/training/verify_feature_dataset.py
```

## Excluded files and reasons

### Recursive directories and generated files

| Path or pattern | Reason |
| --- | --- |
| `.git/**` | Local repository metadata is not project content; Git transports committed history, not this directory as a file. |
| `.venv/**`, `venv/**`, `env/**`, `ENV/**` | Machine-specific Python environments and installed binaries. |
| `datasets/**` | Contains 137,310 local data/generated files (~2.15 GB), including video/frame/feature content. Dataset files are not to be uploaded. Individual dataset filenames are intentionally not reproduced in this manifest. |
| `artifacts/**` | Local/generated artifacts; currently empty. |
| `__pycache__/**`, `*.py[cod]`, `.ipynb_checkpoints/**`, `.pytest_cache/**`, `.mypy_cache/**`, `.ruff_cache/**`, `.cache/**` | Interpreter and tool caches, not source or research results. |
| `logs/**`, `tmp/**`, `temp/**`, `*.log`, `*.tmp`, `*.temp`, editor swap/backup files | Local runtime output and temporary files. |
| `.vscode/**`, `.idea/**`, `.vs/**`, `*.user`, `*.suo`, `*.code-workspace`, `.DS_Store`, `Thumbs.db`, `desktop.ini` | IDE, OS, and machine-specific state. |
| `.env`, `.env.*` (except `.env.example`), `*.pem`, `*.key`, `*.p12`, `*.pfx`, `.secrets/**`, `secrets.*`, `credentials.*` | Secret/credential material. No credential pattern was found in the proposed upload files. |
| `notebooks/**` | No files are present currently; do not include local checkpoints or data if notebooks are later added. |

### Exact local-only config, reports, and model files

| File(s) | Reason |
| --- | --- |
| `configs/daisee_recovery.json` | Contains user-specific absolute paths and a private Drive file link. A sanitized placeholder template is recommended instead. |
| `reports/DATASET_PLAN.md` | Historical plan marks additional audio/fatigue datasets as pending; they are not part of the implemented visual-only project and the plan is not a current result. Preserve locally. |
| `reports/dataset_recovery/corrupt_videos.csv` | Raw dataset recovery output; keep with local data records. |
| `reports/dataset_recovery/dataset_recovery_report.json` | Contains local paths, archive-discovery records, and a private Drive link. |
| `reports/dataset_recovery/dataset_recovery_report.txt` | Contains machine-specific paths. |
| `reports/dataset_recovery/duplicate_videos.csv` | Raw dataset inventory output. |
| `reports/dataset_recovery/final_dataset_manifest.csv` | Contains absolute local paths and per-clip inventory data. |
| `reports/dataset_recovery/missing_videos_test.csv` | Raw dataset inventory output. |
| `reports/dataset_recovery/missing_videos_train.csv` | Raw dataset inventory output. |
| `reports/dataset_recovery/missing_videos_validation.csv` | Raw dataset inventory output. |
| `reports/dataset_recovery/recovery_log.txt` | Contains machine-specific paths and local archive-search details. |
| `models/ordinal_lstm/best_ordinal_lstm.pth`, `models/ordinal_lstm/feature_normalization.npz` | Experimental checkpoint and normalization; not used by live inference. Preserve locally. Its training history remains in the upload list. |
| `models/visual_attention/best_attention_lstm_backup.pth`, `models/visual_attention/champion_backup.pth` | Redundant checkpoint backups; live inference uses `best_attention_lstm.pth`. Preserve locally. |
| `models/visual_attention_focal/best_attention_focal.pth`, `models/visual_attention_focal/feature_normalization.npz` | Experimental checkpoint and normalization; not used by live inference. Preserve locally. |
| `models/visual_attention_sampler/best_attention_sampler.pth`, `models/visual_attention_sampler/feature_normalization.npz` | Experimental checkpoint and normalization; not used by live inference. Preserve locally. |
| `models/visual_focal/best_focal_lstm.pth`, `models/visual_focal/feature_normalization.npz` | Experimental checkpoint and normalization; not used by live inference. Preserve locally. |
| `models/visual_fusion/best_fusion_attention.pth` | Experimental checkpoint; not used by live inference. Preserve locally. |
| `models/visual_late_fusion/best_late_fusion.pth`, `models/visual_late_fusion/late_fusion_backup.pth` | Experimental checkpoints; not used by live inference. Preserve locally. |
| `models/visual_lstm/best_visual_lstm.pth`, `models/visual_lstm/feature_normalization.npz` | Experimental checkpoint and normalization; not used by live inference. Its test metrics and confusion matrices remain in the upload list. Preserve locally. |

The data/recovery source code remains in the proposed upload list. The sanitized example config contains placeholders only; a user must make a local ignored config and point it to their own dataset before running recovery tools.

## Essential inference assets and size review

| Exact required path | Local size | Purpose |
| --- | ---: | --- |
| `models/visual_attention/best_attention_lstm.pth` | 3.02 MB | Champion Temporal Attention BiLSTM checkpoint. |
| `models/visual_attention/feature_normalization.npz` | 0.01 MB | Saved training feature mean and standard deviation. |
| `models/mediapipe/face_landmarker.task` | 3.58 MB | MediaPipe Face Landmarker task asset. |

The largest proposed file is the 3.58 MB MediaPipe task asset. All proposed files are below GitHub's 50 MiB warning threshold and 100 MiB hard per-file block limit; no Git LFS is needed for this proposed set. The excluded local dataset contains at least one file around 191 MB, above GitHub's regular-Git hard limit, in addition to being explicitly prohibited from upload. See [GitHub's repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits).

### TorchVision's external pretrained weights

`src/inference/live_server.py` asks TorchVision for `MobileNet_V3_Small_Weights.DEFAULT`. In a clean environment, TorchVision downloads `mobilenet_v3_small-047dcff4.pth` from `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth` if it is not cached, with hash-prefix checking. The locally cached file is 10,306,551 bytes and its observed SHA-256 is `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. The file is not copied into this repository; a first model start needs outbound access to the official URL. This behavior is documented in the README, so a hidden local cache is not a deployment prerequisite.

## Sanitization and security review

- `configs/daisee_recovery.json` and every file under `reports/dataset_recovery/` are local-only and explicitly ignored. Their useful aggregate counts and model results are summarized without paths or clip IDs in `docs/research-results.md`.
- `configs/daisee_recovery.example.json` uses only placeholders; the private Drive link and all actual machine paths are omitted.
- `reports/IMPLEMENTATION_AUDIT.md` contains no local paths or credential patterns and is included as a historical implementation note; the root README is authoritative for current architecture and status.
- A filename-only scan for common key/token/private-key patterns found none in the proposed upload files. The actual `.env`/credential patterns remain ignored.
- No requirements lockfile, `LICENSE`, Vercel configuration, OCI configuration, or production secrets are added in Phase 2.

## Git safety state

The repository root is the SYNAPSENSE project directory. The remote was inspected read-only; Git transport returned success but advertised no refs (`--exit-code` returned 2). The unauthenticated web page also returned 404, so remote visibility/state cannot be conclusively inferred. No remote was added, and no files were staged, committed, pushed, or published.
