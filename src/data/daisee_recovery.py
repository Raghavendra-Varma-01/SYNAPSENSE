from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import cv2
except ImportError:  # Report dependency failure instead of hiding it at import time.
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None


VIDEO_EXTENSIONS = {".avi", ".mp4"}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar", ".tar", ".tgz", ".gz"}
SPLITS = ("Train", "Validation", "Test")
LABEL_COLUMNS = ("Boredom", "Engagement", "Confusion", "Frustration")
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "$recycle.bin"}


def normalized_name(value: Any) -> str:
    """Normalize a listed or annotated clip identifier without dropping its extension."""
    text = str(value or "").strip().strip("\ufeff").replace("\\", "/")
    return PurePosixPath(text).name.casefold()


def participant_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem[:6] if len(stem) >= 6 else stem


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def setup_logger(report_dir: Path) -> logging.Logger:
    report_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daisee_recovery")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(report_dir / "recovery_log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def scan_archives(search_roots: list[Path], logger: logging.Logger, excluded_roots: list[Path] | None = None, max_depth: int = 5) -> list[Path]:
    found: list[Path] = []
    excluded = {p.resolve() for p in (excluded_roots or []) if p.exists()}
    for root in search_roots:
        if not root.is_dir():
            continue
        logger.info("Searching for archives under %s", root)
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            if any(current_path == skip or skip in current_path.parents for skip in excluded):
                dirs[:] = []
                continue
            depth = len(current_path.relative_to(root).parts)
            if depth >= max_depth:
                dirs[:] = []
            dirs[:] = [d for d in dirs if d.casefold() not in EXCLUDED_DIRS]
            for name in files:
                path = Path(current) / name
                suffixes = {s.casefold() for s in path.suffixes}
                if suffixes & ARCHIVE_EXTENSIONS or name.casefold().endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
                    found.append(path)
    return sorted(set(found), key=lambda p: str(p).casefold())


def seven_zip_executable() -> str | None:
    return shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")


def archive_members(archive: Path) -> tuple[list[dict[str, Any]], str | None]:
    suffix = archive.suffix.casefold()
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as zf:
                return ([{"name": i.filename, "size": i.file_size, "is_dir": i.is_dir()} for i in zf.infolist()], None)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            return [], f"ZIP listing failed: {exc}"
    tool = seven_zip_executable()
    if tool:
        try:
            result = subprocess.run([tool, "l", "-slt", str(archive)], capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                return [], f"7-Zip listing failed: {result.stderr.strip() or result.stdout.strip()}"
            members: list[dict[str, Any]] = []
            current: dict[str, Any] = {}
            for line in result.stdout.splitlines():
                if line.startswith("Path = "):
                    if current.get("name"):
                        members.append(current)
                    current = {"name": line[7:], "size": 0, "is_dir": False}
                elif line.startswith("Size = "):
                    try:
                        current["size"] = int(line[7:])
                    except ValueError:
                        pass
                elif line.strip() == "Folder = +":
                    current["is_dir"] = True
            if current.get("name"):
                members.append(current)
            # The first Path entry describes the archive itself, not a member.
            members = [m for m in members if m.get("name") != str(archive)]
            return members, None
        except (OSError, subprocess.SubprocessError) as exc:
            return [], f"7-Zip listing failed: {exc}"
    if suffix in {".rar", ".7z", ".tar", ".gz", ".tgz"}:
        return [], f"Cannot inspect {suffix} archive: install 7-Zip (7z.exe) and add it to PATH."
    return [], f"Unsupported archive format: {suffix}"


def split_and_relative(member_name: str) -> tuple[str | None, str | None]:
    parts = [p for p in member_name.replace("\\", "/").split("/") if p and p != "."]
    for i, part in enumerate(parts):
        if part.casefold() == "dataset" and i + 1 < len(parts) and parts[i + 1].casefold() in {s.casefold() for s in SPLITS}:
            return parts[i + 1].title(), "/".join(parts[i + 2 :])
    for i, part in enumerate(parts):
        if part.casefold() in {s.casefold() for s in SPLITS}:
            return part.title(), "/".join(parts[i + 1 :])
    return None, None


def index_dataset_videos(dataset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for split in SPLITS:
        for key, paths in _index_one_split(dataset_root / split).items():
            index[key].extend(paths)
    return dict(index)


def index_split_videos(dataset_root: Path) -> dict[str, dict[str, list[Path]]]:
    return {split: _index_one_split(dataset_root / split) for split in SPLITS}


def _index_one_split(split_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if split_dir.is_dir():
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS:
                index[normalized_name(path.name)].append(path)
    return dict(index)


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.strip()]


def read_annotations(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "rows": 0, "duplicate_ids": [], "missing_columns": []}
    if not path.is_file():
        return [], meta
    if pd is None:
        raise RuntimeError("pandas is required. Install it with: python -m pip install pandas")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    meta["rows"] = int(len(frame))
    required = {"ClipID", *LABEL_COLUMNS}
    meta["missing_columns"] = sorted(required - set(frame.columns))
    if "ClipID" not in frame.columns:
        return [], meta
    frame["_norm"] = frame["ClipID"].map(normalized_name)
    dupes = frame.loc[frame["_norm"].duplicated(keep=False), "ClipID"].astype(str).tolist()
    meta["duplicate_ids"] = sorted(set(dupes))
    return frame.to_dict(orient="records"), meta


def validate_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"readable": False, "frame_count": None, "fps": None, "width": None, "height": None, "duration_seconds": None, "sampled_frames": 0, "error": ""}
    if cv2 is None:
        result["error"] = "opencv-python is not installed"
        return result
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result["error"] = "OpenCV could not open file"
            return result
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        result.update(frame_count=frames, fps=fps, width=width, height=height, duration_seconds=(frames / fps if frames > 0 and fps > 0 else None))
        if frames <= 0 or width <= 0 or height <= 0:
            result["error"] = "empty or invalid stream metadata"
            return result
        positions = sorted({0, frames // 2, max(0, frames - 1)})
        decoded = 0
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                decoded += 1
        result["sampled_frames"] = decoded
        result["readable"] = decoded > 0
        if not decoded:
            result["error"] = "no sampled frame could be decoded"
    except Exception as exc:  # codecs can raise native-wrapper exceptions
        result["error"] = f"validation error: {exc}"
    finally:
        cap.release()
    return result


def extract_missing(archive: Path, members: list[dict[str, Any]], missing_names: set[str], dataset_root: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in members:
        if member.get("is_dir"):
            continue
        split, relative = split_and_relative(str(member["name"]))
        if split and relative and Path(relative).suffix.casefold() in VIDEO_EXTENSIONS and normalized_name(relative) in missing_names:
            candidates[normalized_name(relative)].append({**member, "split": split, "relative": relative})
    if archive.suffix.casefold() != ".zip" and not seven_zip_executable():
        return recovered
    for key in sorted(missing_names):
        matches = candidates.get(key, [])
        if len(matches) != 1:
            logger.warning("Not extracting %s: archive has %d matching members", key, len(matches))
            continue
        item = matches[0]
        # Keep the split-relative folder structure; reject traversal and absolute paths.
        relative = PurePosixPath(item["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            logger.error("Rejected unsafe archive path %s", item["name"])
            continue
        target = (dataset_root / item["split"] / Path(*relative.parts)).resolve()
        if not target.is_relative_to(dataset_root.resolve()):
            logger.error("Rejected target outside dataset root: %s", target)
            continue
        expected_size = int(item.get("size", 0) or 0)
        if target.exists():
            state = "already_exists_same_size" if target.is_file() and target.stat().st_size == expected_size else "blocked_existing_file"
            logger.debug("Skipping existing target (%s): %s", state, target)
            recovered.append({"filename": target.name, "path": str(target), "status": state, "bytes": target.stat().st_size if target.is_file() else None})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".daisee-recovery-part")
        try:
            if archive.suffix.casefold() == ".zip":
                with zipfile.ZipFile(archive) as zf, zf.open(str(item["name"])) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
            else:
                tool = seven_zip_executable()
                if not tool:
                    raise RuntimeError("7-Zip CLI unavailable")
                with __import__("tempfile").TemporaryDirectory(prefix="daisee_extract_") as temp:
                    result = subprocess.run([tool, "x", "-y", f"-o{temp}", str(archive), str(item["name"])], capture_output=True, text=True, timeout=3600)
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
                    extracted_candidates = list(Path(temp).rglob(Path(item["name"]).name))
                    if len(extracted_candidates) != 1:
                        raise RuntimeError(f"Expected one extracted member, got {len(extracted_candidates)}")
                    shutil.copyfile(extracted_candidates[0], temporary)
            actual_size = temporary.stat().st_size
            if expected_size and actual_size != expected_size:
                temporary.unlink(missing_ok=True)
                raise IOError(f"size mismatch: expected {expected_size}, got {actual_size}")
            if target.exists():
                temporary.unlink(missing_ok=True)
                raise FileExistsError(f"target appeared during extraction: {target}")
            temporary.replace(target)
            check = validate_video(target)
            status = "recovered_verified" if check["readable"] else "recovered_but_unreadable"
            row = {"filename": target.name, "path": str(target), "status": status, "bytes": actual_size, "video_validation": check}
            recovered.append(row)
            logger.info("%s: %s (%d bytes)", status, target, actual_size)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            logger.exception("Extraction failed for %s: %s", item["name"], exc)
            recovered.append({"filename": Path(item["relative"]).name, "path": str(target), "status": "extraction_failed", "error": str(exc)})
    return recovered


def recovery_records_from_log(log_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    """Restore verified paths from an interrupted earlier run, then revalidate them this run."""
    if not log_path.is_file():
        return []
    pattern = re.compile(r"INFO recovered_verified: (.+) \((\d+) bytes\)$")
    records: dict[str, dict[str, Any]] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = pattern.search(line.rstrip())
                if not match:
                    continue
                path = Path(match.group(1))
                try:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(dataset_root.resolve()) or not resolved.is_file():
                        continue
                    size = int(match.group(2))
                    if resolved.stat().st_size != size:
                        continue
                except OSError:
                    continue
                records[str(resolved).casefold()] = {"filename": resolved.name, "path": str(resolved), "status": "recovered_from_prior_run_pending_revalidation", "bytes": size}
    except OSError:
        return []
    return list(records.values())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            writer.writerow(cooked)


def run(config_path: Path, args: argparse.Namespace) -> int:
    config = read_config(config_path)
    project_root = Path(config["project_root"]).expanduser()
    dataset_root = Path(config["dataset_root"]).expanduser()
    report_dir = project_root / config.get("report_dir", "reports/dataset_recovery")
    logger = setup_logger(report_dir)
    started = datetime.now(timezone.utc).isoformat()
    baseline = config.get("baseline", {})
    report: dict[str, Any] = {"generated_at_utc": started, "project_root": str(project_root), "dataset_root": str(dataset_root), "status": "WARNING", "stages": {}, "splits": {}, "archive_discovery": {}, "recovered_files": []}

    packages = {}
    for package in ("pandas", "opencv-python", "opencv-python-headless", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        free = shutil.disk_usage(dataset_root if dataset_root.exists() else project_root).free
    except OSError:
        free = None
    search_roots = [Path(p) for p in config.get("archive_search_roots", [])]
    archive_paths = scan_archives(search_roots, logger, [dataset_root, *[Path(p) for p in config.get("archive_search_exclude_roots", [])]], int(config.get("archive_search_max_depth", 5)))
    drive_candidates = [p for p in archive_paths if p.name.casefold() == str(config.get("original_archive_filename", "DAiSEE.zip")).casefold()]
    report["stages"]["environment"] = {"python_version": sys.version, "python_compatible": sys.version_info >= (3, 10), "packages": packages, "paths": {"project_root_exists": project_root.is_dir(), "dataset_root_exists": dataset_root.is_dir(), "data_set_dir_exists": (dataset_root / "DataSet").is_dir(), "labels_dir_exists": (dataset_root / "Labels").is_dir()}, "available_disk_bytes": free, "google_drive_link": config.get("google_drive_url"), "local_original_archive_candidates": [str(p) for p in drive_candidates], "archive_assumed_local": False}

    data_root = dataset_root / "DataSet"
    label_root = dataset_root / "Labels"
    splits_lists: dict[str, list[str]] = {s: read_split(data_root / f"{s}.txt") for s in SPLITS}
    labels_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    annotation_meta = {}
    for split in SPLITS:
        records, meta = read_annotations(label_root / f"{split}Labels.csv")
        annotation_meta[split] = meta
        normalized: dict[str, dict[str, Any]] = {}
        for row in records:
            normalized[normalized_name(row.get("ClipID", ""))] = row
        labels_by_split[split] = normalized

    archive_paths = [p for p in archive_paths if str(p).casefold() not in {str(x).casefold() for x in drive_candidates}]
    all_archives = sorted(set(archive_paths + drive_candidates), key=lambda p: str(p).casefold())
    archive_data = []
    for archive in all_archives:
        members, error = archive_members(archive)
        rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
        for member in members:
            split, relative = split_and_relative(str(member.get("name", "")))
            if split and relative and Path(relative).suffix.casefold() in VIDEO_EXTENSIONS and not member.get("is_dir"):
                rows_by_split[split].append(member)
        compare = {}
        for split in SPLITS:
            split_ids = {normalized_name(n) for n in splits_lists[split]}
            archive_ids = {normalized_name(Path(str(m["name"])).name) for m in rows_by_split[split]}
            absent_from_archive = sorted(split_ids - archive_ids)
            compare[split] = {"video_count": len(rows_by_split[split]), "avi_count": sum(Path(str(m["name"])).suffix.casefold() == ".avi" for m in rows_by_split[split]), "mp4_count": sum(Path(str(m["name"])).suffix.casefold() == ".mp4" for m in rows_by_split[split]), "split_list_name_matches": len(split_ids & archive_ids), "missing_split_names_count": len(absent_from_archive), "missing_split_names_sample": absent_from_archive[:20]}
        archive_data.append({"path": str(archive), "size_bytes": archive.stat().st_size, "member_count": len(members), "error": error, "splits": compare, "_members": members})
    report["archive_discovery"] = {"search_roots": config.get("archive_search_roots", []), "max_search_depth": int(config.get("archive_search_max_depth", 5)), "archives_found": [{k: v for k, v in a.items() if k != "_members"} for a in archive_data], "seven_zip_available": bool(seven_zip_executable())}

    # Build current state and missing-name candidates. Extraction is gated twice: explicit flags and valid archive.
    videos = index_dataset_videos(data_root)
    videos_by_split = index_split_videos(data_root)
    missing_by_split: dict[str, list[str]] = {}
    for split in SPLITS:
        list_ids = {normalized_name(n) for n in splits_lists[split]}
        actual_ids = set(videos_by_split[split])
        missing_by_split[split] = sorted(list_ids - actual_ids)
    if args.recover:
        if not args.confirm_recovery:
            logger.error("Recovery requested without --confirm-recovery; no files changed.")
        elif not args.archive:
            logger.error("Recovery requires --archive PATH; no files changed.")
        else:
            chosen = Path(args.archive).expanduser()
            archive_item = next((a for a in archive_data if Path(a["path"]).resolve() == chosen.resolve()), None)
            if not archive_item:
                members, error = archive_members(chosen)
                archive_item = {"path": str(chosen), "_members": members, "error": error}
            if archive_item.get("error"):
                logger.error("Archive is not valid/inspectable: %s", archive_item["error"])
            else:
                prior_records = recovery_records_from_log(report_dir / "recovery_log.txt", data_root)
                missing_union = {name for names in missing_by_split.values() for name in names}
                fresh_records = extract_missing(chosen, archive_item["_members"], missing_union, data_root, logger)
                combined_records = {str(Path(row["path"]).resolve()).casefold(): row for row in prior_records}
                combined_records.update({str(Path(row["path"]).resolve()).casefold(): row for row in fresh_records if row.get("path")})
                report["recovered_files"] = list(combined_records.values())
                videos = index_dataset_videos(data_root)
                videos_by_split = index_split_videos(data_root)
                for split in SPLITS:
                    missing_by_split[split] = sorted({normalized_name(n) for n in splits_lists[split]} - set(videos_by_split[split]))

    manifest: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    corrupt_rows: list[dict[str, Any]] = []
    split_ids: dict[str, set[str]] = {}
    all_video_paths = [p for found in videos.values() for p in found]
    file_counts = Counter(p.suffix.casefold() for p in all_video_paths)
    for key, paths in videos.items():
        if len(paths) > 1:
            duplicate_rows.append({"filename": paths[0].name, "normalized_filename": key, "duplicate_count": len(paths), "paths": [str(p) for p in paths], "reason": "same normalized filename"})

    recovery_records_by_path = {str(Path(row["path"]).resolve()).casefold(): row for row in report["recovered_files"] if row.get("path")}

    for split in SPLITS:
        listed = splits_lists[split]
        listed_ids = [normalized_name(n) for n in listed]
        split_ids[split] = set(listed_ids)
        label_map = labels_by_split[split]
        split_videos = videos_by_split[split]
        annotation = annotation_meta[split]
        available = 0
        pair_count = 0
        checked: set[Path] = set()
        split_rows: list[dict[str, Any]] = []
        for entry_number, name in enumerate(listed, start=1):
            if entry_number == 1 or entry_number % 250 == 0 or entry_number == len(listed):
                logger.info("Reconciling %s: %d/%d split entries", split, entry_number, len(listed))
            key = normalized_name(name)
            paths = split_videos.get(key, [])
            chosen = paths[0] if len(paths) == 1 else None
            if chosen:
                available += 1
            label = label_map.get(key)
            validation = None
            if chosen and chosen not in checked:
                validation = validate_video(chosen)
                checked.add(chosen)
                recovery_record = recovery_records_by_path.get(str(chosen.resolve()).casefold())
                if recovery_record:
                    recovery_record["status"] = "recovered_verified" if validation["readable"] else "recovered_but_unreadable"
                    recovery_record["bytes"] = chosen.stat().st_size
                    recovery_record["video_validation"] = validation
                if not validation["readable"]:
                    corrupt_rows.append({"split": split, "filename": chosen.name, "path": str(chosen), **validation})
            elif chosen:
                # In the unusual case an entry repeats in a split list, avoid decoding it twice.
                validation = {"readable": chosen not in {Path(r["path"]) for r in corrupt_rows}, "reused_validation": True}
            is_usable = bool(chosen and label and validation and validation.get("readable"))
            pair_count += int(is_usable)
            missing_label = label is None
            actual_name_match = chosen is not None
            mismatches = [str(p) for p in videos.get(key, [])] if paths else []
            row = {"split": split, "official_filename": name, "normalized_filename": key, "participant_id": participant_id(name), "video_path": str(chosen) if chosen else "", "video_candidates": json.dumps(mismatches), "video_available": actual_name_match, "duplicate_video_match": len(paths) > 1, "annotation_available": not missing_label, "video_readable": bool(validation and validation.get("readable")), "frame_count": validation.get("frame_count") if validation else "", "fps": validation.get("fps") if validation else "", "width": validation.get("width") if validation else "", "height": validation.get("height") if validation else "", "duration_seconds": validation.get("duration_seconds") if validation else "", "usable_video_label_pair": is_usable, "status": "usable" if is_usable else ("missing_video" if not actual_name_match else "duplicate_video" if len(paths) > 1 else "missing_annotation" if missing_label else "corrupt_video")}
            manifest.append(row)
            split_rows.append(row)
        listed_set = set(listed_ids)
        annot_ids = set(label_map)
        annotation_without_split = sorted(annot_ids - listed_set)
        video_without_split = sorted(set(split_videos) - listed_set)
        missing_labels = sorted(listed_set - annot_ids)
        missing_videos = missing_by_split[split]
        file_missing = [{"split": split, "filename": name, "participant_id": participant_id(name), "status": "missing_video", "annotation_available": normalized_name(name) in annot_ids} for name in [original for original in listed if normalized_name(original) in set(missing_videos)]]
        # Add likely filename mismatch candidates (same stem, different extension/case-independent ID).
        for original in listed:
            key = normalized_name(original)
            if key not in set(missing_videos):
                continue
            stem = Path(original).stem.casefold()
            similar = [p for p in all_video_paths if p.stem.casefold() == stem and p.suffix.casefold() in VIDEO_EXTENSIONS]
            if similar:
                file_missing.append({"split": split, "filename": original, "participant_id": participant_id(original), "status": "filename_mismatch_candidate", "found_paths": [str(p) for p in similar]})
        write_csv(report_dir / f"missing_videos_{split.casefold()}.csv", file_missing)
        unique_participants = sorted({participant_id(n) for n in listed if n})
        annotations_without_video = len(annot_ids - set(split_videos))
        videos_without_annotation = len(set(split_videos) - annot_ids)
        report["splits"][split] = {"official_split_entries": len(listed), "annotation_records": annotation["rows"], "available_videos": available, "usable_video_label_pairs": pair_count, "split_ids_without_labels": len(missing_labels), "annotations_without_split_entries": len(annotation_without_split), "annotations_without_videos": annotations_without_video, "videos_without_annotations": videos_without_annotation, "videos_without_split_entries": len(video_without_split), "duplicate_annotation_ids": len(annotation["duplicate_ids"]), "duplicate_video_filenames": sum(1 for d in duplicate_rows if d["normalized_filename"] in listed_set), "unique_participant_ids": len(unique_participants), "participant_ids": unique_participants, "remaining_missing_videos": len(missing_videos), "annotation_metadata": annotation, "cross_split_ids": []}

    # Leakage: a filename or participant appearing across official splits is recorded explicitly.
    participant_sets = {s: {participant_id(n) for n in splits_lists[s]} for s in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            common_clips = sorted(split_ids[left] & split_ids[right])
            common_participants = sorted(participant_sets[left] & participant_sets[right])
            report["splits"][left]["cross_split_ids"].append({"other_split": right, "duplicate_clip_ids": common_clips, "shared_participant_ids": common_participants})
            report["splits"][right]["cross_split_ids"].append({"other_split": left, "duplicate_clip_ids": common_clips, "shared_participant_ids": common_participants})

    # Duplicate file contents are only hashed among same-size videos to constrain disk I/O.
    size_groups: dict[int, list[Path]] = defaultdict(list)
    for path in all_video_paths:
        try:
            size_groups[path.stat().st_size].append(path)
        except OSError:
            pass
    for group in size_groups.values():
        if len(group) < 2:
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in group:
            digest = hashlib.sha256()
            try:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
                        digest.update(chunk)
                hashes[digest.hexdigest()].append(path)
            except OSError as exc:
                logger.warning("Could not hash %s: %s", path, exc)
        for digest, paths in hashes.items():
            if len(paths) > 1 and not any(set(map(str, paths)) == set(d.get("paths", [])) for d in duplicate_rows):
                duplicate_rows.append({"filename": paths[0].name, "duplicate_count": len(paths), "paths": [str(p) for p in paths], "sha256": digest, "reason": "identical content among same-size files"})

    write_csv(report_dir / "duplicate_videos.csv", duplicate_rows)
    write_csv(report_dir / "corrupt_videos.csv", corrupt_rows)
    write_csv(report_dir / "final_dataset_manifest.csv", manifest)
    cross_split_clips = sorted(set.union(*(split_ids[s] for s in SPLITS)) if SPLITS else set())
    duplicate_clip_ids = []
    for clip in cross_split_clips:
        present = [s for s in SPLITS if clip in split_ids[s]]
        if len(present) > 1:
            duplicate_clip_ids.append({"clip_id": clip, "splits": present})
    shared_participants = []
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            for pid in sorted(participant_sets[left] & participant_sets[right]):
                shared_participants.append({"participant_id": pid, "splits": [left, right]})

    report["stages"]["reconciliation"] = {"dataset_video_file_count": len(all_video_paths), "avi_count": file_counts[".avi"], "mp4_count": file_counts[".mp4"], "annotation_files": annotation_meta, "duplicate_video_filename_groups": len([d for d in duplicate_rows if d.get("reason") == "same normalized filename"]), "corrupt_video_count": len(corrupt_rows), "cross_split_duplicate_clip_ids": duplicate_clip_ids, "cross_split_shared_participants": shared_participants, "archive_comparison": [{k: v for k, v in a.items() if k != "_members"} for a in archive_data]}
    report["stages"]["recovery"] = {"requested": bool(args.recover), "confirmation_flag_present": bool(args.confirm_recovery), "recovered_files": report["recovered_files"], "remaining_missing_by_split": {s: len(missing_by_split[s]) for s in SPLITS}, "safety": "No existing file was overwritten or deleted."}
    video_splits_ok = all(report["splits"][s]["available_videos"] == report["splits"][s]["official_split_entries"] and report["splits"][s]["official_split_entries"] > 0 and report["splits"][s]["remaining_missing_videos"] == 0 for s in SPLITS)
    annotation_pairs_ok = all(report["splits"][s]["usable_video_label_pairs"] > 0 and report["splits"][s]["duplicate_annotation_ids"] == 0 for s in SPLITS)
    report["dataset_integrity"] = {"official_video_splits_available_and_validated": video_splits_ok and not corrupt_rows and not duplicate_clip_ids and not shared_participants, "video_split_status": "PASS" if video_splits_ok and not corrupt_rows and not duplicate_clip_ids and not shared_participants else "FAIL", "all_splits_have_usable_video_label_pairs": annotation_pairs_ok, "usable_labeled_split_status": "PASS" if annotation_pairs_ok else "FAIL", "unlabeled_official_split_entries": {s: report["splits"][s]["split_ids_without_labels"] for s in SPLITS}}
    hard_errors = (not project_root.is_dir()) or (not data_root.is_dir()) or any(not annotation_meta[s]["exists"] for s in SPLITS) or any(report["splits"][s]["duplicate_annotation_ids"] for s in SPLITS)
    report["status"] = "PASS" if video_splits_ok and annotation_pairs_ok and not hard_errors and not corrupt_rows and not duplicate_clip_ids and not shared_participants and all(report["splits"][s]["split_ids_without_labels"] == 0 for s in SPLITS) else "FAIL" if hard_errors else "WARNING"
    integrity_findings = []
    missing_total = sum(report["stages"]["recovery"]["remaining_missing_by_split"].values())
    if missing_total:
        integrity_findings.append(f"{missing_total} official split-list video(s) remain missing.")
    extraction_failures = [row for row in report["recovered_files"] if row.get("status") == "extraction_failed"]
    for row in extraction_failures:
        integrity_findings.append(f"Archive extraction failed for {row.get('filename', 'unknown file')}: {row.get('error', 'unknown error')}")
    unlabeled = {s: report["splits"][s]["split_ids_without_labels"] for s in SPLITS if report["splits"][s]["split_ids_without_labels"]}
    if unlabeled:
        integrity_findings.append(f"Official split entries without annotations: {unlabeled}.")
    if corrupt_rows:
        integrity_findings.append(f"{len(corrupt_rows)} video(s) failed OpenCV readability checks.")
    if duplicate_clip_ids or shared_participants:
        integrity_findings.append("Cross-split leakage was detected.")
    report["dataset_integrity"]["findings"] = integrity_findings
    if report["status"] == "PASS":
        report["status_reason"] = "All official splits have readable videos and usable labels."
    elif extraction_failures:
        report["status_reason"] = f"{len(extraction_failures)} archive member(s) failed integrity checks, leaving {missing_total} official video(s) missing. See dataset_integrity.findings and missing-video CSVs."
    elif video_splits_ok and annotation_pairs_ok:
        report["status_reason"] = "All official split videos are readable and every split has usable pairs, but some official entries lack annotations. See dataset_integrity.findings."
    else:
        report["status_reason"] = "One or more official splits are incomplete, have corrupt/duplicate videos, annotation problems, or cross-split leakage. See dataset_integrity.findings and per-split details."
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - args.started, 2)
    (report_dir / "dataset_recovery_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_lines = ["DAiSEE DATASET RECOVERY AND DIAGNOSTIC REPORT", f"Generated (UTC): {report['generated_at_utc']}", f"Overall status: {report['status']}", report["status_reason"], "", "ENVIRONMENT", f"Python: {sys.version.split()[0]} (3.10+ compatible: {sys.version_info >= (3, 10)})", f"pandas: {packages['pandas']}", f"OpenCV: {packages['opencv-python'] or packages['opencv-python-headless']}", f"Available disk bytes: {free}", f"Project exists: {project_root.is_dir()}; dataset exists: {dataset_root.is_dir()}", f"Original Drive archive locally found by configured filename: {bool(drive_candidates)}"]
    text_lines += ["", "SPLIT RECONCILIATION (baseline -> current)"]
    for split in SPLITS:
        stats = report["splits"][split]
        base = baseline.get(split, {})
        text_lines += [f"{split.upper()}: split entries {base.get('official_split_entries', 'n/a')} -> {stats['official_split_entries']}; annotations {base.get('annotation_records', 'n/a')} -> {stats['annotation_records']}; available videos {base.get('available_videos', 'n/a')} -> {stats['available_videos']}; usable pairs {base.get('usable_video_label_pairs', 'n/a')} -> {stats['usable_video_label_pairs']}; missing videos {stats['remaining_missing_videos']}; split IDs without labels {stats['split_ids_without_labels']}; annotations without videos {stats['annotations_without_videos']}; videos without annotations {stats['videos_without_annotations']}; participant IDs {stats['unique_participant_ids']}"]
    text_lines += ["", "ARCHIVES"]
    if not report["archive_discovery"]["archives_found"]:
        text_lines.append("No archives found under configured search roots. The Drive link was not treated as a local file and no download was attempted.")
    for archive in report["archive_discovery"]["archives_found"]:
        text_lines.append(f"{archive['path']} ({archive['size_bytes']} bytes): {archive['member_count']} members; listing error: {archive['error']}")
        for split in SPLITS:
            ac = archive["splits"][split]
            text_lines.append(f"  {split}: {ac['video_count']} videos (AVI {ac['avi_count']}, MP4 {ac['mp4_count']}), split list matches {ac['split_list_name_matches']}, split names absent from archive {ac['missing_split_names_count']}")
    text_lines += ["", "DATASET INTEGRITY", f"Official video splits available and validated: {report['dataset_integrity']['video_split_status']}", f"Each split has usable video-label pairs: {report['dataset_integrity']['usable_labeled_split_status']}", f"Unlabeled official split entries: {report['dataset_integrity']['unlabeled_official_split_entries']}"]
    text_lines.extend(report["dataset_integrity"]["findings"])
    text_lines += ["", "VIDEO VALIDATION", f"Videos indexed: {len(all_video_paths)}; AVI: {file_counts['.avi']}; MP4: {file_counts['.mp4']}; corrupt/unreadable: {len(corrupt_rows)}", "OpenCV reads metadata and samples up to three frames (start, middle, end) per unique listed video.", "", "CROSS-SPLIT CHECKS", f"Repeated clip IDs across splits: {len(duplicate_clip_ids)}", f"Participant IDs shared across splits: {len(shared_participants)}", "No clips were moved between official splits.", "", "RECOVERY", f"Recovery requested: {bool(args.recover)}; files recovered and verified: {len([r for r in report['recovered_files'] if r.get('status') == 'recovered_verified'])}", f"Remaining missing: {report['stages']['recovery']['remaining_missing_by_split']}", "Existing files were never overwritten or deleted.", "", "OUTPUTS", "dataset_recovery_report.txt", "dataset_recovery_report.json", "missing_videos_train.csv", "missing_videos_validation.csv", "missing_videos_test.csv", "duplicate_videos.csv", "corrupt_videos.csv", "recovery_log.txt", "final_dataset_manifest.csv"]
    (report_dir / "dataset_recovery_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    logger.info("Report complete: status=%s; outputs=%s", report["status"], report_dir)
    return 0 if report["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose DAiSEE splits and safely recover missing videos from a local archive.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "daisee_recovery.json")
    parser.add_argument("--archive", help="Explicit local archive path to inspect/use for recovery")
    parser.add_argument("--recover", action="store_true", help="Extract only missing split-listed videos (requires --confirm-recovery)")
    parser.add_argument("--confirm-recovery", action="store_true", help="Explicitly confirm dataset modifications for this run")
    args = parser.parse_args()
    args.started = time.monotonic()
    try:
        return run(args.config.resolve(), args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        logging.exception("Fatal DAiSEE recovery error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
