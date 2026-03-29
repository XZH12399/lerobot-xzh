# pi05_word Annotation Tool

## Purpose

A tiny browser-based tool for manual point-level annotation on top of:

- `outputs/pi05_word_annotation_round1/candidates.jsonl`
- `outputs/pi05_word_annotation_round1/images/`
- `outputs/pi05_word_annotation_round1/context/`

It is meant to reduce file-by-file clicking during the first gold-set pass.

## Features

- current frame + context strip in one page
- editable fields:
  - `primitive`
  - `target`
  - `relation`
  - `confidence`
  - `note`
- auto-save on navigation when the form is dirty
- `Use Draft` button for quick heuristic prefill
- `Next Empty` button for jumping across unlabeled samples
- filters by semantic family and annotation status

## Local usage

From repo root:

```powershell
python scripts/pi05_word_annotation_tool.py
```

Then open:

```text
http://127.0.0.1:8765
```

If your local mirror is not under `D:\home\...`, pass an explicit path map:

```powershell
python scripts/pi05_word_annotation_tool.py --path-map-to E:\home
```

## A100 usage

On the server:

```bash
python /home/ct_24210860031/XZH/project/lerobot-xzh/scripts/pi05_word_annotation_tool.py \
  --manifest /home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_word_annotation_round1/candidates.jsonl
```

Then access it through SSH port forwarding if needed.

## Shortcuts

- `Ctrl+S`: save
- `Shift+Enter`: save and next
- `Left`: previous record
- `Right`: next record

## Output behavior

Edits are written back into the JSONL manifest in place.

The tool also creates:

- `candidates.jsonl.bak`

on first launch as a simple backup copy.
