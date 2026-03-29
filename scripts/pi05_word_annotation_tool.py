#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import parse_qs, quote, urlparse


PRIMITIVE_OPTIONS = [
    "approach",
    "move_to",
    "open",
    "close",
    "align",
    "retreat",
    "follow_path",
    "hold_pose",
    "maintain",
    "release",
]
RELATION_OPTIONS = ["above", "inside", "on", "near", "contact", "none"]
CONFIDENCE_OPTIONS = ["high", "medium", "low"]


def log_progress(message: str) -> None:
    print(f"[pi05_word_annotation_tool] {message}", file=sys.stderr, flush=True)


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_draft_annotation(record: dict) -> dict[str, str | None]:
    semantic_family = normalize_optional_text(record.get("semantic_family")) or ""
    instruction = (normalize_optional_text(record.get("instruction")) or "").lower()

    primitive = None
    target = None
    relation = None

    if semantic_family == "pick":
        primitive = "approach"
        relation = "above"
        if instruction.startswith("pick up the "):
            target = instruction[len("pick up the ") :].split(" and place ", 1)[0]
    elif semantic_family == "place":
        primitive = "move_to"
        if " in the " in instruction or " inside " in instruction or " compartment " in instruction:
            relation = "inside"
        elif " on top of " in instruction or " on the " in instruction or " place it on " in instruction:
            relation = "on"
        else:
            relation = "near"

        if " and place it in the " in instruction:
            target = instruction.split(" and place it in the ", 1)[1]
        elif " and place it on the " in instruction:
            target = instruction.split(" and place it on the ", 1)[1]
        elif instruction.startswith("put "):
            for needle in [" in the ", " inside the ", " on top of the ", " on the "]:
                if needle in instruction:
                    target = instruction.split(needle, 1)[1]
                    break
    elif semantic_family == "open":
        primitive = "open"
        relation = "none"
        if "drawer" in instruction:
            target = "drawer_handle"
        elif "cabinet" in instruction:
            target = "cabinet_handle"

    if target is not None:
        target = (
            target.replace(" it", "")
            .replace(" the ", " ")
            .replace(" ", "_")
            .replace("-", "_")
            .strip(" _.,")
        )
        if target == "":
            target = None

    return {
        "primitive": primitive,
        "target": target,
        "relation": relation,
        "confidence": None,
        "note": None,
    }


def annotation_status(record: dict) -> str:
    annotation = record.get("annotation", {})
    required = [
        normalize_optional_text(annotation.get("primitive")),
        normalize_optional_text(annotation.get("target")),
        normalize_optional_text(annotation.get("relation")),
        normalize_optional_text(annotation.get("confidence")),
    ]
    if all(required):
        return "complete"
    if any(required):
        return "partial"
    return "empty"


@dataclass
class ManifestStore:
    manifest_path: Path
    path_map_from: str | None = None
    path_map_to: str | None = None

    def __post_init__(self) -> None:
        self.records = self._load_records()
        self._ensure_backup()
        self._attach_drafts()

    def _load_records(self) -> list[dict]:
        records = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "annotation" not in record:
                    record["annotation"] = {
                        "primitive": None,
                        "target": None,
                        "relation": None,
                        "confidence": None,
                        "note": None,
                    }
                records.append(record)
        return records

    def _ensure_backup(self) -> None:
        backup_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".bak")
        if backup_path.exists():
            return
        backup_path.write_text(self.manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _attach_drafts(self) -> None:
        for record in self.records:
            record["draft_annotation"] = infer_draft_annotation(record)
            record["status"] = annotation_status(record)

    def save(self) -> None:
        temp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                serializable = {k: v for k, v in record.items() if k not in {"draft_annotation", "status"}}
                handle.write(json.dumps(serializable, ensure_ascii=True) + "\n")
        temp_path.replace(self.manifest_path)

    def update_record(self, index: int, annotation: dict[str, object]) -> dict:
        record = self.records[index]
        record["annotation"] = {
            "primitive": normalize_optional_text(annotation.get("primitive")),
            "target": normalize_optional_text(annotation.get("target")),
            "relation": normalize_optional_text(annotation.get("relation")),
            "confidence": normalize_optional_text(annotation.get("confidence")),
            "note": normalize_optional_text(annotation.get("note")),
        }
        record["status"] = annotation_status(record)
        self.save()
        return record

    def summary(self) -> dict[str, int]:
        statuses = [record.get("status") or annotation_status(record) for record in self.records]
        confidence_values = [normalize_optional_text(record.get("annotation", {}).get("confidence")) for record in self.records]
        return {
            "total": len(self.records),
            "complete": sum(status == "complete" for status in statuses),
            "partial": sum(status == "partial" for status in statuses),
            "empty": sum(status == "empty" for status in statuses),
            "high_confidence": sum(conf == "high" for conf in confidence_values),
        }

    def target_suggestions(self) -> list[str]:
        suggestions = set()
        for record in self.records:
            target = normalize_optional_text(record.get("annotation", {}).get("target"))
            draft_target = normalize_optional_text(record.get("draft_annotation", {}).get("target"))
            if target:
                suggestions.add(target)
            if draft_target:
                suggestions.add(draft_target)
        return sorted(suggestions)

    def resolve_asset_path(self, raw_path: str) -> Path | None:
        path = Path(raw_path)
        if path.exists():
            return path

        try:
            windows_path = Path(PureWindowsPath(raw_path))
            if windows_path.exists():
                return windows_path
        except Exception:
            pass

        try:
            posix_path = PurePosixPath(raw_path)
            if posix_path.is_absolute():
                candidates: list[Path] = []
                if self.path_map_from and self.path_map_to and raw_path.startswith(self.path_map_from):
                    suffix = raw_path[len(self.path_map_from) :].lstrip("/\\")
                    candidates.append(Path(self.path_map_to, *PurePosixPath(suffix).parts))
                if self.manifest_path.anchor:
                    candidates.append(Path(self.manifest_path.anchor, *posix_path.parts[1:]))
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        except Exception:
            pass

        return None

    def api_state(self) -> dict:
        records = []
        for idx, record in enumerate(self.records):
            records.append(
                {
                    "index": idx,
                    "sample_id": record.get("sample_id"),
                    "semantic_family": record.get("semantic_family"),
                    "phase_hint": record.get("phase_hint"),
                    "task_family": record.get("task_family"),
                    "instruction": record.get("instruction"),
                    "episode_id": record.get("episode_id"),
                    "frame_index": record.get("frame_index"),
                    "annotation": record.get("annotation", {}),
                    "draft_annotation": record.get("draft_annotation", {}),
                    "status": record.get("status"),
                    "image_url": f"/api/asset?path={quote(str(record.get('image_path', '')))}",
                    "context_image_url": f"/api/asset?path={quote(str(record.get('context_image_path', '')))}",
                }
            )
        return {
            "manifest_path": str(self.manifest_path),
            "summary": self.summary(),
            "options": {
                "primitive": PRIMITIVE_OPTIONS,
                "relation": RELATION_OPTIONS,
                "confidence": CONFIDENCE_OPTIONS,
                "target_suggestions": self.target_suggestions(),
            },
            "records": records,
        }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>pi05_word Annotation Tool</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f0e8;
      --panel: #fffaf3;
      --panel-2: #f8f4ee;
      --border: #d7cbb8;
      --text: #2d251a;
      --muted: #7a6c59;
      --accent: #7f3d1f;
      --accent-2: #3f6b52;
      --empty: #b4ab9d;
      --partial: #d18b1f;
      --complete: #2d7d4f;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: linear-gradient(180deg, #f1ebe0 0%, #efe7da 100%); color: var(--text); }
    .app { display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--border); background: rgba(255, 250, 243, 0.94); display: flex; flex-direction: column; min-height: 100vh; }
    .sidebar-header, .main-header { padding: 18px 20px; border-bottom: 1px solid var(--border); background: rgba(255, 255, 255, 0.55); }
    .sidebar-header h1 { margin: 0 0 6px 0; font-size: 20px; }
    .summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .summary-card { padding: 10px; border: 1px solid var(--border); border-radius: 12px; background: var(--panel); font-size: 13px; }
    .filters { padding: 14px 18px 10px 18px; border-bottom: 1px solid var(--border); display: grid; gap: 10px; }
    .filters label { font-size: 13px; color: var(--muted); display: grid; gap: 6px; }
    select, input[type="text"], textarea { width: 100%; border: 1px solid var(--border); border-radius: 10px; background: #fff; padding: 10px 12px; font-size: 14px; color: var(--text); }
    textarea { min-height: 100px; resize: vertical; }
    .record-list { overflow: auto; padding: 10px; display: grid; gap: 8px; }
    .record-item { border: 1px solid var(--border); border-radius: 14px; background: var(--panel); padding: 10px 12px; cursor: pointer; }
    .record-item.active { outline: 2px solid rgba(127, 61, 31, 0.35); background: #fff; }
    .record-item-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 6px; }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
    .status-empty { background: var(--empty); }
    .status-partial { background: var(--partial); }
    .status-complete { background: var(--complete); }
    .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 10px; border: 1px solid var(--border); background: var(--panel-2); font-size: 12px; margin-right: 6px; margin-bottom: 6px; color: var(--muted); }
    .main { display: flex; flex-direction: column; min-width: 0; }
    .main-content { padding: 22px; display: grid; gap: 18px; }
    .card { border: 1px solid var(--border); border-radius: 18px; background: rgba(255, 250, 243, 0.92); padding: 18px; box-shadow: 0 10px 28px rgba(73, 52, 29, 0.06); }
    .instruction { font-size: 24px; line-height: 1.35; margin: 0 0 12px 0; }
    .media-grid { display: grid; grid-template-columns: minmax(0, 360px) minmax(0, 1fr); gap: 16px; align-items: start; }
    .media-card img { width: 100%; border-radius: 14px; border: 1px solid var(--border); background: #fff; }
    .media-label { font-size: 13px; color: var(--muted); margin-bottom: 8px; }
    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .form-grid .wide { grid-column: 1 / -1; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; }
    button { border: 0; border-radius: 999px; padding: 10px 16px; background: #e8d8c2; color: var(--text); cursor: pointer; font-size: 14px; }
    button.primary { background: var(--accent); color: #fff7ef; }
    button.secondary { background: var(--accent-2); color: #f5fff7; }
    button.subtle { background: #ede5da; }
    .hint { color: var(--muted); font-size: 13px; }
    .draft-box { border: 1px dashed var(--border); border-radius: 14px; background: #fff; padding: 12px; font-size: 14px; }
    .empty-state { color: var(--muted); font-size: 15px; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; }
      .media-grid { grid-template-columns: 1fr; }
      .form-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>pi05_word Annotation</h1>
        <div class="hint" id="manifestPath"></div>
        <div class="summary" id="summary"></div>
      </div>
      <div class="filters">
        <label>
          Filter by family
          <select id="familyFilter">
            <option value="all">All families</option>
            <option value="pick">pick</option>
            <option value="place">place</option>
            <option value="open">open</option>
          </select>
        </label>
        <label>
          Status filter
          <select id="statusFilter">
            <option value="all">All statuses</option>
            <option value="empty">Empty only</option>
            <option value="partial">Partial only</option>
            <option value="complete">Complete only</option>
          </select>
        </label>
      </div>
      <div class="record-list" id="recordList"></div>
    </aside>
    <main class="main">
      <div class="main-header">
        <div class="actions">
          <button class="subtle" id="prevBtn">Prev</button>
          <button class="subtle" id="nextBtn">Next</button>
          <button class="subtle" id="nextEmptyBtn">Next Empty</button>
          <button id="draftBtn">Use Draft</button>
          <button class="secondary" id="saveBtn">Save</button>
          <button class="primary" id="saveNextBtn">Save and Next</button>
          <button class="subtle" id="clearBtn">Clear</button>
        </div>
        <div class="hint" style="margin-top: 10px;">
          Shortcuts: Left / Right navigate, Ctrl+S save, Shift+Enter save and next.
        </div>
      </div>
      <div class="main-content" id="mainContent">
        <div class="card empty-state">Loading...</div>
      </div>
    </main>
  </div>
  <script>
    let appState = null;
    let currentIndex = 0;

    function byId(id) { return document.getElementById(id); }

    function getVisibleRecords() {
      if (!appState) return [];
      const familyFilter = byId('familyFilter').value;
      const statusFilter = byId('statusFilter').value;
      return appState.records.filter((record) => {
        const familyOk = familyFilter === 'all' || record.semantic_family === familyFilter;
        const statusOk = statusFilter === 'all' || record.status === statusFilter;
        return familyOk && statusOk;
      });
    }

    function findRecord(index) {
      return appState.records.find((record) => record.index === index);
    }

    function getCurrentRecord() {
      return findRecord(currentIndex);
    }

    function renderSummary() {
      const summary = appState.summary;
      byId('manifestPath').textContent = appState.manifest_path;
      byId('summary').innerHTML = `
        <div class="summary-card"><strong>${summary.total}</strong><br>Total</div>
        <div class="summary-card"><strong>${summary.complete}</strong><br>Complete</div>
        <div class="summary-card"><strong>${summary.partial}</strong><br>Partial</div>
        <div class="summary-card"><strong>${summary.empty}</strong><br>Empty</div>
      `;
    }

    function renderRecordList() {
      const visibleRecords = getVisibleRecords();
      if (visibleRecords.length === 0) {
        byId('recordList').innerHTML = '<div class="hint">No records match the current filters.</div>';
        return;
      }
      byId('recordList').innerHTML = visibleRecords.map((record) => `
        <div class="record-item ${record.index === currentIndex ? 'active' : ''}" data-index="${record.index}">
          <div class="record-item-top">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="status-dot status-${record.status}"></span>
              <strong>#${record.index + 1}</strong>
            </div>
            <span class="pill">${record.semantic_family}</span>
          </div>
          <div style="font-size:13px; line-height:1.4;">${record.sample_id}</div>
          <div class="hint" style="margin-top:6px;">ep ${record.episode_id} - frame ${record.frame_index}</div>
        </div>
      `).join('');
      byId('recordList').querySelectorAll('.record-item').forEach((node) => {
        node.addEventListener('click', async () => {
          await navigateTo(Number(node.dataset.index));
        });
      });
    }
"""

INDEX_HTML += """
    function makeMetaPills(record) {
      return `
        <span class="pill">sample ${record.sample_id}</span>
        <span class="pill">${record.semantic_family}</span>
        <span class="pill">${record.phase_hint}</span>
        <span class="pill">episode ${record.episode_id}</span>
        <span class="pill">frame ${record.frame_index}</span>
        <span class="pill">status ${record.status}</span>
      `;
    }

    function renderMain() {
      const record = getCurrentRecord();
      if (!record) {
        byId('mainContent').innerHTML = '<div class="card empty-state">No record selected.</div>';
        return;
      }

      const annotation = record.annotation || {};
      const draft = record.draft_annotation || {};
      const primitiveOptions = appState.options.primitive.map((value) =>
        `<option value="${value}" ${annotation.primitive === value ? 'selected' : ''}>${value}</option>`
      ).join('');
      const relationOptions = appState.options.relation.map((value) =>
        `<option value="${value}" ${annotation.relation === value ? 'selected' : ''}>${value}</option>`
      ).join('');
      const confidenceOptions = [''].concat(appState.options.confidence).map((value) => {
        const label = value || 'Unspecified';
        return `<option value="${value}" ${((annotation.confidence || '') === value) ? 'selected' : ''}>${label}</option>`;
      }).join('');
      const targetSuggestions = appState.options.target_suggestions.map((value) => `<option value="${value}"></option>`).join('');

      byId('mainContent').innerHTML = `
        <div class="card">
          <div class="instruction">${record.instruction}</div>
          <div>${makeMetaPills(record)}</div>
        </div>
        <div class="card media-grid">
          <div class="media-card">
            <div class="media-label">Current frame</div>
            <img src="${record.image_url}" alt="Current frame">
          </div>
          <div class="media-card">
            <div class="media-label">Context strip</div>
            <img src="${record.context_image_url}" alt="Context strip">
          </div>
        </div>
        <div class="card">
          <div class="draft-box">
            <strong>Draft suggestion</strong><br>
            primitive=${draft.primitive || '-'} - target=${draft.target || '-'} - relation=${draft.relation || '-'}
          </div>
          <div class="form-grid" style="margin-top: 16px;">
            <label>
              Primitive
              <select id="primitiveInput">
                <option value="">Unspecified</option>
                ${primitiveOptions}
              </select>
            </label>
            <label>
              Relation
              <select id="relationInput">
                <option value="">Unspecified</option>
                ${relationOptions}
              </select>
            </label>
            <label class="wide">
              Target
              <input id="targetInput" type="text" list="targetSuggestions" value="${annotation.target || ''}" placeholder="bowl / basket / drawer_handle ...">
              <datalist id="targetSuggestions">${targetSuggestions}</datalist>
            </label>
            <label>
              Confidence
              <select id="confidenceInput">
                ${confidenceOptions}
              </select>
            </label>
            <label class="wide">
              Note
              <textarea id="noteInput" placeholder="Optional note for ambiguity, occlusion, or removal reason.">${annotation.note || ''}</textarea>
            </label>
          </div>
        </div>
      `;
    }

    function currentFormAnnotation() {
      return {
        primitive: byId('primitiveInput')?.value || '',
        target: byId('targetInput')?.value || '',
        relation: byId('relationInput')?.value || '',
        confidence: byId('confidenceInput')?.value || '',
        note: byId('noteInput')?.value || '',
      };
    }

    function isDirty() {
      const record = getCurrentRecord();
      if (!record || !byId('primitiveInput')) return false;
      const current = currentFormAnnotation();
      const existing = {
        primitive: record.annotation?.primitive || '',
        target: record.annotation?.target || '',
        relation: record.annotation?.relation || '',
        confidence: record.annotation?.confidence || '',
        note: record.annotation?.note || '',
      };
      return JSON.stringify(current) !== JSON.stringify(existing);
    }

    async function saveCurrent() {
      const record = getCurrentRecord();
      if (!record) return true;
      const response = await fetch(`/api/records/${record.index}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotation: currentFormAnnotation() }),
      });
      if (!response.ok) {
        alert('Failed to save annotation.');
        return false;
      }
      const payload = await response.json();
      appState.summary = payload.summary;
      appState.records[record.index] = payload.record;
      renderSummary();
      renderRecordList();
      renderMain();
      return true;
    }
"""

INDEX_HTML += """
    async function navigateTo(nextIndex) {
      if (nextIndex === currentIndex) return;
      if (isDirty()) {
        const saved = await saveCurrent();
        if (!saved) return;
      }
      currentIndex = nextIndex;
      renderRecordList();
      renderMain();
    }

    async function stepRecord(direction) {
      const visibleRecords = getVisibleRecords();
      const visibleIndices = visibleRecords.map((record) => record.index);
      const position = visibleIndices.indexOf(currentIndex);
      if (position === -1 && visibleIndices.length > 0) {
        await navigateTo(visibleIndices[0]);
        return;
      }
      const nextPosition = position + direction;
      if (nextPosition >= 0 && nextPosition < visibleIndices.length) {
        await navigateTo(visibleIndices[nextPosition]);
      }
    }

    async function nextEmptyRecord() {
      const visibleRecords = getVisibleRecords();
      const start = visibleRecords.findIndex((record) => record.index === currentIndex);
      for (let offset = 1; offset <= visibleRecords.length; offset += 1) {
        const record = visibleRecords[(start + offset) % visibleRecords.length];
        if (record.status === 'empty') {
          await navigateTo(record.index);
          return;
        }
      }
    }

    function applyDraft() {
      const record = getCurrentRecord();
      if (!record) return;
      const draft = record.draft_annotation || {};
      if (draft.primitive && byId('primitiveInput')) byId('primitiveInput').value = draft.primitive;
      if (draft.target && byId('targetInput')) byId('targetInput').value = draft.target;
      if (draft.relation && byId('relationInput')) byId('relationInput').value = draft.relation;
    }

    function clearForm() {
      if (byId('primitiveInput')) byId('primitiveInput').value = '';
      if (byId('targetInput')) byId('targetInput').value = '';
      if (byId('relationInput')) byId('relationInput').value = '';
      if (byId('confidenceInput')) byId('confidenceInput').value = '';
      if (byId('noteInput')) byId('noteInput').value = '';
    }

    async function initialize() {
      const response = await fetch('/api/state');
      appState = await response.json();
      renderSummary();
      renderRecordList();
      renderMain();

      byId('familyFilter').addEventListener('change', () => {
        const visibleRecords = getVisibleRecords();
        if (visibleRecords.length > 0 && !visibleRecords.some((record) => record.index === currentIndex)) {
          currentIndex = visibleRecords[0].index;
        }
        renderRecordList();
        renderMain();
      });
      byId('statusFilter').addEventListener('change', () => {
        const visibleRecords = getVisibleRecords();
        if (visibleRecords.length > 0 && !visibleRecords.some((record) => record.index === currentIndex)) {
          currentIndex = visibleRecords[0].index;
        }
        renderRecordList();
        renderMain();
      });
      byId('prevBtn').addEventListener('click', async () => stepRecord(-1));
      byId('nextBtn').addEventListener('click', async () => stepRecord(1));
      byId('nextEmptyBtn').addEventListener('click', nextEmptyRecord);
      byId('draftBtn').addEventListener('click', applyDraft);
      byId('saveBtn').addEventListener('click', saveCurrent);
      byId('saveNextBtn').addEventListener('click', async () => {
        const ok = await saveCurrent();
        if (ok) await stepRecord(1);
      });
      byId('clearBtn').addEventListener('click', clearForm);

      document.addEventListener('keydown', async (event) => {
        const isMac = navigator.platform.toUpperCase().includes('MAC');
        const saveCombo = isMac ? event.metaKey : event.ctrlKey;
        if (saveCombo && event.key.toLowerCase() === 's') {
          event.preventDefault();
          await saveCurrent();
        } else if (event.shiftKey && event.key === 'Enter') {
          event.preventDefault();
          const ok = await saveCurrent();
          if (ok) await stepRecord(1);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          await stepRecord(-1);
        } else if (event.key === 'ArrowRight') {
          event.preventDefault();
          await stepRecord(1);
        }
      });
    }

    initialize();
  </script>
</body>
</html>
"""


class AnnotationHandler(BaseHTTPRequestHandler):
    store: ManifestStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(self.store.api_state())
            return
        if parsed.path == "/api/asset":
            params = parse_qs(parsed.query)
            raw_path = params.get("path", [""])[0]
            self._send_asset(raw_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/records/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        index_text = parsed.path.rsplit("/", 1)[-1]
        try:
            index = int(index_text)
            if index < 0 or index >= len(self.store.records):
                raise ValueError
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid record index.")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        self.store.update_record(index=index, annotation=payload.get("annotation", {}))
        response = {
            "record": self.store.api_state()["records"][index],
            "summary": self.store.summary(),
        }
        self._send_json(response)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        log_progress(format % args)

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, raw_path: str) -> None:
        resolved = self.store.resolve_asset_path(raw_path)
        if resolved is None or not resolved.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Asset not found: {raw_path}")
            return
        content = resolved.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(resolved))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small local browser UI for pi05_word manual annotation.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/candidates.jsonl"),
        help="Path to the JSONL manifest produced by pi05_word_export_annotation_candidates.py",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path-map-from", default="/home")
    parser.add_argument("--path-map-to", default=None, help="Optional local filesystem prefix for remote /home paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.path_map_to is None:
        path_map_to = manifest_path.anchor + "home" if manifest_path.anchor else None
    else:
        path_map_to = args.path_map_to

    store = ManifestStore(
        manifest_path=manifest_path,
        path_map_from=args.path_map_from,
        path_map_to=path_map_to,
    )
    AnnotationHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    log_progress(f"Serving {manifest_path} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_progress("Shutting down annotation tool.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
