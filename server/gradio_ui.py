"""
SQLab — Gradio Web UI.

Three-tab interface:
1. Interactive Playground — type SQL, see results
2. Demo Traces — pre-recorded model runs
3. Leaderboard — model comparison + heatmap
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

import gradio as gr

from sqlab.models import DBSreAction
from sqlab.server.tasks import TASK_REGISTRY

# ── Results loading ──────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"

def _load_all_results() -> dict:
    """Load all result JSON files from the results directory."""
    results = {}
    if not RESULTS_DIR.exists():
        return results
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            model_name = data.get("model", f.stem)
            results[model_name] = data
        except Exception:
            pass
    return results


def _model_display_name(model: str) -> str:
    """Shorten model names for display."""
    names = {
        "devstral-small-2:latest": "Devstral 15B",
        "qwen2.5-coder:7b": "Qwen2.5-Coder 7B",
        "qwen2.5-coder:14b": "Qwen2.5-Coder 14B",
        "deepseek-coder-v2:16b": "DeepSeek-Coder-V2 16B",
        "phi4:14b": "Phi-4 14B",
        "qwen3:8b": "Qwen3 8B",
    }
    return names.get(model, model)


# ── Custom CSS (bench-mark.org inspired) ─────────────────────────────

CUSTOM_CSS = """
/* ══════════════════════════════════════════════════════════════
   VibeCheck-inspired design system
   ──────────────────────────────────────────────────────────────
   Color blocks:
     Yellow  #fde047  — headers, primary actions, table headers
     Orange  #fed7aa  — interactive panels (playground controls)
     Pink    #fecdd3  — alerts, errors, hard badges
     Green   #d9f99d  — success, easy badges, grader earned
     Blue    #bfdbfe  — metrics, info panels, traces
     Lime    #ecfccb  — command history, trace bg
     White   #fff     — text inputs, code output bg
   All text: #000 for contrast. Borders: 2-3px solid #000.
   Background: cyan dot pattern.
   ══════════════════════════════════════════════════════════════ */

/* ── Page background — cyan dot pattern ── */
.gradio-container {
    background-color: #e0f7fa !important;
    background-image: radial-gradient(circle, #00bcd4 0.8px, transparent 0.8px) !important;
    background-size: 16px 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #000 !important;
}

/* ── Global text — always black for contrast ── */
.gradio-container, .gradio-container p, .gradio-container span,
.gradio-container div, .gradio-container label, .gradio-container h1,
.gradio-container h2, .gradio-container h3, .gradio-container h4,
.gradio-container strong, .gradio-container b, .gradio-container td,
.gradio-container th, .gradio-container li, .gradio-container summary,
.gradio-container details, .gradio-container a,
.gradio-container .tabitem *, .gradio-container [role="tabpanel"] *,
.prose, .prose *, .html-container, .html-container * {
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
}
/* Exception: terminal output keeps green text — high specificity to beat global black */
.sql-output, .sql-output *,
.gradio-container .sql-output, .gradio-container .sql-output *,
.gradio-container .tabitem .sql-output, .gradio-container .tabitem .sql-output *,
.gradio-container [role="tabpanel"] .sql-output, .gradio-container [role="tabpanel"] .sql-output *,
.html-container .sql-output, .html-container .sql-output * {
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
    background: #0a1628 !important;
}
/* Exception: inputs keep their own color */
textarea, input[type="text"] {
    -webkit-text-fill-color: #000 !important;
}

/* ── Tabs wrapper ── */
.tabs {
    background: transparent !important;
    border: none !important;
    overflow: visible !important;
    background-image: none !important;
    margin-top: 12px !important;
}

/* ── Tab wrapper/container — kill the bottom line and fixed height ── */
.tabs > div:first-child,
div[class*="tab-wrapper"],
div[class*="tab-container"] {
    height: auto !important;
    padding-bottom: 12px !important;
    overflow: visible !important;
}
div[class*="tab-container"]::after,
.tabs > div:first-child > div::after {
    display: none !important;
    background: transparent !important;
    height: 0 !important;
}

/* ── Tab nav container — center the buttons with gaps ── */
.tabs > div:first-child > div,
div[class*="tab-container"] {
    display: flex !important;
    gap: 14px !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
}

/* ── Tab buttons — floating colored cards with neon hover ── */
.tabs button,
.tabs > div:first-child button,
div[class*="tab-container"] button {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-weight: 900 !important;
    font-size: 15px !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    border: 3px solid #000 !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    text-align: center !important;
    white-space: nowrap !important;
    height: auto !important;
    position: relative !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    box-shadow: 4px 4px 0 #000 !important;
    background: #fed7aa !important;
}
/* Individual button colors */
.tabs button:nth-child(1) { background: #ffe0b2 !important; }
.tabs button:nth-child(2) { background: #d1fae5 !important; }
.tabs button:nth-child(3) { background: #bfdbfe !important; }
.tabs button:nth-child(4) { background: #fde047 !important; }
.tabs button:nth-child(5) { background: #fecdd3 !important; }
/* Hover — neon glow + lift */
.tabs button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 16px #fde047, 4px 4px 0 #000 !important;
    background-color: inherit !important;
}
/* Selected — pressed down, no glow, just flat */
.tabs button.selected,
.tabs button[class*="selected"] {
    transform: translateY(1px) !important;
    box-shadow: 2px 2px 0 #000 !important;
}
/* Kill the ::after underline on selected */
.tabs button.selected::after,
.tabs button[class*="selected"]::after {
    display: none !important;
    height: 0 !important;
    background: transparent !important;
}
/* Selected tabs — darker shade to show active state */
.tabs button.selected:nth-child(1) { background: #ffb74d !important; }
.tabs button.selected:nth-child(2) { background: #a7f3d0 !important; }
.tabs button.selected:nth-child(3) { background: #93c5fd !important; }
.tabs button.selected:nth-child(4) { background: #fbbf24 !important; }
.tabs button.selected:nth-child(5) { background: #fca5a5 !important; }

/* ── Structural resets — no borders on layout wrappers ── */
.form, .row, .column, .gap, .contain,
.html-container, .prose {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
/* Tab content panels — own white card (shadow matches header 6px) */
.tabitem, .tab-content, [role="tabpanel"] {
    border: 3px solid #000 !important;
    border-radius: 8px !important;
    background: #fff !important;
    background-image: none !important;
    padding: 20px !important;
    box-shadow: 6px 6px 0 #000 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* ── Block containers — ORANGE for interactive panels ── */
.block {
    border: 2px solid #000 !important;
    border-radius: 4px !important;
    background: #fed7aa !important;
}

/* Dropdown & textbox wrapper blocks — orange */
.block:has(select), .block:has(textarea), .block:has(input) {
    background: #fed7aa !important;
}

/* ── HTML blocks — transparent (they render their own styled content) ── */
.block:has(.html-container) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Labels — use CSS variables to override Gradio's scoped Svelte styles ── */
.gradio-container {
    --section-header-text-size: 14px !important;
    --section-header-text-weight: 900 !important;
    --block-label-text-size: 14px !important;
    --block-label-text-weight: 900 !important;
    --block-label-text-color: #000 !important;
    --body-text-color: #000 !important;
    --body-text-color-subdued: #000 !important;
}
label, .label-text, span[data-testid="block-label"],
.gradio-container label, .gradio-container .label-text,
.gradio-container span[data-testid="block-label"],
.gradio-container .block > span:first-child,
.gradio-container .wrap > label,
.block label span,
[class*="svelte"] > span {
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    font-weight: 900 !important;
    font-size: 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    text-shadow: 1px 1px 0 #fb923c !important;
}

/* ── Section headings (h2/h3 inside tabs) — neon shadow ── */
h2, .gradio-container h2,
.gradio-container .tabitem h2,
.gradio-container [role="tabpanel"] h2 {
    font-size: 22px !important;
    font-weight: 900 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    text-shadow: 1.5px 1.5px 0 #a78bfa !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 8px !important;
}
h3, .gradio-container h3,
.gradio-container .tabitem h3,
.gradio-container [role="tabpanel"] h3 {
    font-size: 18px !important;
    font-weight: 900 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    text-shadow: 1.5px 1.5px 0 #60a5fa !important;
}

/* ── Text inputs — white bg for writing ── */
textarea, input[type="text"] {
    border: 2px solid #000 !important;
    border-radius: 4px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
    background: #fff !important;
    color: #000 !important;
    font-size: 13px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
}

/* ── Dropdowns — white bg, black text, styled list ── */
.wrap .wrap-inner, [data-testid="dropdown"],
.secondary-wrap, .dropdown-container {
    border: 2px solid #000 !important;
    border-radius: 4px !important;
    background: #fff !important;
    color: #000 !important;
}
/* Dropdown input text */
.wrap .wrap-inner input,
.wrap .wrap-inner span,
.wrap .secondary-wrap,
input[data-testid="textbox"],
.single-select {
    color: #000 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
/* Dropdown placeholder */
.wrap .wrap-inner input::placeholder {
    color: #6b7280 !important;
    font-weight: 500 !important;
}
/* Dropdown open list */
.dropdown-content, ul[role="listbox"], .options {
    background: #fff !important;
    border: 3px solid #000 !important;
    border-radius: 4px !important;
    box-shadow: 4px 4px 0 #000 !important;
}
/* Dropdown list items */
ul[role="listbox"] li, .dropdown-content li,
.options li, .option {
    color: #000 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid #e5e7eb !important;
}
ul[role="listbox"] li:hover, .dropdown-content li:hover,
.options li:hover, .option:hover {
    background: #fde047 !important;
    color: #000 !important;
}
ul[role="listbox"] li.selected, .dropdown-content li.selected,
.options li.selected, .option.selected {
    background: #fed7aa !important;
    color: #000 !important;
    font-weight: 800 !important;
}

/* ── Header — YELLOW card ── */
.gym-header {
    text-align: center;
    padding: 24px 0 16px 0;
    background: #fde047;
    border: 3px solid #000;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 6px 6px 0 #000;
}
.gym-header h1 {
    font-size: 42px;
    font-weight: 900;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    margin: 0;
    text-shadow: 1.5px 1.5px 0 #f472b6;
    letter-spacing: -0.02em;
}
.gym-header p {
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    font-size: 15px;
    font-weight: 700;
    margin: 8px 0 0 0;
    text-shadow: none;
}

/* ── Accent bar — removed ── */
.accent-bar {
    display: none !important;
}

/* ── Hide Gradio footer ── */
footer, .gradio-container > footer,
div[class*="footer"], .built-with {
    display: none !important;
}

/* ── Uniform width: pin the outermost container so all tabs match ── */
.gradio-container > .main,
.gradio-container > .main > .wrap {
    max-width: 1200px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
}
.gym-header {
    width: 100% !important;
    box-sizing: border-box !important;
}
.tabitem, .tab-content, [role="tabpanel"] {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

/* ── Playground subblocks — CSS :has() with data-pg markers ── */
/* (Gradio 6 bug: elem_id/elem_classes don't reach DOM for layout components) */
.gr-group:has([data-pg]),
.gr-group:has([data-pg]) > .styler {
    border: 2px solid #000 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin-bottom: 14px !important;
    box-shadow: 3px 3px 0 #000 !important;
}

/* ── Block 1: Task Selection — green (outer darker, inner lighter) ── */
.gr-group:has([data-pg="task-select"]) { background: #6ee7b7 !important; }
.gr-group:has([data-pg="task-select"]) > .styler { background: #a7f3d0 !important; }

/* ── Block 2: SQL Workflow — rose/pink ── */
.gr-group:has([data-pg="sql-workflow"]) { background: #fb7185 !important; }
.gr-group:has([data-pg="sql-workflow"]) > .styler { background: #ffe4e6 !important; }
.gr-group:has([data-pg="sql-workflow"]) .metric-card {
    background: #fff !important;
    border: 2px solid #000 !important;
}
/* All buttons inside sql-workflow: gray */
.gr-group:has([data-pg="sql-workflow"]) button,
.gr-group:has([data-pg="sql-workflow"]) .primary-btn,
.gr-group:has([data-pg="sql-workflow"]) .secondary,
.gr-group:has([data-pg="sql-workflow"]) .hint-pill {
    background: #e4e4e7 !important;
}
/* All form containers, inputs, wraps inside sql-workflow: white */
.gr-group:has([data-pg="sql-workflow"]) input,
.gr-group:has([data-pg="sql-workflow"]) textarea,
.gr-group:has([data-pg="sql-workflow"]) .wrap-inner,
.gr-group:has([data-pg="sql-workflow"]) .wrap,
.gr-group:has([data-pg="sql-workflow"]) .block,
.gr-group:has([data-pg="sql-workflow"]) .checkbox-container,
.gr-group:has([data-pg="sql-workflow"]) label {
    background: transparent !important;
}
.gr-group:has([data-pg="sql-workflow"]) input,
.gr-group:has([data-pg="sql-workflow"]) textarea,
.gr-group:has([data-pg="sql-workflow"]) .wrap-inner {
    background: #fff !important;
}

/* ── Block 3: Grader — lime green (outer darker, inner lighter) ── */
.gr-group:has([data-pg="grader"]) { background: #a3e635 !important; }
.gr-group:has([data-pg="grader"]) > .styler { background: #d9f99d !important; }
/* Task Selection — blocks inherit mint bg instead of generic peach */
.gr-group:has([data-pg="task-select"]) .block {
    background: transparent !important;
}
/* Align dropdown + Reset button vertically in Task Selection row */
.gr-group:has([data-pg="task-select"]) .row {
    align-items: flex-end !important;
    gap: 12px !important;
}
.gr-group:has([data-pg="task-select"]) .row > .block {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}
.gr-group:has([data-pg="task-select"]) .row button {
    min-height: 42px !important;
    margin-bottom: 1px !important;
}
/* Align SQL input + Execute with Step/Reward/Status cards */
.gr-group:has([data-pg="sql-workflow"]) > .styler > .row {
    align-items: flex-start !important;
}
.playground-subblock-title {
    font-weight: 900;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
    color: #000;
    text-shadow: 2px 2px 0 #a78bfa;
}
/* Hint sub-subblock (nested inside sql-workflow — exclude outer group) */
.gr-group:has(.hint-note):not(:has([data-pg="sql-workflow"])),
.gr-group:has(.hint-note):not(:has([data-pg="sql-workflow"])) > .styler {
    border: 2px dashed #9ca3af !important;
    border-radius: 4px !important;
    padding: 10px !important;
    margin-top: 8px !important;
    background: #fef3c7 !important;
    box-shadow: none !important;
}
/* REPL observation log (nested inside sql-workflow — exclude outer group) */
.gr-group:has([data-pg="repl"]):not(:has([data-pg="sql-workflow"])),
.gr-group:has([data-pg="repl"]):not(:has([data-pg="sql-workflow"])) > .styler {
    border: 2px solid #000 !important;
    border-radius: 4px !important;
    background: #0a1628 !important;
    padding: 0 !important;
    margin-top: 10px !important;
    box-shadow: 3px 3px 0 #000 !important;
}
.gr-group:has([data-pg="repl"]):not(:has([data-pg="sql-workflow"])) .playground-subblock-title {
    color: #93c5fd !important;
    -webkit-text-fill-color: #93c5fd !important;
    padding: 10px 14px 4px 14px;
    font-size: 14px !important;
    font-weight: 900 !important;
    letter-spacing: 0.08em !important;
    text-shadow: 0 0 8px rgba(96, 165, 250, 0.4);
}
.repl-log {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 11px !important;
    background: #0a1628 !important;
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
    padding: 10px 12px !important;
    white-space: pre-wrap !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    border: none !important;
    box-shadow: none !important;
}
/* Override global black text inside REPL — base green, class overrides for prompt/cmd/error */
.repl-log, .gradio-container .repl-log,
.gradio-container .tabitem .repl-log,
.gradio-container [role="tabpanel"] .repl-log {
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
}
.gradio-container .tabitem .repl-log .rp,
.gradio-container [role="tabpanel"] .repl-log .rp,
.repl-log .rp { color: #60a5fa !important; -webkit-text-fill-color: #60a5fa !important; }
.gradio-container .tabitem .repl-log .rc,
.gradio-container [role="tabpanel"] .repl-log .rc,
.repl-log .rc { color: #fde047 !important; -webkit-text-fill-color: #fde047 !important; }
.gradio-container .tabitem .repl-log .re,
.gradio-container [role="tabpanel"] .repl-log .re,
.repl-log .re { color: #f87171 !important; -webkit-text-fill-color: #f87171 !important; }
.gradio-container .tabitem .repl-log .rr,
.gradio-container [role="tabpanel"] .repl-log .rr,
.repl-log .rr { font-size: 10px; }
.repl-log .rr.pos { color: #4ade80 !important; -webkit-text-fill-color: #4ade80 !important; }
.repl-log .rr.neg { color: #f87171 !important; -webkit-text-fill-color: #f87171 !important; }
.repl-log .rr.zero { color: #94a3b8 !important; -webkit-text-fill-color: #94a3b8 !important; }

/* ── Alert panel — PINK/RED card ── */
.alert-panel {
    border: 3px solid #000;
    border-left: 6px solid #dc2626;
    background: #fecdd3;
    padding: 12px 16px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
    color: #000;
    white-space: pre-wrap;
}

/* ── SQL output — terminal theme: dark navy bg, green text ── */
.sql-output {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    background: #0a1628 !important;
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
    padding: 12px;
    border-radius: 4px;
    border: 3px solid #000;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.3);
}
.gradio-container .sql-output, .gradio-container .sql-output * { color: #4ade80 !important; -webkit-text-fill-color: #4ade80 !important; }

/* ── Error output — PINK ── */
.sql-error {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    background: #fecdd3;
    color: #000 !important;
    padding: 12px;
    border-radius: 4px;
    white-space: pre-wrap;
    border: 3px solid #000;
}

/* ── Metrics cards — BLUE ── */
.metric-card {
    background: #bfdbfe;
    border: 2px solid #000;
    border-radius: 4px;
    padding: 12px;
    text-align: center;
}
.metric-value {
    font-size: 24px;
    font-weight: 900;
    color: #000 !important;
}
.metric-label {
    font-size: 11px;
    color: #000 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
}

/* ── Difficulty badges — colored with black border ── */
.badge-easy { background: #d9f99d; color: #000; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: 800; border: 2px solid #000; display: inline-block; }
.badge-medium { background: #fde047; color: #000; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: 800; border: 2px solid #000; display: inline-block; }
.badge-hard { background: #fecdd3; color: #000; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: 800; border: 2px solid #000; display: inline-block; }

/* ── Step cards in traces ── */
.step-card {
    background: #fff;
    border: 2px solid #000;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 6px solid #d1d5db;
    font-size: 13px;
    color: #000;
}
.step-card.positive { border-left-color: #16a34a; background: #d9f99d; }
.step-card.negative { border-left-color: #dc2626; background: #fecdd3; }
.step-card .step-num { font-weight: 900; color: #000; margin-right: 8px; }
.step-card .step-cmd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #000; }
.step-card .step-reward { float: right; font-weight: 800; }
.step-card .step-reward.pos { color: #166534; }
.step-card .step-reward.neg { color: #991b1b; }

/* ── Force black text on ALL table elements (override Gradio grays) ── */
.gradio-container table,
.gradio-container table th,
.gradio-container table td,
.gradio-container table tr,
.gradio-container table thead,
.gradio-container table tbody,
.gradio-container .prose table,
.gradio-container .prose th,
.gradio-container .prose td {
    color: #000 !important;
}

/* ── Heatmap table — YELLOW headers ── */
.heatmap-table { border-collapse: collapse; width: 100%; font-size: 12px; border: 3px solid #000; box-shadow: 4px 4px 0 #000; border-radius: 4px; overflow: hidden; }
.heatmap-table th { padding: 8px 10px; text-align: center; font-weight: 900; color: #000 !important; border: 2px solid #000; background: #fde047; text-transform: uppercase; letter-spacing: 0.02em; }
.heatmap-table td { padding: 6px 8px; text-align: center; font-weight: 700; border: 2px solid #000; color: #000 !important; }
.heatmap-table tr:nth-child(even) { background: #fef9c3; }

/* ── Grader breakdown — GREEN sections ── */
.breakdown-section { margin-bottom: 12px; padding: 10px; background: #ecfccb; border: 2px solid #000; border-radius: 4px; }
.breakdown-title { font-weight: 900; font-size: 14px; color: #000; margin-bottom: 6px; background: #fde047; display: inline-block; padding: 2px 10px; border-radius: 2px; border: 1px solid #000; }
.checkpoint { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; color: #000; }
.checkpoint-name { color: #000; font-weight: 600; }
.checkpoint-value { font-weight: 800; }
.checkpoint-value.earned { color: #166534; }
.checkpoint-value.missed { color: #991b1b; }

/* ── Buttons — colored bg, black border, pop up on hover (no glow at rest) ── */
.primary-btn,
button.primary, button[class*="primary"],
.gradio-container button.primary,
.gradio-container button[class*="primary"] {
    background: #fde047 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    border: 3px solid #000 !important;
    border-radius: 8px !important;
    font-weight: 900 !important;
    font-size: 13px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    box-shadow: 3px 3px 0 #000 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}
.primary-btn:hover,
button.primary:hover, button[class*="primary"]:hover,
.gradio-container button.primary:hover,
.gradio-container button[class*="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 5px 5px 0 #000 !important;
}
button.secondary, button[class*="secondary"],
.gradio-container button.secondary,
.gradio-container button[class*="secondary"] {
    background: #e5e7eb !important;
    color: #000 !important;
    border: 2px solid #000 !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    box-shadow: 2px 2px 0 #000 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
button.secondary:hover, button[class*="secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 4px 4px 0 #000 !important;
}

/* ── Resolved badge ── */
.resolved-yes { background: #d9f99d; color: #000; padding: 4px 12px; border-radius: 4px; font-weight: 900; border: 2px solid #000; }
.resolved-no { background: #fecdd3; color: #000; padding: 4px 12px; border-radius: 4px; font-weight: 900; border: 2px solid #000; }

/* ── Leaderboard table — YELLOW headers, BLUE rank-1, thick borders ── */
.leaderboard-table { border-collapse: collapse; width: 100%; font-size: 13px; border: 3px solid #000; box-shadow: 4px 4px 0 #000; border-radius: 4px; overflow: hidden; }
.leaderboard-table th { padding: 12px 14px; text-align: left; font-weight: 900; color: #000 !important; border: 2px solid #000; background: #fde047; font-size: 14px; text-transform: uppercase; letter-spacing: 0.03em; }
.leaderboard-table td { padding: 10px 14px; border: 2px solid #000; color: #000 !important; font-weight: 700; }
.leaderboard-table tr:hover { background: #fef9c3; }
.leaderboard-table .score-cell { font-weight: 900; color: #000 !important; }
.leaderboard-table .rank-1 { background: #bfdbfe !important; }

/* ── Task descriptions accordion — ORANGE bg ── */
.task-accordion {
    margin-bottom: 8px;
    background: #fed7aa;
    border: 2px solid #000;
    border-radius: 4px;
}
.task-accordion summary {
    padding: 12px 16px;
    cursor: pointer;
    font-weight: 800;
    color: #000;
}
.task-accordion .acc-body {
    padding: 0 16px 12px 16px;
    font-size: 13px;
    color: #000;
}

/* ── Environment overview — BLUE card ── */
.env-overview {
    background: #bfdbfe;
    border: 3px solid #000;
    border-radius: 4px;
    box-shadow: 3px 3px 0 #000;
    padding: 16px;
    margin-bottom: 12px;
    color: #000;
}
.env-overview h3 { font-weight: 900; margin: 0 0 8px 0; }
.env-overview p { margin: 4px 0; font-weight: 600; }

/* ── Decision tree — guided path buttons ── */
.path-prompt {
    font-size: 13px; font-weight: 800; color: #000;
    background: #bfdbfe; border: 2px solid #000; border-radius: 4px;
    padding: 6px 12px; margin-bottom: 4px;
    display: flex; align-items: center; gap: 8px;
}
.path-step-badge {
    background: #fde047; border: 2px solid #000; border-radius: 4px;
    padding: 2px 8px; font-size: 11px; font-weight: 900;
    white-space: nowrap;
}
.path-done {
    background: #d9f99d !important;
    border-color: #16a34a !important;
}
.path-fail {
    background: #fecdd3 !important;
    border-color: #dc2626 !important;
}
.hint-pill,
.gradio-container .hint-pill,
button.hint-pill {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 10px !important; font-weight: 600 !important; color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    background: #e4e4e7 !important; border: 2px solid #000 !important; border-radius: 6px !important;
    padding: 4px 8px !important; cursor: pointer !important;
    transition: all 0.12s ease !important;
    box-shadow: 2px 2px 0 #000 !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    text-transform: none !important; letter-spacing: 0 !important;
    min-height: 0 !important; line-height: 1.3 !important;
    max-width: 100% !important; display: block !important; text-align: left !important;
    margin-bottom: 4px !important;
}
.hint-pill:hover {
    transform: translateY(-1px) !important; box-shadow: 3px 3px 0 #000 !important;
    overflow-x: auto !important; text-overflow: unset !important;
}
/* Reveal: correct (primary variant) = green bg, wrong (stop variant) = red bg */
button.hint-pill[class*="primary"],
.gradio-container button.hint-pill[class*="primary"] {
    background: #d1fae5 !important;
    border-color: #16a34a !important;
    border-left: 5px solid #16a34a !important;
}
button.hint-pill[class*="stop"],
.gradio-container button.hint-pill[class*="stop"] {
    background: #fecdd3 !important;
    border-color: #dc2626 !important;
    border-left: 5px solid #dc2626 !important;
}
/* Reveal checkbox styling */
.reveal-check { min-height: 0 !important; }
.reveal-check label { font-size: 11px !important; text-shadow: none !important; text-transform: none !important; letter-spacing: 0 !important; }
.reveal-check input[type="checkbox"] {
    accent-color: #000 !important;
    width: 16px !important;
    height: 16px !important;
}
.reveal-check input[type="checkbox"]:checked {
    background: #000 !important;
    border-color: #000 !important;
}
/* Hint note */
.hint-note {
    font-size: 10px; color: #6b7280 !important; -webkit-text-fill-color: #6b7280 !important;
    font-style: italic; margin-top: 2px; font-weight: 500;
    text-shadow: none !important; letter-spacing: 0 !important; text-transform: none !important;
}

/* ── Compact playground — reduce spacing so it fits without scrolling ── */
.gradio-container { padding-top: 0 !important; }
.gradio-container > .main { padding-top: 0 !important; }
.gym-header {
    padding: 6px 0 5px 0 !important;
    margin-bottom: 3px !important;
    margin-top: 0 !important;
}
.gym-header h1 { font-size: 30px !important; }
.gym-header p { font-size: 12px !important; margin-top: 2px !important; }

/* Smaller gaps between elements */
.gradio-container .gap { gap: 6px !important; }
.gradio-container .form { gap: 6px !important; }

/* Compact alert panel */
.alert-panel { padding: 8px 12px !important; font-size: 12px !important; }

/* Compact metric cards */
.metric-card { padding: 6px 8px !important; }
.metric-value { font-size: 18px !important; }
.metric-label { font-size: 10px !important; }

/* Fatal path banner */
.path-fatal {
    background: #7f1d1d !important;
    border-color: #dc2626 !important;
    color: #fecaca !important;
}
.path-fatal, .path-fatal * {
    color: #fecaca !important;
    -webkit-text-fill-color: #fecaca !important;
}
.path-fatal .path-step-badge {
    background: #dc2626 !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
}

/* Compact SQL output */
.sql-output { max-height: 200px !important; padding: 8px !important; font-size: 11px !important; }
.sql-error { padding: 8px !important; font-size: 11px !important; }

/* Compact step cards */
.step-card { padding: 6px 10px !important; margin-bottom: 4px !important; font-size: 12px !important; }

/* Smaller block padding */
.block { padding: 8px !important; }
.block:has(.html-container) { padding: 0 !important; }

/* Tab content less padding */
.tabitem, .tab-content, [role="tabpanel"] { padding: 12px !important; }
"""


# ── HTML builders ────────────────────────────────────────────────────

# ── Multi-step guided decision tree per task ─────────────────────────
# Each task has a list of steps. Each step has:
#   "prompt": what the user should do next
#   "correct": the right SQL command
#   "wrong": list of 2 wrong/dangerous alternatives
# User clicks correct → advance. Wrong → fail message.

TASK_PATHS = {
    # ══ EASY ═══════════════════════════════════════════════════════
    "task_1": [  # Missing Index — resolved when: index on (flight_id) exists
        {"prompt": "Investigate: Something is slow — where do you start?",
         "correct": "EXPLAIN ANALYZE SELECT * FROM bookings.ticket_flights WHERE flight_id = 1",
         "wrong": [("SELECT * FROM pg_stat_bgwriter", "mild"),
                    ("ALTER SYSTEM SET work_mem = '1GB'", "bad")]},
        {"prompt": "Identify: The plan shows a sequential scan. Why?",
         "correct": "SELECT indexname FROM pg_indexes WHERE tablename = 'ticket_flights' AND schemaname = 'bookings'",
         "wrong": [("SHOW shared_buffers", "mild"),
                    ("SELECT * FROM pg_stat_user_tables WHERE relname = 'bookings'", "mild")]},
        {"prompt": "Resolve: Create the missing index",
         "correct": "CREATE INDEX idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
         "wrong": [("ANALYZE bookings.ticket_flights", "bad"),
                    ("SET enable_seqscan = off", "bad")]},
    ],
    "task_2": [  # Stale Statistics — resolved when: ANALYZE ran within 5 min
        {"prompt": "Investigate: Queries returning wrong row estimates — what to check?",
         "correct": "EXPLAIN ANALYZE SELECT * FROM bookings.flights WHERE status = 'Delayed'",
         "wrong": [("SELECT * FROM pg_locks", "mild"),
                    ("SHOW max_connections", "mild")]},
        {"prompt": "Identify: Estimated vs actual rows differ wildly. Check stats freshness",
         "correct": "SELECT relname, n_live_tup, last_analyze FROM pg_stat_user_tables WHERE relname = 'flights'",
         "wrong": [("SELECT * FROM pg_stat_activity", "mild"),
                    ("SELECT indexname FROM pg_indexes WHERE tablename = 'flights'", "mild")]},
        {"prompt": "Resolve: Update the stale statistics",
         "correct": "ANALYZE bookings.flights",
         "wrong": [("REINDEX TABLE bookings.flights", "bad"),
                    ("SET default_statistics_target = 1000", "bad")]},
    ],
    "task_3": [  # Connection Exhaustion — resolved when: idle-in-tx < 5 AND timeout set
        {"prompt": "Investigate: New connections are being refused — what's happening?",
         "correct": "SELECT state, count(*) FROM pg_stat_activity GROUP BY state",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_locks", "mild")]},
        {"prompt": "Identify: Many connections in one state — which ones are the problem?",
         "correct": "SELECT pid, state, query_start FROM pg_stat_activity WHERE state = 'idle in transaction'",
         "wrong": [("ALTER SYSTEM SET max_connections = 500", "bad"),
                    ("SELECT * FROM pg_stat_user_tables", "mild")]},
        {"prompt": "Resolve: Free up the stuck connections",
         "correct": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction'",
         "wrong": [("ALTER SYSTEM SET max_connections = 500", "bad"),
                    ("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active'", "bad")]},
    ],
    "task_4": [  # Permission Error — resolved when: app_user has SELECT on ticket_flights
        {"prompt": "Investigate: A user can't access a table — check permissions",
         "correct": "SELECT grantee, privilege_type FROM information_schema.role_table_grants WHERE table_name = 'ticket_flights'",
         "wrong": [("SELECT * FROM pg_stat_activity", "mild"),
                    ("SHOW max_connections", "mild")]},
        {"prompt": "Identify: What role and privileges does the app use?",
         "correct": "SELECT rolname, rolsuper FROM pg_roles WHERE rolname = 'app_user'",
         "wrong": [("ALTER USER app_user WITH SUPERUSER", "fatal"),
                    ("SELECT * FROM pg_locks", "mild")]},
        {"prompt": "Resolve: Grant the minimum required access",
         "correct": "GRANT SELECT ON bookings.ticket_flights TO app_user",
         "wrong": [("ALTER USER app_user WITH SUPERUSER", "fatal"),
                    ("GRANT INSERT ON bookings.ticket_flights TO app_user", "bad")]},
    ],
    "task_5": [  # Sequence Exhaustion — resolved when: sequence >= max(flight_id)
        {"prompt": "Investigate: INSERTs are failing — check the sequence",
         "correct": "SELECT last_value FROM bookings.flights_flight_id_seq",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_activity", "mild")]},
        {"prompt": "Identify: Is the sequence out of sync with actual data?",
         "correct": "SELECT MAX(flight_id) FROM bookings.flights",
         "wrong": [("ALTER SEQUENCE bookings.flights_flight_id_seq RESTART WITH 1", "bad"),
                    ("SELECT * FROM pg_locks", "mild")]},
        {"prompt": "Resolve: Reset the sequence to the correct value",
         "correct": "SELECT setval('bookings.flights_flight_id_seq', (SELECT MAX(flight_id) FROM bookings.flights))",
         "wrong": [("ALTER SEQUENCE bookings.flights_flight_id_seq RESTART WITH 1", "bad"),
                    ("SELECT nextval('bookings.flights_flight_id_seq')", "bad")]},
    ],
    # ══ MEDIUM ═════════════════════════════════════════════════════
    "task_6": [  # Bad Config — resolved when: work_mem >= 1MB AND eff_cache >= 512MB in pg_file_settings
        {"prompt": "Investigate: Queries are slow — check server configuration",
         "correct": "SELECT name, setting, unit FROM pg_settings WHERE name IN ('work_mem', 'effective_cache_size')",
         "wrong": [("SELECT * FROM pg_stat_activity", "mild"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Identify: Which parameter looks wrong?",
         "correct": "SHOW work_mem",
         "wrong": [("SET work_mem = '64kB'", "bad"),
                    ("SELECT * FROM pg_locks", "mild")]},
        {"prompt": "Resolve: Set the parameter to a reasonable value",
         "correct": "ALTER SYSTEM SET work_mem = '256MB'",
         "wrong": [("SET work_mem = '256MB'", "bad"),
                    ("ALTER SYSTEM SET maintenance_work_mem = '8kB'", "bad")]},
        {"prompt": "Finalize: Make the change take effect",
         "correct": "SELECT pg_reload_conf()",
         "wrong": [("SELECT pg_terminate_backend(pg_backend_pid())", "bad"),
                    ("ALTER SYSTEM RESET ALL", "fatal")]},
    ],
    "task_7": [  # Lock Contention — resolved when: blocker PID gone
        {"prompt": "Investigate: Queries are hanging — check for waits",
         "correct": "SELECT pid, wait_event_type, wait_event, query FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
         "wrong": [("LOCK TABLE bookings.flights IN EXCLUSIVE MODE", "fatal"),
                    ("SHOW deadlock_timeout", "mild")]},
        {"prompt": "Identify: Who is blocking whom?",
         "correct": "SELECT blocked.pid, blocking.pid AS blocker FROM pg_locks blocked JOIN pg_locks blocking ON blocked.locktype = blocking.locktype WHERE NOT blocked.granted",
         "wrong": [("ALTER SYSTEM SET deadlock_timeout = '10s'", "bad"),
                    ("SELECT * FROM pg_stat_user_tables", "mild")]},
        {"prompt": "Resolve: Remove the blocking session",
         "correct": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
         "wrong": [("LOCK TABLE bookings.flights IN EXCLUSIVE MODE", "fatal"),
                    ("ALTER SYSTEM SET lock_timeout = '0'", "bad")]},
    ],
    "task_8": [  # Table Bloat — resolved when: blocker PID gone AND dead tuples < 50%
        {"prompt": "Investigate: Table performance degraded — check table health",
         "correct": "SELECT relname, n_dead_tup, n_live_tup FROM pg_stat_user_tables ORDER BY n_dead_tup DESC LIMIT 5",
         "wrong": [("SELECT * FROM pg_locks", "mild"),
                    ("SHOW work_mem", "mild")]},
        {"prompt": "Identify: Is something blocking autovacuum? Check for long transactions",
         "correct": "SELECT pid, state, age(now(), xact_start), query FROM pg_stat_activity WHERE state != 'idle' ORDER BY xact_start LIMIT 10",
         "wrong": [("VACUUM FULL bookings.ticket_flights", "fatal"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Resolve: Clean up the bloated table",
         "correct": "VACUUM ANALYZE bookings.bookings",
         "wrong": [("VACUUM FULL bookings.bookings", "fatal"),
                    ("REINDEX TABLE bookings.bookings", "bad")]},
    ],
    "task_9": [  # Over-Indexing — resolved when: <=30% junk indexes remain
        {"prompt": "Investigate: Writes are slow — check index overhead",
         "correct": "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'ticket_flights' AND schemaname = 'bookings'",
         "wrong": [("CREATE INDEX idx_extra ON bookings.ticket_flights(amount)", "bad"),
                    ("SHOW work_mem", "mild")]},
        {"prompt": "Identify: Which indexes are actually being used?",
         "correct": "SELECT indexrelname, idx_scan FROM pg_stat_user_indexes WHERE relname = 'ticket_flights'",
         "wrong": [("CREATE INDEX idx_extra ON bookings.ticket_flights(amount)", "bad"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Resolve: Remove the unused junk indexes",
         "correct": "DROP INDEX IF EXISTS bookings.idx_tf_junk1",
         "wrong": [("CREATE INDEX idx_extra ON bookings.ticket_flights(amount)", "bad"),
                    ("DROP INDEX bookings.ticket_flights_pkey", "fatal")]},
    ],
    "task_10": [  # Index Bloat — resolved when: index size decreased
        {"prompt": "Investigate: Index scan latency is high — check index sizes",
         "correct": "SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid)) FROM pg_stat_user_indexes WHERE relname = 'ticket_flights'",
         "wrong": [("SELECT * FROM pg_stat_bgwriter", "mild"),
                    ("SHOW shared_buffers", "mild")]},
        {"prompt": "Identify: How bloated is the index compared to table?",
         "correct": "SELECT pg_size_pretty(pg_relation_size('bookings.idx_ticket_flights_flight')) AS idx_size",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_activity", "mild")]},
        {"prompt": "Resolve: Rebuild the bloated index without downtime",
         "correct": "REINDEX INDEX CONCURRENTLY bookings.idx_ticket_flights_flight",
         "wrong": [("ANALYZE bookings.ticket_flights", "bad"),
                    ("SET random_page_cost = 1", "bad")]},
    ],
    "task_11": [  # Wrong Index Column Order — resolved when: standalone index on (flight_id) exists
        {"prompt": "Investigate: Lookups by flight_id are slow — check the query plan",
         "correct": "EXPLAIN ANALYZE SELECT * FROM bookings.ticket_flights WHERE flight_id = 1",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Identify: There's a composite PK (ticket_no, flight_id) — flight_id is second",
         "correct": "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'ticket_flights' AND schemaname = 'bookings'",
         "wrong": [("ANALYZE bookings.ticket_flights", "mild"),
                    ("SELECT * FROM pg_stat_activity", "mild")]},
        {"prompt": "Resolve: Create a standalone index on the leading column",
         "correct": "CREATE INDEX ON bookings.ticket_flights(flight_id)",
         "wrong": [("ANALYZE bookings.ticket_flights", "bad"),
                    ("SET enable_seqscan = off", "bad")]},
    ],
    # ══ HARD ═══════════════════════════════════════════════════════
    "task_12": [  # Compound: Stale Stats + Missing Index
        {"prompt": "Investigate: Multiple issues reported — assess overall health",
         "correct": "EXPLAIN ANALYZE SELECT tf.ticket_no, f.status FROM bookings.ticket_flights tf JOIN bookings.flights f ON f.flight_id = tf.flight_id WHERE f.status = 'Delayed'",
         "wrong": [("SELECT * FROM pg_stat_bgwriter", "mild"),
                    ("SHOW max_connections", "mild")]},
        {"prompt": "Identify: Check if table statistics are current",
         "correct": "SELECT relname, last_analyze, n_dead_tup FROM pg_stat_user_tables WHERE schemaname = 'bookings' ORDER BY n_dead_tup DESC",
         "wrong": [("SELECT * FROM pg_stat_activity WHERE state = 'idle'", "mild"),
                    ("SHOW shared_buffers", "mild")]},
        {"prompt": "Resolve step 1: Fix stale statistics",
         "correct": "ANALYZE bookings.flights",
         "wrong": [("REINDEX TABLE bookings.flights", "bad"),
                    ("SET default_statistics_target = 1000", "bad")]},
        {"prompt": "Resolve step 2: Add the missing index",
         "correct": "CREATE INDEX ON bookings.ticket_flights(flight_id)",
         "wrong": [("ANALYZE bookings.ticket_flights", "bad"),
                    ("SET enable_seqscan = off", "bad")]},
    ],
    "task_13": [  # Compound: Lock + Bloat
        {"prompt": "Investigate: System is unresponsive — check for contention",
         "correct": "SELECT pid, wait_event_type, wait_event, query FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
         "wrong": [("ALTER SYSTEM SET deadlock_timeout = '10s'", "bad"),
                    ("SHOW work_mem", "mild")]},
        {"prompt": "Identify: Find the root blocker",
         "correct": "SELECT blocked.pid, blocking.pid AS blocker FROM pg_locks blocked JOIN pg_locks blocking ON blocked.locktype = blocking.locktype WHERE NOT blocked.granted",
         "wrong": [("ALTER SYSTEM SET deadlock_timeout = '10s'", "bad"),
                    ("LOCK TABLE bookings.flights IN EXCLUSIVE MODE", "fatal")]},
        {"prompt": "Resolve step 1: Terminate the blocking transaction",
         "correct": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND pid != pg_backend_pid()",
         "wrong": [("ALTER SYSTEM SET lock_timeout = '1s'", "bad"),
                    ("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active'", "bad")]},
        {"prompt": "Resolve step 2: Clean up dead tuples after the blocker is gone",
         "correct": "VACUUM ANALYZE bookings.bookings",
         "wrong": [("REINDEX TABLE bookings.bookings", "bad"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
    ],
    "task_14": [  # Deadlock Chain — resolved when: meta.deadlock_detected set by grader
        {"prompt": "Investigate: Deadlock detected — check active transactions",
         "correct": "SELECT pid, state, wait_event_type, query FROM pg_stat_activity WHERE datname = current_database() AND pid != pg_backend_pid()",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Identify: Look for the deadlock pattern in recent activity",
         "correct": "SELECT pid, wait_event_type, wait_event, query FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
         "wrong": [("ALTER SYSTEM SET deadlock_timeout = '1ms'", "bad"),
                    ("SELECT * FROM pg_stat_user_tables", "mild")]},
        {"prompt": "Resolve: Check conflicting locks between processes",
         "correct": "SELECT blocked.pid AS waiting, blocking.pid AS blocking FROM pg_locks blocked JOIN pg_locks blocking ON blocked.locktype = blocking.locktype AND blocked.relation = blocking.relation WHERE NOT blocked.granted AND blocked.pid != blocking.pid",
         "wrong": [("ALTER SYSTEM SET deadlock_timeout = '10s'", "bad"),
                    ("LOCK TABLE bookings.bookings IN EXCLUSIVE MODE", "fatal")]},
    ],
    "task_15": [  # Query Plan Flip — resolved when: random_page_cost <= 10
        {"prompt": "Investigate: Query suddenly slower — check if plan changed",
         "correct": "EXPLAIN ANALYZE SELECT * FROM bookings.ticket_flights WHERE flight_id = 1",
         "wrong": [("SELECT * FROM pg_stat_bgwriter", "mild"),
                    ("SHOW max_connections", "mild")]},
        {"prompt": "Identify: Plan uses Seq Scan when Index Scan expected — check planner settings",
         "correct": "SELECT name, setting FROM pg_settings WHERE name IN ('random_page_cost', 'seq_page_cost', 'enable_indexscan')",
         "wrong": [("SHOW work_mem", "mild"),
                    ("ANALYZE bookings.ticket_flights", "mild")]},
        {"prompt": "Resolve: Reset the bad planner parameter",
         "correct": "ALTER SYSTEM SET random_page_cost = 4",
         "wrong": [("SET random_page_cost = 4", "bad"),
                    ("ALTER SYSTEM SET work_mem = '256MB'", "bad")]},
        {"prompt": "Finalize: Apply the configuration change",
         "correct": "SELECT pg_reload_conf()",
         "wrong": [("ALTER SYSTEM RESET ALL", "fatal"),
                    ("SELECT pg_terminate_backend(pg_backend_pid())", "bad")]},
    ],
    "task_16": [  # Cascading Bloat — resolved when: blocker PID gone AND dead tuples reduced
        {"prompt": "Investigate: Dead tuples spiking across tables — check what's blocking vacuum",
         "correct": "SELECT pid, state, age(now(), xact_start) AS tx_age, query FROM pg_stat_activity WHERE state != 'idle' ORDER BY xact_start LIMIT 10",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Identify: Find the long-running transaction holding a snapshot",
         "correct": "SELECT pid, state, backend_xmin, query FROM pg_stat_activity WHERE backend_xmin IS NOT NULL AND pid != pg_backend_pid() ORDER BY age(backend_xmin) DESC LIMIT 5",
         "wrong": [("SELECT * FROM pg_locks", "mild"),
                    ("ALTER SYSTEM SET autovacuum_naptime = '1s'", "bad")]},
        {"prompt": "Resolve step 1: Terminate the snapshot-holding transaction",
         "correct": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state != 'idle' AND pid != pg_backend_pid() AND age(now(), xact_start) > interval '1 minute'",
         "wrong": [("ALTER SYSTEM SET autovacuum_naptime = '1s'", "bad"),
                    ("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active'", "bad")]},
        {"prompt": "Resolve step 2: Vacuum all affected tables",
         "correct": "VACUUM ANALYZE",
         "wrong": [("ANALYZE", "bad"),
                    ("REINDEX TABLE bookings.bookings", "bad")]},
    ],
    "task_17": [  # Compound: Conn Exhaustion + Deadlock — resolved when: idle < 5 AND timeout AND deadlock_detected
        {"prompt": "Investigate: Connections failing and transactions stuck — check sessions",
         "correct": "SELECT state, count(*) FROM pg_stat_activity GROUP BY state",
         "wrong": [("SHOW work_mem", "mild"),
                    ("SELECT * FROM pg_stat_bgwriter", "mild")]},
        {"prompt": "Identify: Many idle-in-transaction sessions — how many and how old?",
         "correct": "SELECT pid, state, age(now(), query_start) FROM pg_stat_activity WHERE state = 'idle in transaction'",
         "wrong": [("ALTER SYSTEM SET max_connections = 500", "bad"),
                    ("SELECT * FROM pg_stat_user_tables", "mild")]},
        {"prompt": "Resolve step 1: Terminate idle sessions to free connection slots",
         "correct": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND pid != pg_backend_pid()",
         "wrong": [("ALTER SYSTEM SET max_connections = 500", "bad"),
                    ("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active'", "bad")]},
        {"prompt": "Resolve step 2: Set a timeout to prevent recurrence",
         "correct": "ALTER SYSTEM SET idle_in_transaction_session_timeout = '60s'",
         "wrong": [("ALTER SYSTEM SET statement_timeout = '0'", "bad"),
                    ("SHOW idle_in_transaction_session_timeout", "mild")]},
    ],
}

HINT_TRUNCATE = 50  # chars to show before "..."


def _badge(difficulty: str) -> str:
    return f'<span class="badge-{difficulty}">{difficulty}</span>'


def _metrics_html(metrics: Optional[dict]) -> str:
    if not metrics:
        return '<div style="color:#000">Reset a task to see metrics</div>'
    items = [
        ("Active Conns", metrics.get("active_connections", "—")),
        ("Idle-in-Tx", metrics.get("idle_in_transaction", "—")),
        ("Dead Tuples", f'{metrics.get("total_dead_tuples", 0):,}'),
        ("Lock Waits", metrics.get("lock_waits", "—")),
    ]
    cards = ""
    for label, value in items:
        cards += f'''<div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>'''
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px">{cards}</div>'


_CHECKPOINT_LABELS = {
    # Investigation (under Diagnosis)
    "inv_explain": ("Ran EXPLAIN/EXPLAIN ANALYZE", "Checked query execution plan"),
    "inv_ran_explain": ("Ran EXPLAIN/EXPLAIN ANALYZE", "Checked query execution plan"),
    "inv_checked_activity": ("Checked pg_stat_activity", "Inspected active sessions and their states"),
    "inv_checked_locks": ("Checked lock information", "Queried pg_locks or lock-related views"),
    "inv_checked_max_conn": ("Checked max_connections", "Verified connection limit settings"),
    "inv_checked_stats": ("Checked table statistics", "Queried pg_stat_user_tables or similar"),
    "inv_checked_settings": ("Checked server settings", "Inspected pg_settings or SHOW commands"),
    "inv_checked_grants": ("Checked permissions", "Reviewed GRANT/privilege configuration"),
    "inv_checked_role": ("Checked role membership", "Inspected user roles and privileges"),
    "inv_checked_max_pk": ("Checked max primary key", "Inspected sequence vs. max PK values"),
    "inv_checked_sequence": ("Checked sequences", "Queried sequence state and values"),
    "inv_checked_catalogs": ("Checked system catalogs", "Queried pg_catalog tables"),
    "inv_checked_index_stats": ("Checked index stats", "Reviewed pg_stat_user_indexes"),
    "inv_checked_size": ("Checked table/index sizes", "Queried pg_total_relation_size or similar"),
    "inv_checked_table": ("Inspected table structure", "Examined table definition or columns"),
    "inv_checked_tables": ("Inspected multiple tables", "Examined several table structures"),
    # Identification (under Diagnosis)
    "id_target_table": ("Identified problem table", "Correctly pinpointed the affected table"),
    "id_target_column": ("Identified problem column", "Found the specific column involved"),
    "id_target_index": ("Identified problem index", "Found the relevant index"),
    "id_target_role": ("Identified affected role", "Found the user/role with the issue"),
    "id_stale_stats": ("Detected stale statistics", "Recognized outdated planner stats"),
    "id_dead_tuples": ("Detected dead tuple bloat", "Found excessive dead tuples needing vacuum"),
    "id_bloat_detected": ("Detected table/index bloat", "Recognized bloated tables or indexes"),
    "id_bloat_issue": ("Identified bloat as root cause", "Connected bloat to the performance issue"),
    "id_idle_sessions": ("Found idle sessions", "Identified idle-in-transaction connections"),
    "id_idle_problem": ("Diagnosed idle connection issue", "Understood impact of idle sessions"),
    "id_blocker_pattern": ("Identified blocking pattern", "Found the lock contention source"),
    "id_blocking_tx": ("Found blocking transaction", "Identified the specific blocking PID/query"),
    "id_lock_issue": ("Diagnosed lock contention", "Recognized lock waits as the problem"),
    "id_missing_index": ("Found missing index", "Identified need for a new index"),
    "id_mismatch": ("Detected data mismatch", "Found inconsistency in data"),
    "id_bad_param": ("Found bad parameter", "Identified a misconfigured setting"),
    "id_bad_params": ("Found bad parameters", "Identified misconfigured settings"),
    "id_both_params": ("Found both bad parameters", "Identified all misconfigured settings"),
    "id_column_order": ("Identified column order issue", "Found wrong column order in composite index"),
    "id_composite_key": ("Identified composite key issue", "Found multi-column key problem"),
    "id_sequence_name": ("Identified sequence name", "Found the correct sequence"),
    "id_snapshot_holder": ("Found snapshot holder", "Identified long-running transaction holding snapshot"),
    "id_terminate_idle": ("Decided to terminate idle", "Chose correct fix: terminate idle sessions"),
    "id_multi_table": ("Identified multiple tables affected", "Found issue spans several tables"),
    # Resolution
    "res_index_exists": ("Created needed index", "New index exists and is valid"),
    "res_index_created": ("Created the index", "Successfully ran CREATE INDEX"),
    "res_index_rebuilt": ("Rebuilt corrupted index", "Dropped and recreated index"),
    "res_plan_improved": ("Query plan improved", "Execution plan now uses better strategy"),
    "res_plan_uses_index": ("Plan uses new index", "Query now leverages the created index"),
    "res_analyze_ran": ("Ran ANALYZE", "Updated planner statistics"),
    "res_estimates_accurate": ("Planner estimates fixed", "Row estimates now match reality"),
    "res_idle_cleared": ("Cleared idle connections", "Idle-in-transaction sessions removed"),
    "res_idle_terminated": ("Terminated idle sessions", "Forcefully ended idle connections"),
    "res_locks_freed": ("Freed held locks", "Blocked transactions can now proceed"),
    "res_no_blocked_queries": ("No more blocked queries", "All queries running freely"),
    "res_no_blocked_txids": ("No blocked transactions", "No transactions waiting on locks"),
    "res_no_lock_waits": ("No lock waits remaining", "Lock contention fully resolved"),
    "res_no_deadlocks": ("No deadlocks", "Deadlock situation resolved"),
    "res_blocker_gone": ("Blocker removed", "Blocking transaction terminated"),
    "res_permission_granted": ("Permission granted", "Required access rights applied"),
    "res_sequence_reset": ("Sequence reset", "Sequence value corrected"),
    "res_insert_succeeds": ("Insert works now", "Previously failing INSERT succeeds"),
    "res_dead_tuples_reduced": ("Dead tuples reduced", "Vacuum cleaned up dead rows"),
    "res_tables_cleaned": ("Tables cleaned", "Bloated tables restored to health"),
    "res_junk_dropped": ("Junk indexes dropped", "Unnecessary indexes removed"),
    "res_pk_preserved": ("Primary key preserved", "Essential PK index kept intact"),
    "res_standalone_index": ("Created standalone index", "Index created without dependencies"),
    "res_fully_resolved": ("Fully resolved", "All aspects of the issue fixed"),
    "res_both_resolved": ("Both issues resolved", "Fixed all sub-problems"),
    "res_timeout_set": ("Timeout configured", "Statement timeout properly set"),
    # Best Practice
    "bp_no_destructive": ("No destructive commands", "Avoided DROP/TRUNCATE on production data"),
    "bp_clean_execution": ("Clean execution", "No errors or failed commands"),
    "bp_analyzed_after": ("Analyzed after fix", "Ran ANALYZE to update stats post-fix"),
    "bp_targeted_analyze": ("Targeted ANALYZE", "Analyzed specific table, not entire DB"),
    "bp_targeted_kill": ("Targeted termination", "Killed only the problem session, not all"),
    "bp_targeted_terminate": ("Targeted terminate", "Terminated only the blocking PID"),
    "bp_minimal_grants": ("Minimal privilege", "Granted only necessary permissions"),
    "bp_concurrently": ("Used CONCURRENTLY", "Created index without blocking writes"),
    "bp_ran_vacuum": ("Ran VACUUM", "Cleaned up dead tuples"),
    "bp_vacuumed_all": ("Vacuumed all tables", "Cleaned all affected tables"),
    "bp_alter_system": ("Used ALTER SYSTEM", "Changed settings via proper command"),
    "bp_reload_conf": ("Reloaded config", "Applied config changes with pg_reload_conf()"),
    "bp_correct_value": ("Set correct value", "Parameter set to appropriate value"),
    "bp_diagnosed_first": ("Diagnosed before fixing", "Investigated root cause before acting"),
    "bp_essential_preserved": ("Preserved essentials", "Kept critical indexes/constraints"),
    "bp_pk_preserved": ("Preserved primary key", "Didn't drop PK constraint"),
    "bp_prevention": ("Preventive measures", "Added safeguards against recurrence"),
    "bp_used_setval": ("Used setval()", "Reset sequence with correct function"),
}


def _grader_breakdown_html(breakdown: Optional[dict], score: Optional[float]) -> str:
    if not breakdown:
        return ""

    investigation = []
    identification = []
    resolution = []
    best_practice = []
    eff = breakdown.get("_efficiency_mult", 1.0)

    for k, v in sorted(breakdown.items()):
        if k.startswith("_"):
            continue
        label_info = _CHECKPOINT_LABELS.get(k, (k.replace("_", " ").title(), ""))
        entry = (label_info[0], label_info[1], v)
        if k.startswith("inv_"):
            investigation.append(entry)
        elif k.startswith("id_"):
            identification.append(entry)
        elif k.startswith("res_"):
            resolution.append(entry)
        elif k.startswith("bp_"):
            best_practice.append(entry)

    html = f'<div style="background:#ecfccb;border:3px solid #000;border-radius:4px;padding:16px">'
    html += f'<h3 style="margin:0 0 8px 0;color:#000;font-weight:900;font-size:18px">Grader Breakdown</h3>'
    html += f'<div style="display:flex;gap:16px;align-items:center;margin-bottom:16px">'
    html += f'<span style="font-size:28px;font-weight:900;color:#000">{score:.3f}</span>'

    # Efficiency badge with explanation
    if eff >= 1.0:
        eff_bg = "#d9f99d"
        eff_label = "Perfect"
    elif eff >= 0.8:
        eff_bg = "#fde047"
        eff_label = "Good"
    else:
        eff_bg = "#fecdd3"
        eff_label = "Slow"
    html += f'<span style="background:{eff_bg};border:2px solid #000;border-radius:4px;padding:4px 12px;font-weight:800;font-size:13px">'
    html += f'Efficiency: {eff:.2f}x ({eff_label})</span>'
    html += f'</div>'
    html += f'<div style="font-size:12px;color:#000;font-weight:600;margin-bottom:16px;background:#fff;border:1px solid #000;border-radius:4px;padding:8px">'
    html += f'The efficiency multiplier rewards solving the problem in fewer steps. Using all 15 steps gives ~0.5x; solving in under 5 steps gives 1.0x. It scales the final score.</div>'

    sections = [
        ("Diagnosis: Investigation", "#bfdbfe", investigation, "Did the agent inspect the right system views and metrics?"),
        ("Diagnosis: Identification", "#bfdbfe", identification, "Did the agent correctly identify the root cause?"),
        ("Resolution", "#d9f99d", resolution, "Did the agent successfully fix the problem?"),
        ("Best Practice", "#fde047", best_practice, "Did the agent follow SRE best practices?"),
    ]

    for section_name, bg, checks, desc in sections:
        if not checks:
            continue
        html += f'<div class="breakdown-section" style="background:{bg}">'
        html += f'<div class="breakdown-title">{section_name}</div>'
        html += f'<div style="font-size:11px;color:#000;font-weight:500;margin-bottom:8px;font-style:italic">{desc}</div>'
        for label, hint, val in checks:
            cls = "earned" if val > 0 else "missed"
            icon = "+" if val > 0 else "-"
            html += f'<div class="checkpoint">'
            html += f'<span class="checkpoint-name">{icon} {label}'
            if hint:
                html += f' <span style="font-weight:400;font-size:11px;color:#000">— {hint}</span>'
            html += f'</span>'
            html += f'<span class="checkpoint-value {cls}">{val:.2f}</span></div>'
        html += '</div>'

    html += '</div>'
    return html


def _trace_html(result: dict) -> str:
    """Render a single task trace as HTML."""
    steps = result.get("steps", [])
    task_name = result.get("task_name", "")
    score = result.get("grader_score", 0) or 0
    resolved = result.get("is_resolved", False)
    breakdown = result.get("grader_breakdown", {})

    res_badge = '<span class="resolved-yes">RESOLVED</span>' if resolved else '<span class="resolved-no">NOT RESOLVED</span>'

    html = f'''<div style="margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div>
                <strong style="font-size:18px;color:#000;font-weight:900;text-shadow:1.5px 1.5px 0 #a78bfa">{task_name}</strong>
                <span style="margin-left:8px">{_badge(result.get("difficulty", ""))}</span>
            </div>
            <div>{res_badge} <span style="margin-left:12px;font-size:18px;font-weight:900;color:#000">{score:.3f}</span></div>
        </div>
        <div style="font-size:12px;color:#000;font-weight:600;margin-bottom:12px">
            Steps: {result.get("steps_used", 0)} | Time: {result.get("elapsed_s", 0):.1f}s
        </div>
    '''

    for step in steps:
        reward = step.get("reward", 0)
        error = step.get("error")
        cls = "positive" if reward > 0 else ("negative" if (error or reward < 0) else "")
        rew_cls = "pos" if reward > 0 else "neg"
        cmd = step.get("command", step.get("error", "—"))
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."

        html += f'''<div class="step-card {cls}">
            <span class="step-num">Step {step.get("step", "?")}</span>
            <span class="step-cmd">{_escape(cmd)}</span>
            <span class="step-reward {rew_cls}">{reward:+.3f}</span>
        </div>'''

        if error:
            html += f'<div style="font-size:11px;color:#dc2626;margin:-4px 0 8px 28px;font-family:monospace">{_escape(error[:200])}</div>'

    html += '</div>'

    if breakdown:
        html += _grader_breakdown_html(breakdown, score)

    return html


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _leaderboard_html(all_results: dict) -> str:
    """Build leaderboard table HTML."""
    if not all_results:
        return '<div style="color:#000;text-align:center;padding:40px;font-weight:600">No results available yet.</div>'

    rows = []
    for model, data in sorted(all_results.items(), key=lambda x: x[1].get("summary", {}).get("total_score", 0), reverse=True):
        s = data.get("summary", {})
        rows.append({
            "model": _model_display_name(model),
            "total": s.get("total_score", 0),
            "avg": s.get("average_score", 0),
            "resolved": s.get("resolved_count", 0),
            "tasks": s.get("total_tasks", 17),
        })

    html = '''<div style="overflow-x:auto"><table class="leaderboard-table" style="margin-bottom:24px">
        <thead><tr>
            <th style="text-align:left">Rank</th>
            <th style="text-align:left">Model</th>
            <th>Total Score</th>
            <th>Average</th>
            <th>Resolved</th>
        </tr></thead><tbody>'''

    for i, r in enumerate(rows):
        rank_cls = ' class="rank-1"' if i == 0 else ""
        bg = "#fef9c3" if i % 2 else "#fff"
        html += f'''<tr{rank_cls} style="background:{bg}">
            <td style="font-weight:900;text-align:center">{i+1}</td>
            <td style="text-align:left;font-weight:700">{r["model"]}</td>
            <td class="score-cell">{r["total"]:.3f} / {r["tasks"]}</td>
            <td>{r["avg"]:.3f}</td>
            <td>{r["resolved"]} / {r["tasks"]}</td>
        </tr>'''

    html += '</tbody></table></div>'
    return html


def _heatmap_html(all_results: dict) -> str:
    """Build score heatmap: models × tasks."""
    if not all_results:
        return ""

    task_ids = [f"task_{i}" for i in range(1, 18)]
    models = sorted(all_results.keys(), key=lambda m: all_results[m].get("summary", {}).get("total_score", 0), reverse=True)

    # Build task ID → result mapping per model
    model_scores = {}
    for model in models:
        by_task = {}
        for r in all_results[model].get("results", []):
            by_task[r.get("task_id", "")] = r.get("grader_score", 0) or 0
        model_scores[model] = by_task

    html = '<div style="overflow-x:auto"><table class="heatmap-table"><thead><tr><th style="text-align:left">Model</th>'
    for tid in task_ids:
        num = tid.split("_")[1]
        html += f'<th>T{num}</th>'
    html += '</tr></thead><tbody>'

    for model in models:
        html += f'<tr><td style="text-align:left;font-weight:600;white-space:nowrap">{_model_display_name(model)}</td>'
        for tid in task_ids:
            score = model_scores[model].get(tid, 0)
            # Color: red (0) → yellow (0.5) → green (1.0) — solid backgrounds
            if score >= 0.7:
                bg = "#d9f99d"
            elif score >= 0.4:
                bg = "#fde047"
            elif score > 0:
                bg = "#fecdd3"
            else:
                bg = "#fee2e2"
            html += f'<td style="background:{bg};color:#000;font-weight:700">{score:.2f}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def _readme_tab_html() -> str:
    """Build the README landing page with VibeCheck-style colored blocks."""
    # Block style helper
    def _block(color, content, extra_style=""):
        return (
            f'<div style="background:{color};border:3px solid #000;border-radius:8px;'
            f'padding:24px 28px;margin-bottom:20px;color:#000;box-shadow:3px 3px 0 #000;{extra_style}">'
            f'{content}</div>'
        )

    blocks = []

    # ── Block 1: Hero / The Hook ──
    blocks.append(_block("#ffb74d", '''
        <h2 style="font-size:24px;font-weight:900;margin:0 0 12px 0">
            SQLab: Database Incident Response Training for LLM Agents</h2>
        <p style="font-size:15px;line-height:1.6;margin:0 0 12px 0">
            SQL databases power nearly every production application &mdash; from booking systems
            to financial platforms. When they break, the symptoms are cryptic: queries that ran in
            milliseconds now take seconds, connections pile up until the pool is exhausted, transactions
            deadlock each other, and bloated tables silently degrade performance. Diagnosing these
            failures requires reading execution plans, inspecting lock graphs, and understanding how
            the query planner makes decisions &mdash; skills that take years to develop.</p>
        <p style="font-size:15px;line-height:1.6;margin:0 0 16px 0">
            SQLab is an OpenEnv environment where LLM agents learn these skills. It presents
            <b>17 production-realistic PostgreSQL faults</b> &mdash; missing indexes, stale statistics,
            deadlock chains, cascading bloat, misconfigured parameters, and more &mdash; against a live
            database with 20 million rows of airline booking data. The agent receives an alert, has
            15 steps to investigate and fix the issue using raw SQL, and is scored by a deterministic
            grader on diagnosis, resolution, and best practices (0&ndash;1 scale, fully reproducible,
            no LLM judge).</p>
        <p style="font-size:14px;font-weight:600;margin:0">
            Try it in the <b>Playground</b> tab, or read on for details.</p>
    '''))

    # ── Block 1b: Episode Loop Diagram ──
    blocks.append(_block("#d1fae5", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">How an Episode Works</h3>
        <div style="text-align:center;margin:0 0 14px 0">
            <img src="/static/episode-flow.png" alt="Episode flow: Alert, Diagnose, Fix, Verify"
                 style="max-width:100%;height:auto;border:2px solid #000;border-radius:4px" />
        </div>
        <p style="font-size:14px;line-height:1.6;margin:0">
            The agent receives an alert and a live PostgreSQL database. It issues raw SQL commands
            to investigate and fix the issue. At the end, a deterministic grader scores the episode
            across diagnosis, resolution, and best practices.</p>
    '''))

    # ── Block 1c: Example Episode Walkthrough ──
    blocks.append(_block("#bfdbfe", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">Example: Missing Index</h3>
        <div class="sql-output" style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
                    line-height:1.6;background:#0a1628 !important;color:#4ade80 !important;border:2px solid #000;
                    border-radius:4px;padding:14px 16px;margin:0 0 14px 0;overflow-x:auto;max-height:none !important">
<b style="color:#fde047">Alert:</b> High query latency on ticket_flights (avg 2.3s, p99 8.1s)
<br><br>
<b style="color:#fde047">Step 1:</b> EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1
<br><span style="color:#94a3b8">  → Seq Scan on ticket_flights (cost=0.00..287434.12)</span>  <span style="color:#f87171">← No index!</span>
<br><span style="color:#60a5fa">  → reward: +0.05 (targeted diagnostic)</span>
<br><br>
<b style="color:#fde047">Step 2:</b> SELECT * FROM pg_indexes WHERE tablename = 'ticket_flights'
<br><span style="color:#94a3b8">  → Only primary key, no index on flight_id</span>
<br><span style="color:#60a5fa">  → reward: +0.05 (right-table diagnostic)</span>
<br><br>
<b style="color:#fde047">Step 3:</b> CREATE INDEX ON bookings.ticket_flights(flight_id)
<br><span style="color:#94a3b8">  → CREATE INDEX (success)</span>
<br><span style="color:#60a5fa">  → reward: +0.10 (correct fix for missing_index)</span>
<br><br>
<b style="color:#fde047">Step 4:</b> EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1
<br><span style="color:#94a3b8">  → Index Scan using idx_ticket_flights_flight_id (cost=0.43..8.45)</span>  <span style="color:#4ade80">← Fixed!</span>
<br><span style="color:#60a5fa">  → Grader: 0.85 (diagnosis 0.4 + resolution 0.4 + best practice 0.05)</span>
</div>
        <p style="font-size:14px;line-height:1.6;margin:0">
            Four steps: investigate, confirm, fix, verify. The grader rewards both the
            journey and the outcome. Try this task in the <b>Playground</b> tab.</p>
    '''))

    # ── Block 2: Real-World Utility ──
    blocks.append(_block("#fef3c7", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">Real-World Utility</h3>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            Every fault in SQLab is modeled on real PostgreSQL failure modes: a missing
            index causing 100x query slowdowns, bloated tables blocking autovacuum, a misconfigured
            <code>work_mem</code> silently degrading every query on the server. These are the
            same issues that production SREs (Site Reliability Engineers) encounter regularly.</p>
        <p style="font-size:15px;line-height:1.7;margin:0 0 14px 0">
            The training database is the <a href="https://postgrespro.com/community/demodb" target="_blank" style="color:#1e40af;text-decoration:underline;font-weight:700">Airlines demo</a>: 20 million rows of flights, tickets,
            and bookings. Realistic enough that EXPLAIN plans behave like production, indexes
            matter, and lock contention actually blocks. The skills agents learn here transfer
            directly to real database operations.</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;font-weight:700">
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:8px 12px">
                <b>Performance</b>: missing indexes, stale statistics, wrong column order</div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:8px 12px">
                <b>Resources</b>: connection exhaustion, lock contention, deadlocks</div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:8px 12px">
                <b>Storage</b>: table bloat, index bloat, cascading multi-table bloat</div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:8px 12px">
                <b>Configuration</b>: bad settings, query plan flips</div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:8px 12px">
                <b>Access &amp; Integrity</b>: permission errors, sequence exhaustion</div>
        </div>
    '''))

    # ── Block 3: Task & Grader Quality ──
    blocks.append(_block("#d1fae5", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">Reward Design</h3>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            SQLab has 17 tasks across three difficulty tiers. Easy tasks involve a single clear
            fault. Medium tasks require multi-step investigation. Hard tasks throw two simultaneous
            faults at the agent, forcing it to prioritize and coordinate fixes.</p>
        <p style="font-size:15px;line-height:1.7;margin:0 0 14px 0">
            Every task is scored by a deterministic grader. No LLM judge, fully reproducible.
            The grader evaluates three things:</p>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:14px">
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:10px 14px">
                <div style="font-weight:900;font-size:14px;margin-bottom:6px">Diagnosis (40%)</div>
                <div style="font-size:13px;line-height:1.5">
                    Did the agent investigate with the right tools? Did it identify the specific
                    fault? Not just "did it run EXPLAIN" but "did it EXPLAIN the right table?"</div>
            </div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:10px 14px">
                <div style="font-weight:900;font-size:14px;margin-bottom:6px">Resolution (40%)</div>
                <div style="font-size:13px;line-height:1.5">
                    Is the database actually fixed? The grader checks real DB state, not keywords.
                    If the agent said CREATE INDEX but it failed silently, the grader catches that.
                    Solving faster scores higher.</div>
            </div>
            <div style="background:#fff;border:2px solid #000;border-radius:4px;padding:10px 14px">
                <div style="font-weight:900;font-size:14px;margin-bottom:6px">Best Practice (20%)</div>
                <div style="font-size:13px;line-height:1.5">
                    Did the agent avoid destructive commands? Keep its error rate low? Use safety
                    measures like CONCURRENTLY?</div>
            </div>
        </div>
        <p style="font-size:14px;line-height:1.6;margin:0">
            Browse all 17 tasks in the <b>Tasks</b> tab, or check model performance in the <b>Leaderboard</b>.</p>
    '''))

    # ── Block 3b: Anti-Reward-Hacking ──
    blocks.append(_block("#fde047", '''
        <h3 style="font-size:18px;font-weight:900;margin:0 0 12px 0">Anti-Reward-Hacking</h3>
        <p style="font-size:14px;line-height:1.7;margin:0 0 10px 0">
            Per-step rewards are fault-type-gated: running <code>CREATE INDEX</code> on a bloat
            task earns zero. Diagnostics must target the correct table. Each reward category
            fires at most once per episode, preventing score accumulation through repetition.
            Applying the wrong fix incurs a -0.03 penalty.</p>
        <p style="font-size:14px;line-height:1.7;margin:0">
            Validated by <b>255 adversarial unit tests</b> covering cross-task fix matrices,
            repetition gaming, wrong-table diagnostics, and cumulative overflow.</p>
    '''))

    # ── Block 4: Environment Design ──
    blocks.append(_block("#bfdbfe", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">Environment Design</h3>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            An episode in SQLab works like a real incident. The agent receives an alert and a
            live database it can query freely with SQL. No multiple-choice menus, no constrained
            action space. Just raw SQL, the way a real SRE works.</p>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            Each step returns the SQL output, an error message if something went wrong, and live
            database metrics: active connections, dead tuples, lock waits. The agent has 15 steps
            to diagnose and fix the issue before the episode ends.</p>
        <p style="font-size:15px;line-height:1.7;margin:0">
            Destructive commands (DROP TABLE, VACUUM FULL, ALTER USER) immediately terminate
            the episode with a -0.5 penalty, teaching agents to avoid unrecoverable actions.
            Task-aware exceptions allow commands that are the correct fix, such as DROP INDEX
            for over-indexing tasks. Fault injection uses pre-baked SQL for fast resets
            (2 to 5 seconds).</p>
    '''))

    # ── Block 5: Baselines ──
    blocks.append(_block("#fecdd3", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">Baseline Results</h3>
        <p style="font-size:15px;line-height:1.7;margin:0 0 14px 0">
            Five open-source models tested against all 17 tasks with anti-hack reward shaping.
            Average scores range from 0.42 to 0.64. Full per-task breakdown in the <b>Leaderboard</b> tab.</p>
        <table style="width:auto;margin:0 auto;border-collapse:collapse;font-size:12px;font-weight:600">
            <tr style="background:#fff;border:2px solid #000">
                <th style="padding:5px 10px;text-align:left;border:1px solid #000">Model</th>
                <th style="padding:5px 10px;text-align:center;border:1px solid #000">Avg Score</th>
                <th style="padding:5px 10px;text-align:center;border:1px solid #000">Resolved</th>
            </tr>
            <tr style="border:1px solid #000">
                <td style="padding:5px 10px;border:1px solid #000">Phi-4 14B</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">0.635</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">8 / 17</td>
            </tr>
            <tr style="border:1px solid #000">
                <td style="padding:5px 10px;border:1px solid #000">Qwen2.5-Coder 14B</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">0.596</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">7 / 17</td>
            </tr>
            <tr style="border:1px solid #000">
                <td style="padding:5px 10px;border:1px solid #000">Devstral 15B</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">0.595</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">6 / 17</td>
            </tr>
            <tr style="border:1px solid #000">
                <td style="padding:5px 10px;border:1px solid #000">Qwen2.5-Coder 7B</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">0.445</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">1 / 17</td>
            </tr>
            <tr style="border:1px solid #000">
                <td style="padding:5px 10px;border:1px solid #000">DeepSeek-Coder-V2 16B</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">0.417</td>
                <td style="padding:5px 10px;text-align:center;border:1px solid #000">3 / 17</td>
            </tr>
        </table>
    '''))

    # ── Block 7: Vision ──
    blocks.append(_block("#ffb74d", '''
        <h3 style="font-size:20px;font-weight:900;margin:0 0 14px 0">
            Vision: Multi-Agent Database Operations</h3>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            Today, SQLab trains a single agent on a single incident in 15-step episodes.
            A focused training ground for the fundamentals.</p>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            The natural extension is multi-agent database fleet management: a <b>triage agent</b>
            prioritizing incidents across a cluster, a <b>diagnostic agent</b> building fault
            hypotheses, a <b>remediation agent</b> applying fixes with rollback plans, and a
            <b>monitoring agent</b> watching for regressions. Agents would coordinate across
            replicas: failover, fix, resync.</p>
        <p style="font-size:15px;line-height:1.7;margin:0 0 12px 0">
            SQLab is where these agents learn the fundamentals, the same way a junior SRE
            learns on single-node incidents before managing a fleet. The compound tasks
            (tasks 12 to 17) are a first step: two simultaneous faults requiring multi-step
            reasoning. The next step is multi-agent coordination.</p>
        <p style="font-size:15px;line-height:1.7;margin:0;font-style:italic">
            We believe database operations will be among the first domains where multi-agent
            systems deliver production value. The workflow is structured, the feedback is
            immediate, and the stakes are high enough to demand reliability.</p>
    '''))

    return '\n'.join(blocks)


def _task_descriptions_html() -> str:
    """Build accordion of task descriptions."""
    html = '<div style="margin-top:24px">'
    for tid, task in TASK_REGISTRY.items():
        num = tid.split("_")[1]
        html += f'''<details class="task-accordion">
            <summary>Task {num}: {task["name"]} {_badge(task["difficulty"])}</summary>
            <div class="acc-body">
                <p>{task["description"]}</p>
                <div class="alert-panel" style="margin-top:8px">{_escape(task["alert"])}</div>
            </div>
        </details>'''
    html += '</div>'
    return html


# ── Gradio App ───────────────────────────────────────────────────────

def create_gradio_app(env, env_lock: threading.Lock) -> gr.Blocks:
    """Build the 3-tab Gradio interface.

    Args:
        env: DBSreEnvironment instance (shared with FastAPI)
        env_lock: Threading lock for serializing env access
    """
    all_results = _load_all_results()

    # Task choices for dropdown
    task_choices = []
    for tid, task in TASK_REGISTRY.items():
        num = tid.split("_")[1]
        task_choices.append((f"Task {num}: {task['name']} [{task['difficulty']}]", tid))

    # Model choices for traces
    model_choices = [(f"{_model_display_name(m)} — {d.get('summary', {}).get('total_score', 0):.1f}/17", m) for m, d in
                     sorted(all_results.items(), key=lambda x: x[1].get("summary", {}).get("total_score", 0), reverse=True)]

    with gr.Blocks(title="SQLab") as demo:

        gr.HTML(f'<style>{CUSTOM_CSS}</style>')

        # Header
        gr.HTML('''<div class="gym-header">
            <h1>SQLab</h1>
            <p>PostgreSQL Incident Response Training for LLM Agents</p>
        </div>''')

        # ── Tab 0: README (landing page) ─────────────────────────
        with gr.Tab("\u25A4 README"):
            gr.HTML(_readme_tab_html())

        # ── Tab 1: Interactive Playground ──────────────────────────
        with gr.Tab("\u2318 Playground"):

            # ── SUBBLOCK 1: Task Selection ──
            with gr.Group():
                gr.HTML('<div class="playground-subblock-title" data-pg="task-select">Task Selection</div>')
                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=task_choices, label="Select Task", show_label=False, scale=3,
                    )
                    reset_btn = gr.Button("Reset", elem_classes=["primary-btn"], scale=1)
                alert_display = gr.HTML(
                    '<div class="alert-panel" style="color:#000">Select a task and click Reset to begin.</div>',
                    label="Alert",
                )

            # ── SUBBLOCK 2: SQL Workflow ──
            with gr.Group():
                gr.HTML('<div class="playground-subblock-title" data-pg="sql-workflow">SQL Workflow</div>')
                with gr.Row():
                    with gr.Column(scale=3):
                        sql_input = gr.Textbox(
                            label="SQL Command", show_label=False,
                            placeholder="e.g. SELECT * FROM pg_stat_activity",
                            lines=1,
                        )
                        execute_btn = gr.Button("Execute", elem_classes=["primary-btn"])
                        # ── Guided hints sub-subblock ──
                        with gr.Group():
                            gr.HTML('<div class="playground-subblock-title" style="font-size:13px;margin-bottom:6px">Hint System</div>')
                            path_prompt = gr.HTML('<div class="path-prompt">Select a task and reset to start the guided path.</div>')
                            hint_btn_1 = gr.Button("—", size="sm", elem_classes=["hint-pill"])
                            hint_btn_2 = gr.Button("—", size="sm", elem_classes=["hint-pill"])
                            hint_btn_3 = gr.Button("—", size="sm", elem_classes=["hint-pill"])
                            with gr.Row():
                                reveal_check = gr.Checkbox(label="Reveal", value=False, elem_classes=["reveal-check"])
                            gr.HTML('<div class="hint-note">Guided hints for illustration only — the model receives no hints. Action space is the entire SQL language.</div>')
                    with gr.Column(scale=1):
                        step_display = gr.HTML('<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Step</div></div>')
                        reward_display = gr.HTML('<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Reward</div></div>')
                        status_display = gr.HTML('<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Status</div></div>')
                # Metrics row (updates after each step)
                metrics_display = gr.HTML(_metrics_html(None), label="Database Metrics")
                # ── REPL observation log sub-subblock ──
                with gr.Group():
                    gr.HTML('<div class="playground-subblock-title" data-pg="repl">Observation Log</div>')
                    obs_log_display = gr.HTML(
                        '<div class="repl-log" style="opacity:0.5">Execute commands to see results here.</div>',
                    )

            # State for the decision tree
            hint_state = gr.State({"task_id": "", "path_idx": 0, "path_done": False, "path_failed": False})

            # ── SUBBLOCK 3: Grader Breakdown ──
            with gr.Group():
                gr.HTML('<div class="playground-subblock-title" data-pg="grader">Grader Breakdown</div>')
                grader_display = gr.HTML('<div style="color:#6b7280;font-size:13px">Complete an episode to see the grader breakdown.</div>')

            # State
            playground_state = gr.State({
                "active": False,
                "step": 0,
                "cumulative_reward": 0.0,
                "obs_log_html": "",
                "done": False,
            })

            def _get_path_step_options(task_id, path_idx):
                """Return shuffled options: [(cmd, is_correct, severity), ...] and prompt.

                severity is "correct", "mild", "bad", or "fatal".
                """
                import random
                path = TASK_PATHS.get(task_id, [])
                if not path or path_idx >= len(path):
                    return [("—", False, "mild"), ("—", False, "mild"), ("—", False, "mild")], "Path complete."
                step = path[path_idx]
                items = [(step["correct"], True, "correct")]
                for w in step["wrong"][:2]:
                    # w is (sql, severity) tuple
                    items.append((w[0], False, w[1]))
                random.shuffle(items)
                return items, step["prompt"]

            def _path_prompt_html(prompt, path_idx, total_steps, done=False, failed=False, fatal=False, mild_msg=None, bad_msg=None):
                """Render the guided path prompt bar."""
                if done:
                    return '<div class="path-prompt path-done"><span class="path-step-badge">COMPLETE</span> All steps finished — well done!</div>'
                if fatal:
                    return ('<div class="path-prompt path-fatal">'
                            '<span class="path-step-badge">CRITICAL FAILURE</span> '
                            'Destructive action terminated the episode with penalty. Reset to try again.</div>')
                if failed:
                    return '<div class="path-prompt path-fail"><span class="path-step-badge">WRONG</span> Incorrect choice. Click Reset to try again.</div>'
                if bad_msg:
                    return (f'<div class="path-prompt path-fail">'
                            f'<span class="path-step-badge">Step {path_idx + 1}/{total_steps}</span> '
                            f'Dangerous approach! That wasted a step with negative reward. Try another option.</div>')
                if mild_msg:
                    return (f'<div class="path-prompt" style="background:#fef9c3;border-color:#ca8a04">'
                            f'<span class="path-step-badge">Step {path_idx + 1}/{total_steps}</span> '
                            f'Not quite — this doesn\'t help here. Try another option.</div>')
                return (f'<div class="path-prompt">'
                        f'<span class="path-step-badge">Step {path_idx + 1}/{total_steps}</span> {_escape(prompt)}'
                        f'</div>')

            def do_reset(task_id, state):
                empty_hs = {"task_id": "", "path_idx": 0, "path_done": False, "path_failed": False, "options": [], "disabled": [False, False, False]}
                empty_hints = (
                    '<div class="path-prompt">Select a task and reset to start the guided path.</div>',
                    gr.update(value="—", variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value="—", variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value="—", variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value=False),  # reveal checkbox
                    empty_hs,
                )

                if not task_id:
                    return (
                        '<div class="alert-panel" style="color:#000;font-weight:700">Please select a task first.</div>',
                        '<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Step</div></div>',
                        '<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Reward</div></div>',
                        '<div class="metric-card"><div class="metric-value">—</div><div class="metric-label">Status</div></div>',
                        '<div class="repl-log" style="opacity:0.5">Execute commands to see results here.</div>',
                        _metrics_html(None),
                        '<div style="color:#6b7280;font-size:13px">Complete an episode to see the grader breakdown.</div>',
                        {"active": True, "step": 0, "cumulative_reward": 0.0, "obs_log_html": "", "done": False},
                        *empty_hints,
                    )

                with env_lock:
                    obs = env.reset(task_id=task_id)

                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                alert_text = obs_dict.get("alert", "")
                metrics = obs_dict.get("metrics", {})

                # Set up first step of the guided path
                path = TASK_PATHS.get(task_id, [])
                total_steps = len(path)
                options, prompt = _get_path_step_options(task_id, 0)
                hint_st = {
                    "task_id": task_id,
                    "path_idx": 0,
                    "path_done": False,
                    "path_failed": False,
                    "options": [(o[0], o[1], o[2]) for o in options],  # (cmd, is_correct, severity)
                    "disabled": [False, False, False],
                }

                # Build initial REPL content showing the system prompt and alert
                init_log = (
                    '<span style="color:#60a5fa">── System Prompt ──</span>\n'
                    'You are an expert PostgreSQL Database SRE.\n'
                    'Diagnose the problem and fix it by issuing SQL commands.\n'
                    'Wrap your SQL in &lt;sql&gt; tags. One command per turn. 15 steps max.\n'
                    'Database: demo | Schema: bookings\n\n'
                    f'<span style="color:#60a5fa">── Alert ──</span>\n'
                    f'{_escape(alert_text)}\n\n'
                )

                return (
                    f'<div class="alert-panel">{_escape(alert_text)}</div>',
                    '<div class="metric-card"><div class="metric-value">0 / 15</div><div class="metric-label">Step</div></div>',
                    '<div class="metric-card"><div class="metric-value">0.000</div><div class="metric-label">Reward</div></div>',
                    '<div class="metric-card"><div class="metric-value">Active</div><div class="metric-label">Status</div></div>',
                    f'<div class="repl-log">{init_log}</div>',
                    _metrics_html(metrics),
                    '<div style="color:#6b7280;font-size:13px">Complete an episode to see the grader breakdown.</div>',
                    {"active": True, "step": 0, "cumulative_reward": 0.0, "obs_log_html": init_log, "done": False},
                    _path_prompt_html(prompt, 0, total_steps),
                    gr.update(value=options[0][0], variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value=options[1][0], variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value=options[2][0], variant="secondary", interactive=True, elem_classes=["hint-pill"]),
                    gr.update(value=False),  # reveal checkbox
                    hint_st,
                )

            def _build_repl_entry(sql, output, error, reward):
                """Build a single REPL-style entry for the observation log."""
                prompt = '<span class="rp">postgres=#</span>'
                cmd_span = f'<span class="rc">{_escape(sql.strip())};</span>'
                if error:
                    out_span = f'<span class="re">{_escape(error[:500])}</span>'
                elif output:
                    out_span = _escape(output[:2000])
                else:
                    out_span = '<span style="opacity:0.5">(no output)</span>'
                rew_cls = "pos" if reward > 0 else ("neg" if reward < 0 else "zero")
                rew_span = f'<span class="rr {rew_cls}">reward: {reward:+.3f}</span>'
                return f'{prompt} {cmd_span}\n{out_span}\n{rew_span}\n\n'

            def _execute_and_build(sql, state):
                """Execute SQL against env, return (obs_log_html, metrics_html,
                step_html, reward_html, status_html, grader_html, new_state, obs_dict)."""
                action = DBSreAction(command=sql.strip())
                with env_lock:
                    obs = env.step(action)

                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                output = obs_dict.get("command_output", "")
                error = obs_dict.get("error")
                reward = obs_dict.get("reward", 0)
                done = obs_dict.get("done", False)
                metrics = obs_dict.get("metrics", {})
                metadata = obs_dict.get("metadata", {})
                is_resolved = metadata.get("is_resolved", False)

                step = state["step"] + 1
                cum_reward = metadata.get("cumulative_reward", state["cumulative_reward"] + reward)

                # Build REPL entry and append to log
                repl_entry = _build_repl_entry(sql, output, error, reward)
                obs_log_inner = state["obs_log_html"] + repl_entry
                obs_log_html = f'<div class="repl-log">{obs_log_inner}</div>'

                if done and is_resolved:
                    status_html = ('<div style="background:#d9f99d;border:3px solid #16a34a;border-radius:6px;padding:10px;text-align:center">'
                                   '<div style="font-size:20px;font-weight:900;color:#166534">RESOLVED</div>'
                                   '<div style="font-size:12px;font-weight:700;color:#166534">Incident fixed in {step} steps</div></div>'.format(step=step))
                elif done:
                    reason = "Fatal action" if metadata.get("fatal_action") else f"Step {step}/15"
                    status_html = ('<div style="background:#fecdd3;border:3px solid #dc2626;border-radius:6px;padding:10px;text-align:center">'
                                   '<div style="font-size:20px;font-weight:900;color:#991b1b">NOT RESOLVED</div>'
                                   '<div style="font-size:12px;font-weight:700;color:#991b1b">{reason}</div></div>'.format(reason=reason))
                else:
                    status_html = '<div class="metric-card"><div class="metric-value">Active</div><div class="metric-label">Step {step}/15</div></div>'.format(step=step)

                grader_html = '<div style="color:#6b7280;font-size:13px">Complete an episode to see the grader breakdown.</div>'
                if done:
                    grader_result = getattr(type(env), 'last_grader_result', None)
                    if grader_result:
                        grader_html = _grader_breakdown_html(
                            grader_result.get("breakdown"),
                            grader_result.get("score", 0),
                        )

                new_state = {
                    "active": True,
                    "step": step,
                    "cumulative_reward": cum_reward,
                    "obs_log_html": obs_log_inner,
                    "done": done,
                }

                return (obs_log_html, _metrics_html(metrics),
                        f'<div class="metric-card"><div class="metric-value">{step} / 15</div><div class="metric-label">Step</div></div>',
                        f'<div class="metric-card"><div class="metric-value">{cum_reward:.3f}</div><div class="metric-label">Reward</div></div>',
                        status_html, grader_html, new_state, obs_dict)

            def do_execute(sql, state, hs):
                """Execute SQL and provide path feedback if the command matches a hint option."""
                no_path = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), hs)

                if not state.get("active"):
                    err_log = state.get("obs_log_html", "")
                    err_log += '<span class="re">No active episode. Reset a task first.</span>\n\n'
                    return (
                        f'<div class="repl-log">{err_log}</div>',
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        state, *no_path,
                    )
                if not sql or not sql.strip():
                    err_log = state.get("obs_log_html", "")
                    err_log += '<span class="re">Please enter a SQL command.</span>\n\n'
                    return (
                        f'<div class="repl-log">{err_log}</div>',
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        state, *no_path,
                    )

                result = _execute_and_build(sql, state)
                obs_log_html, metrics_html, step_html, reward_html, status_html, grader_html, new_state, obs_dict = result

                # ── Path feedback: check if executed SQL matches a current option ──
                options = hs.get("options", [])
                disabled = hs.get("disabled", [False, False, False])
                path_done = hs.get("path_done", False)
                path_failed = hs.get("path_failed", False)

                if not options or path_done or path_failed:
                    return (obs_log_html, metrics_html, step_html,
                            reward_html, status_html, grader_html, new_state, *no_path)

                # Find which option (if any) was executed
                sql_stripped = sql.strip()
                matched_idx = None
                for i, opt in enumerate(options):
                    if opt[0].strip() == sql_stripped:
                        matched_idx = i
                        break

                if matched_idx is None:
                    # SQL doesn't match any hint option — no path update
                    return (obs_log_html, metrics_html, step_html,
                            reward_html, status_html, grader_html, new_state, *no_path)

                cmd, is_correct, severity = options[matched_idx]
                is_fatal_action = obs_dict.get("metadata", {}).get("fatal_action", False)

                task_id = hs.get("task_id", "")
                path_idx = hs.get("path_idx", 0)
                path = TASK_PATHS.get(task_id, [])
                total_steps = len(path)

                if is_correct:
                    # ── Correct: reveal all, advance path ──
                    btn_updates = []
                    for i in range(3):
                        if i < len(options):
                            _, ic, _ = options[i]
                            variant = "primary" if ic else "stop"
                            btn_updates.append(gr.update(variant=variant, interactive=False))
                        else:
                            btn_updates.append(gr.update())

                    next_idx = path_idx + 1
                    if next_idx >= total_steps:
                        new_hs = {**hs, "path_idx": next_idx, "path_done": True, "options": [], "disabled": [True, True, True]}
                        prompt_html = _path_prompt_html("", next_idx, total_steps, done=True)
                    else:
                        next_options, next_prompt = _get_path_step_options(task_id, next_idx)
                        new_hs = {
                            **hs, "path_idx": next_idx,
                            "options": [(o[0], o[1], o[2]) for o in next_options],
                            "disabled": [False, False, False],
                        }
                        prompt_html = _path_prompt_html(next_prompt, next_idx, total_steps)
                        btn_updates = [
                            gr.update(value=next_options[0][0], variant="secondary", interactive=True),
                            gr.update(value=next_options[1][0], variant="secondary", interactive=True),
                            gr.update(value=next_options[2][0], variant="secondary", interactive=True),
                        ]

                elif severity == "fatal" or is_fatal_action:
                    # ── Fatal: disable all ──
                    btn_updates = [gr.update(interactive=False, variant="stop") for _ in range(3)]
                    new_hs = {**hs, "path_failed": True, "disabled": [True, True, True]}
                    prompt_html = _path_prompt_html("", path_idx, total_steps, fatal=True)

                elif severity == "bad":
                    # ── Bad: disable matched button ──
                    new_disabled = list(disabled)
                    new_disabled[matched_idx] = True
                    btn_updates = []
                    for i in range(3):
                        if i == matched_idx:
                            btn_updates.append(gr.update(variant="stop", interactive=False))
                        else:
                            btn_updates.append(gr.update())
                    new_hs = {**hs, "disabled": new_disabled}
                    prompt_html = _path_prompt_html("", path_idx, total_steps, bad_msg=True)

                else:  # mild
                    # ── Mild: disable matched button ──
                    new_disabled = list(disabled)
                    new_disabled[matched_idx] = True
                    btn_updates = []
                    for i in range(3):
                        if i == matched_idx:
                            btn_updates.append(gr.update(variant="stop", interactive=False))
                        else:
                            btn_updates.append(gr.update())
                    new_hs = {**hs, "disabled": new_disabled}
                    prompt_html = _path_prompt_html("", path_idx, total_steps, mild_msg=True)

                return (
                    obs_log_html, metrics_html,
                    step_html, reward_html, status_html, grader_html,
                    new_state,
                    prompt_html, btn_updates[0], btn_updates[1], btn_updates[2],
                    gr.update(value=False),  # reset reveal checkbox
                    new_hs,
                )

            _reset_outputs = [alert_display, step_display, reward_display, status_display, obs_log_display, metrics_display, grader_display, playground_state, path_prompt, hint_btn_1, hint_btn_2, hint_btn_3, reveal_check, hint_state]
            reset_btn.click(
                do_reset,
                inputs=[task_dropdown, playground_state],
                outputs=_reset_outputs,
            )

            # Hint buttons — just fill the SQL input box
            def load_hint(idx, hs):
                options = hs.get("options", [])
                disabled = hs.get("disabled", [False, False, False])
                if idx < len(options) and not (idx < len(disabled) and disabled[idx]):
                    return options[idx][0]  # cmd text
                return gr.update()

            hint_btn_1.click(lambda hs: load_hint(0, hs), inputs=[hint_state], outputs=[sql_input])
            hint_btn_2.click(lambda hs: load_hint(1, hs), inputs=[hint_state], outputs=[sql_input])
            hint_btn_3.click(lambda hs: load_hint(2, hs), inputs=[hint_state], outputs=[sql_input])

            # Reveal checkbox — change button variants to show correct (green) / wrong (red)
            def toggle_reveal(checked, hs):
                options = hs.get("options", [])
                disabled = hs.get("disabled", [False, False, False])
                btns = []
                for i in range(3):
                    if i < len(disabled) and disabled[i]:
                        btns.append(gr.update())  # already styled, don't change
                    elif checked and i < len(options):
                        _, is_correct, _ = options[i]
                        variant = "primary" if is_correct else "stop"
                        btns.append(gr.update(variant=variant))
                    else:
                        btns.append(gr.update(variant="secondary"))
                return btns

            reveal_check.change(
                toggle_reveal,
                inputs=[reveal_check, hint_state],
                outputs=[hint_btn_1, hint_btn_2, hint_btn_3],
            )

            # Execute button — executes SQL + path feedback if command matches a hint
            _exec_inputs = [sql_input, playground_state, hint_state]
            _exec_outputs = [obs_log_display, metrics_display, step_display, reward_display, status_display, grader_display, playground_state, path_prompt, hint_btn_1, hint_btn_2, hint_btn_3, reveal_check, hint_state]
            execute_btn.click(do_execute, inputs=_exec_inputs, outputs=_exec_outputs)
            sql_input.submit(do_execute, inputs=_exec_inputs, outputs=_exec_outputs)

        # ── Tab 2: Demo Traces ────────────────────────────────────
        with gr.Tab("\u21AF Traces"):
            if not model_choices:
                gr.HTML('<div style="text-align:center;padding:40px;color:#000">No demo results available yet.</div>')
            else:
                with gr.Row():
                    trace_model = gr.Dropdown(choices=model_choices, label="Model", scale=2)
                    trace_task = gr.Dropdown(choices=task_choices, label="Task", scale=2)

                trace_display = gr.HTML(
                    '<div style="text-align:center;padding:40px;color:#000">Select a model and task to view the trace.</div>'
                )

                def show_trace(model_id, task_id):
                    if not model_id or not task_id:
                        return '<div style="color:#000;text-align:center;padding:20px">Select both a model and task.</div>'
                    data = all_results.get(model_id)
                    if not data:
                        return '<div style="color:#000;font-weight:700">Model results not found.</div>'
                    for r in data.get("results", []):
                        if r.get("task_id") == task_id:
                            return _trace_html(r)
                    return '<div style="color:#000;font-weight:700">Task not found in results.</div>'

                trace_model.change(show_trace, inputs=[trace_model, trace_task], outputs=[trace_display])
                trace_task.change(show_trace, inputs=[trace_model, trace_task], outputs=[trace_display])

        # ── Tab 3: Leaderboard ────────────────────────────────────
        with gr.Tab("\u265B Leaderboard"):
            gr.HTML('<h2>Model Comparison</h2>')
            gr.HTML(_leaderboard_html(all_results))

            gr.HTML('<h2 style="margin-top:24px">Score Heatmap</h2>')
            gr.HTML('<p style="color:#000;font-size:13px;font-weight:600;margin-bottom:12px">Scores by model × task. Green = high, red = low.</p>')
            gr.HTML(_heatmap_html(all_results))

            # Environment overview
            gr.HTML(f'''<div class="env-overview" style="margin-top:24px">
                <h3 style="margin:0 0 12px 0;color:#000;font-weight:900">Environment Overview</h3>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:16px;text-align:center">
                    <div class="metric-card"><div class="metric-value">17</div><div class="metric-label">Tasks</div></div>
                    <div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Difficulty Levels</div></div>
                    <div class="metric-card"><div class="metric-value">PG 16</div><div class="metric-label">PostgreSQL</div></div>
                    <div class="metric-card"><div class="metric-value">~20M</div><div class="metric-label">Database Rows</div></div>
                </div>
            </div>''')

        # ── Tab 4: Task Catalogue ──────────────────────────────────
        with gr.Tab("\u2699 Tasks"):
            gr.HTML(f'''<div class="env-overview" style="margin-bottom:16px">
                <h2 style="margin:0 0 8px 0;color:#000;font-weight:900">Task Catalogue</h2>
                <p style="color:#000;font-weight:600;font-size:14px">17 PostgreSQL incident scenarios across 3 difficulty levels. Each task presents a realistic alert and grades your diagnostic and resolution skills.</p>
            </div>''')
            gr.HTML(_task_descriptions_html())

    return demo
