#!/usr/bin/env python3
"""Build a standalone LenVM demo HTML from an output log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROW_RE = re.compile(
    r"^\s*(-?\d+)\s+(\d+)(\s+)(.*?)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(.+)$"
)

HTML_TEMPLATE = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LenVM Hover Demo</title>
  <style>
    :root {
      --panel: rgba(255, 252, 246, 0.88);
      --ink: #1f2933;
      --muted: #52606d;
      --curve: #0f6ab4;
      --truth: #c2410c;
      --active: #fde68a;
      --near: #fff7d6;
      --border: rgba(31, 41, 51, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 34%),
        linear-gradient(180deg, #faf7f0 0%, #efe9de 100%);
    }
    .page {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 24px 32px;
    }
    .hero { margin-bottom: 14px; }
    .eyebrow {
      font-size: 20px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 6px;
    }
    .subtitle {
      max-width: 900px;
      margin-top: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }
    .metadata {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .metadata-item {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.55);
      padding: 5px 10px;
      color: var(--muted);
      font-size: 12px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }
    .metadata-item strong { color: var(--ink); }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) 320px;
      gap: 18px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 18px 40px rgba(31, 41, 51, 0.08);
      backdrop-filter: blur(10px);
    }
    .chart-panel {
      padding: 14px;
      display: flex;
      flex-direction: column;
    }
    .chart-frame {
      position: relative;
      width: 100%;
      flex: 1;
      min-height: 300px;
      border-radius: 14px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(247, 242, 233, 0.92));
      border: 1px solid rgba(31, 41, 51, 0.08);
    }
    #chart { width: 100%; height: 100%; display: block; }
    .chart-note {
      margin: 8px 2px 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .middle-note {
      margin: 12px 4px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .td-stats-panel {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
      padding: 12px 14px;
    }
    .td-stat { border-right: 1px solid rgba(31, 41, 51, 0.08); }
    .td-stat:last-child { border-right: 0; }
    .td-stat-label { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .td-stat-value {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 15px;
      line-height: 1.3;
      word-break: break-word;
    }
    .side-panel { padding: 16px; position: sticky; top: 14px; min-height: 100%; }
    .side-title {
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .metric {
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(31, 41, 51, 0.08);
    }
    .metric:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 4px; }
    .value {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 16px;
      line-height: 1.35;
      word-break: break-word;
    }
    .text-panel { margin-top: 0; padding: 16px; }
    .text-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .text-title { margin: 0; font-size: 20px; }
    .text-subtitle { color: var(--muted); font-size: 13px; line-height: 1.4; }
    .question-static {
      margin: 0 0 12px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.5);
      border: 1px solid rgba(31, 41, 51, 0.08);
      overflow: hidden;
    }
    .question-header {
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .question-header:hover {
      background: rgba(31, 41, 51, 0.03);
    }
    .question-toggle {
      font-size: 11px;
      transition: transform 0.2s ease;
    }
    .question-toggle.collapsed {
      transform: rotate(-90deg);
    }
    .question-content {
      padding: 0 14px 12px;
      font-size: 14px;
      line-height: 1.6;
      color: var(--ink);
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 500px;
      overflow: auto;
      transition: max-height 0.3s ease, padding 0.3s ease;
    }
    .question-content.collapsed {
      max-height: 0;
      padding: 0 14px;
      overflow: hidden;
    }
    .text-box {
      max-height: 32vh;
      overflow: auto;
      padding: 14px;
      border-radius: 12px;
      border: 1px solid rgba(31, 41, 51, 0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,246,239,0.95));
      font-size: 14px;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
      scroll-behavior: auto;
    }
    .token {
      border-radius: 5px;
      transition: background-color 90ms ease, color 90ms ease;
      cursor: default;
    }
    .token.outlier-positive { box-shadow: inset 0 -3px 0 rgba(220, 38, 38, 0.75); }
    .token.outlier-negative { box-shadow: inset 0 -3px 0 rgba(37, 99, 235, 0.75); }
    .token.outlier-run { outline: 1px solid rgba(124, 58, 237, 0.45); }
    .token.near { background: var(--near); }
    .token.active { background: var(--active); color: #111827; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .side-panel { position: static; }
      .text-box { max-height: 44vh; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">LenVM Output Demo</div>
      <div class="subtitle">
        The blue curve is the predicted remaining-step length, the dashed orange line is the ground truth,
        and the text panel below follows the same token sequence from the log.
      </div>
      <div class="metadata" id="metadataPanel"></div>
      <div class="subtitle"><span style="color:#dc2626;font-weight:700;">Red underline</span>: positive TD outliers, <span style="color:#2563eb;font-weight:700;">blue underline</span>: negative TD outliers, and <span style="color:#7c3aed;font-weight:700;">purple outline</span>: consecutive outliers.</div>
    </section>

    <section class="layout">
      <div class="panel chart-panel">
        <div class="chart-frame"><canvas id="chart"></canvas></div>
        <div class="chart-note">X axis is remaining steps and is shown in descending order, so the ground-truth line runs from upper left to lower right.</div>
      </div>
      <aside class="panel side-panel">
        <div class="side-title">Selected Point</div>
        <div class="metric"><div class="label">Remaining steps</div><div class="value" id="remainingValue">-</div></div>
        <div class="metric"><div class="label">Predicted steps</div><div class="value" id="predictedValue">-</div></div>
        <div class="metric"><div class="label">TD error</div><div class="value" id="tdErrorValue">-</div><div class="label">Positive means this token increases expected remaining length relative to the previous step.</div></div>
        <div class="metric"><div class="label">Step index</div><div class="value" id="stepValue">-</div></div>
        <div class="metric"><div class="label">Token</div><div class="value" id="tokenValue">-</div></div>
      </aside>
    </section>

    <section class="panel td-stats-panel">
      <div class="td-stat"><div class="td-stat-label">Mean TD error</div><div class="td-stat-value" id="tdMeanValue">-</div></div>
      <div class="td-stat"><div class="td-stat-label">Mean |TD error|</div><div class="td-stat-value" id="tdAbsMeanValue">-</div></div>
      <div class="td-stat"><div class="td-stat-label">Positive threshold</div><div class="td-stat-value" id="tdPosThresholdValue">-</div></div>
      <div class="td-stat"><div class="td-stat-label">Negative threshold</div><div class="td-stat-value" id="tdNegThresholdValue">-</div></div>
      <div class="td-stat"><div class="td-stat-label">Outliers</div><div class="td-stat-value" id="tdOutlierCountValue">-</div></div>
    </section>

    <div class="middle-note"><strong>Click</strong> the curve to jump to the corresponding token below.</div>

    <section class="panel text-panel">
      <div class="text-head">
        <h2 class="text-title">Token Text</h2>
        <div class="text-subtitle">Click the chart to locate the matching token below.</div>
      </div>
      <div class="question-static">
        <div class="question-header" id="questionHeader">
          <span class="question-toggle collapsed">▼</span>
          <span>Question</span>
        </div>
        <div class="question-content collapsed" id="questionContent"></div>
      </div>
      <div class="text-box" id="textBox"></div>
    </section>
  </div>

  <script>
    const demoData = __DATA__;
    const canvas = document.getElementById("chart");
    const textBox = document.getElementById("textBox");
    const questionContent = document.getElementById("questionContent");
    const questionHeader = document.getElementById("questionHeader");
    const questionToggle = questionHeader.querySelector(".question-toggle");
    const metadataPanel = document.getElementById("metadataPanel");
    const tdMeanValue = document.getElementById("tdMeanValue");
    const tdAbsMeanValue = document.getElementById("tdAbsMeanValue");
    const tdPosThresholdValue = document.getElementById("tdPosThresholdValue");
    const tdNegThresholdValue = document.getElementById("tdNegThresholdValue");
    const tdOutlierCountValue = document.getElementById("tdOutlierCountValue");
    const remainingValue = document.getElementById("remainingValue");
    const predictedValue = document.getElementById("predictedValue");
    const tdErrorValue = document.getElementById("tdErrorValue");
    const stepValue = document.getElementById("stepValue");
    const tokenValue = document.getElementById("tokenValue");
    const tokens = demoData.points;
    const tokenNodes = [];
    let activeIndex = 0;

    function fmtNumber(value) {
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: value % 1 === 0 ? 0 : 2 });
    }

    function renderMetadata() {
      const metadata = demoData.metadata || {};
      const entries = [
        ["Base generator", metadata.base_generator],
        ["LenVM", metadata.lenvm],
      ].filter(([, value]) => value);
      entries.forEach(([label, value]) => {
        const item = document.createElement("span");
        item.className = "metadata-item";
        item.innerHTML = `<strong>${label}:</strong> ${String(value)}`;
        metadataPanel.appendChild(item);
      });
    }

    function renderTdStats() {
      const stats = demoData.td_stats || {};
      tdMeanValue.textContent = fmtNumber(stats.mean || 0);
      tdAbsMeanValue.textContent = fmtNumber(stats.abs_mean || 0);
      tdPosThresholdValue.textContent = fmtNumber(stats.positive_threshold || 0);
      tdNegThresholdValue.textContent = fmtNumber(stats.negative_threshold || 0);
      tdOutlierCountValue.textContent = `${stats.outlier_count || 0} / ${stats.count || 0}`;
    }

    function renderText() {
      const label = document.createElement("span");
      label.style.cssText = "display:block;font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;pointer-events:none;";
      label.textContent = "Answer";
      textBox.appendChild(label);
      const frag = document.createDocumentFragment();
      tokens.forEach((point, index) => {
        const span = document.createElement("span");
        span.className = "token";
        if (point.td_outlier) span.classList.add(point.td_err >= 0 ? "outlier-positive" : "outlier-negative");
        if (point.td_outlier_run) span.classList.add("outlier-run");
        span.dataset.index = String(index);
        span.textContent = point.display_token;
        span.addEventListener("mouseenter", () => setActive(index, false));
        span.addEventListener("click", () => setActive(index, true));
        frag.appendChild(span);
        tokenNodes.push(span);
      });
      textBox.appendChild(frag);
    }

    function getChartRect() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const dpr = Math.max(window.devicePixelRatio || 1, 1);
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width, height, padLeft: 58, padRight: 24, padTop: 24, padBottom: 40 };
    }

    function makeScales(rect) {
      const minTrue = demoData.meta.true_min;
      const maxTrue = demoData.meta.true_max;
      const minPred = demoData.meta.pred_min;
      const maxPred = demoData.meta.pred_max;
      const plotWidth = rect.width - rect.padLeft - rect.padRight;
      const plotHeight = rect.height - rect.padTop - rect.padBottom;
      const xScale = value => rect.padLeft + ((maxTrue - value) / (maxTrue - minTrue || 1)) * plotWidth;
      const yScale = value => rect.padTop + (1 - (value - minPred) / (maxPred - minPred || 1)) * plotHeight;
      return { minTrue, maxTrue, minPred, maxPred, plotWidth, plotHeight, xScale, yScale };
    }

    function drawAxes(ctx, rect, scales) {
      ctx.strokeStyle = "rgba(31, 41, 51, 0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(rect.padLeft, rect.padTop);
      ctx.lineTo(rect.padLeft, rect.height - rect.padBottom);
      ctx.lineTo(rect.width - rect.padRight, rect.height - rect.padBottom);
      ctx.stroke();

      ctx.fillStyle = "#52606d";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Remaining steps", rect.padLeft + scales.plotWidth / 2, rect.height - 10);

      ctx.save();
      ctx.translate(15, rect.padTop + scales.plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Predicted steps", 0, 0);
      ctx.restore();

      const ticks = 5;
      for (let i = 0; i <= ticks; i += 1) {
        const frac = i / ticks;
        const trueValue = scales.maxTrue - frac * (scales.maxTrue - scales.minTrue);
        const x = scales.xScale(trueValue);
        ctx.strokeStyle = "rgba(31, 41, 51, 0.08)";
        ctx.beginPath();
        ctx.moveTo(x, rect.padTop);
        ctx.lineTo(x, rect.height - rect.padBottom);
        ctx.stroke();
        ctx.fillStyle = "#52606d";
        ctx.textAlign = "center";
        ctx.fillText(Math.round(trueValue).toString(), x, rect.height - rect.padBottom + 17);
      }

      for (let i = 0; i <= ticks; i += 1) {
        const frac = i / ticks;
        const predValue = scales.minPred + frac * (scales.maxPred - scales.minPred);
        const y = scales.yScale(predValue);
        ctx.strokeStyle = "rgba(31, 41, 51, 0.08)";
        ctx.beginPath();
        ctx.moveTo(rect.padLeft, y);
        ctx.lineTo(rect.width - rect.padRight, y);
        ctx.stroke();
        ctx.fillStyle = "#52606d";
        ctx.textAlign = "right";
        ctx.fillText(Math.round(predValue).toString(), rect.padLeft - 8, y + 4);
      }
    }

    function drawCurve(ctx, scales, accessor, style) {
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = style.width;
      ctx.setLineDash(style.dash || []);
      ctx.beginPath();
      tokens.forEach((point, index) => {
        const x = scales.xScale(point.true);
        const y = scales.yScale(accessor(point));
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawMarker(ctx, scales, point, rect) {
      const x = scales.xScale(point.true);
      const y = scales.yScale(point.len_pred);
      ctx.strokeStyle = "rgba(15, 118, 110, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, rect.padTop);
      ctx.lineTo(x, rect.height - rect.padBottom);
      ctx.stroke();
      ctx.fillStyle = "#0f766e";
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    function redraw() {
      const rect = getChartRect();
      const scales = makeScales(rect);
      const ctx = rect.ctx;
      ctx.clearRect(0, 0, rect.width, rect.height);
      drawAxes(ctx, rect, scales);
      drawCurve(ctx, scales, point => point.true, {
        stroke: getComputedStyle(document.documentElement).getPropertyValue("--truth").trim(),
        width: 2.2,
        dash: [8, 6],
      });
      drawCurve(ctx, scales, point => point.len_pred, {
        stroke: getComputedStyle(document.documentElement).getPropertyValue("--curve").trim(),
        width: 2.4,
      });
      drawMarker(ctx, scales, tokens[activeIndex], rect);
    }

    function updateSidebar(point) {
      remainingValue.textContent = fmtNumber(point.true);
      predictedValue.textContent = fmtNumber(point.len_pred);
      tdErrorValue.textContent = fmtNumber(point.td_err || 0);
      stepValue.textContent = fmtNumber(point.step);
      tokenValue.textContent = JSON.stringify(point.display_token);
    }

    function clearHighlights() {
      tokenNodes.forEach((node, index) => {
        const point = tokens[index];
        node.className = "token";
        if (point.td_outlier) node.classList.add(point.td_err >= 0 ? "outlier-positive" : "outlier-negative");
        if (point.td_outlier_run) node.classList.add("outlier-run");
      });
    }

    function highlightToken(index, shouldScroll) {
      clearHighlights();
      for (let offset = -6; offset <= 6; offset += 1) {
        const target = tokenNodes[index + offset];
        if (target) target.classList.add("near");
      }
      const activeNode = tokenNodes[index];
      if (!activeNode) return;
      activeNode.classList.remove("near");
      activeNode.classList.add("active");
      if (!shouldScroll) return;

      const boxRect = textBox.getBoundingClientRect();
      const nodeRect = activeNode.getBoundingClientRect();
      const relativeTop = nodeRect.top - boxRect.top;
      const relativeBottom = nodeRect.bottom - boxRect.top;
      const boxHeight = textBox.clientHeight;
      if (relativeTop < 0 || relativeBottom > boxHeight) {
        const targetScroll = textBox.scrollTop + relativeTop - boxHeight / 2 + nodeRect.height / 2;
        textBox.scrollTo({ top: targetScroll, behavior: "auto" });
      }
    }

    function setActive(index, shouldScroll = false) {
      activeIndex = Math.max(0, Math.min(tokens.length - 1, index));
      const point = tokens[activeIndex];
      updateSidebar(point);
      highlightToken(activeIndex, shouldScroll);
      redraw();
    }

    function nearestIndexForEvent(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const width = canvas.clientWidth;
      let bestIndex = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      tokens.forEach((point, index) => {
        const pointX = 58 + ((demoData.meta.true_max - point.true) / (demoData.meta.true_max - demoData.meta.true_min || 1)) * (width - 58 - 24);
        const dist = Math.abs(pointX - x);
        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = index;
        }
      });
      return bestIndex;
    }

    canvas.addEventListener("click", event => {
      const nextIndex = nearestIndexForEvent(event);
      setActive(nextIndex, true);
    });

    window.addEventListener("resize", redraw);
    // Render question with toggle
    questionContent.textContent = demoData.question || "";
    questionHeader.addEventListener("click", () => {
      const collapsed = questionContent.classList.toggle("collapsed");
      questionToggle.classList.toggle("collapsed", collapsed);
    });
    renderMetadata();
    renderTdStats();
    renderText();
    setActive(Math.min(120, tokens.length - 1), false);
  </script>
</body>
</html>
'''


def parse_log(text: str) -> tuple[str, dict[str, str], list[dict[str, object]]]:
    lines = text.splitlines()

    metadata: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        for key in ("base_generator", "lenvm", "sample_file"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                metadata[key] = stripped[len(prefix):].strip()

    # Extract question text from USER section
    question_lines: list[str] = []
    in_user_section = False
    for line in lines:
        if line.strip() == "#### USER":
            in_user_section = True
            continue
        if in_user_section:
            if line.strip() == "---":
                break
            question_lines.append(line)
    question = "\n".join(question_lines).strip()

    # Parse table data
    points: list[dict[str, object]] = []
    for line in lines:
        match = ROW_RE.match(line)
        if match is None:
            continue

        separator = match.group(3)
        base_token = match.group(4)
        if base_token == "":
            raw_token = ""
        else:
            leading_spaces = max(0, len(separator) - 2)
            raw_token = (" " * leading_spaces) + base_token

        points.append(
            {
                "step": int(match.group(1)),
                "token_id": int(match.group(2)),
                "raw_token": raw_token,
                "display_token": raw_token.replace("\\n", "\n").replace("\\t", "\t"),
                "len_pred": round(float(match.group(7)), 3),
                "td_err": round(float(match.group(8)), 6),
                "true": int(match.group(9)),
            }
        )

    if not points:
        raise ValueError("No table rows found in the provided output log.")
    points.sort(key=lambda item: item["true"], reverse=True)
    return question, metadata, points


def mark_td_outliers(points: list[dict[str, object]]) -> dict[str, float | int]:
    td_signed = [float(point.get("td_err", 0.0)) for point in points if int(point.get("step", 0)) >= 0]
    if not td_signed:
        return {
            "count": 0,
            "mean": 0.0,
            "abs_mean": 0.0,
            "positive_threshold": 0.0,
            "negative_threshold": 0.0,
            "outlier_count": 0,
        }

    positive_values = sorted(value for value in td_signed if value > 0)
    negative_values = sorted(value for value in td_signed if value < 0)
    positive_threshold = positive_values[max(0, int(0.95 * (len(positive_values) - 1)))] if positive_values else float("inf")
    negative_threshold = negative_values[min(len(negative_values) - 1, int(0.05 * (len(negative_values) - 1)))] if negative_values else float("-inf")
    if positive_values:
        positive_threshold = max(positive_threshold, 0.01)
    if negative_values:
        negative_threshold = min(negative_threshold, -0.01)

    previous_outlier = False
    outlier_count = 0
    for point in points:
        td_err = float(point.get("td_err", 0.0))
        is_outlier = int(point.get("step", 0)) >= 0 and (
            (td_err > 0 and td_err >= positive_threshold) or (td_err < 0 and td_err <= negative_threshold)
        )
        point["td_outlier"] = is_outlier
        point["td_outlier_run"] = is_outlier and previous_outlier
        previous_outlier = is_outlier
        if is_outlier:
            outlier_count += 1

    return {
        "count": len(td_signed),
        "mean": sum(td_signed) / len(td_signed),
        "abs_mean": sum(abs(value) for value in td_signed) / len(td_signed),
        "positive_threshold": positive_threshold if positive_values else 0.0,
        "negative_threshold": negative_threshold if negative_values else 0.0,
        "outlier_count": outlier_count,
    }


def build_html(question: str, metadata: dict[str, str], points: list[dict[str, object]]) -> str:
    td_stats = mark_td_outliers(points)
    payload = {
        "question": question,
        "metadata": metadata,
        "td_stats": td_stats,
        "meta": {
            "true_min": min(point["true"] for point in points),
            "true_max": max(point["true"] for point in points),
            "pred_min": min(point["len_pred"] for point in points),
            "pred_max": max(point["len_pred"] for point in points),
            "count": len(points),
        },
        "points": points,
    }
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to output.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output_hover_demo.html"),
        help="Output HTML file path",
    )
    args = parser.parse_args()

    question, metadata, points = parse_log(args.input.read_text(encoding="utf-8"))
    args.output.write_text(build_html(question, metadata, points), encoding="utf-8")
    print(f"Wrote {args.output} with {len(points)} points.")


if __name__ == "__main__":
    main()