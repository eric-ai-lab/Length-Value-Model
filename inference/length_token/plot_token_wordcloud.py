from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


def load_items(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list JSON: {path}")
    return data


def choose_default_weight_key(path: Path) -> str:
    return "count_total"


def _looks_like_emoji_or_symbol(s: str) -> bool:
    for ch in s:
        cat = unicodedata.category(ch)
        if cat in {"So", "Sk"}:
            return True
        cp = ord(ch)
        if 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or cp == 0xFE0F:
            return True
    return False


def _emoji_token_alias(s: str) -> str:
    skip = {"white", "black", "heavy", "medium", "small", "emoji", "variation", "selector"}
    parts: List[str] = []
    for ch in s:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = f"u{ord(ch):04x}"
        words = [w for w in name.lower().replace("-", " ").split() if w not in skip]
        if not words:
            words = [f"u{ord(ch):04x}"]
        parts.append(words[0])
    if not parts:
        return "emoji"
    return "emoji_" + "_".join(parts[:1])


def auto_pick_font(items: List[dict]) -> str | None:
    # Prefer a font that can actually draw emoji/symbol tokens like "✅".
    symbol_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    text_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    candidates = text_candidates
    for it in items:
        tok = str(it.get("token_text", ""))
        if tok and _looks_like_emoji_or_symbol(tok):
            candidates = symbol_candidates + text_candidates
            break
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def build_freq(
    items: List[dict],
    weight_key: str,
    top_k: int,
    min_weight: float,
    min_count_total: int,
    max_count_total: int,
    exclude_regex: str,
    keep_whitespace_token: bool,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pat = re.compile(exclude_regex) if exclude_regex else None
    picked = 0

    def _normalize_token_text(s: str) -> str:
        # WordCloud cannot render multiline "words".
        return s.replace("\r", "\\r").replace("\n", "\\n")

    def _visualize_leading_spaces(s: str) -> str:
        # Make leading spaces visible in the rendered wordcloud.
        n = len(s) - len(s.lstrip(" "))
        if n <= 0:
            return s
        return ("_" * n) + s[n:]

    def _visualize_token(s: str) -> str:
        if _looks_like_emoji_or_symbol(s) and not any(ch.isalnum() for ch in s):
            return _emoji_token_alias(s)
        return s

    for it in items:
        tok = _normalize_token_text(str(it.get("token_text", "")))
        if not tok:
            continue
        if not keep_whitespace_token and tok.strip() == "":
            continue
        if pat is not None and pat.search(tok):
            continue
        if int(it.get("count_total", 0)) < int(min_count_total):
            continue
        tok = _visualize_leading_spaces(tok)
        tok = _visualize_token(tok)
        if weight_key == "count_total":
            w = float(min(int(it.get("count_total", 0)), int(max_count_total)))
        else:
            w = float(it.get(weight_key, 0.0))
            if weight_key == "mean_score":
                w = abs(w)
        if w < min_weight:
            continue
        out[tok] = out.get(tok, 0.0) + w
        picked += 1
        if top_k > 0 and picked >= top_k:
            break
    return out


def render_wordcloud(freq: Dict[str, float], out_png: Path, title: str, font_path: str | None) -> None:
    if not freq:
        raise RuntimeError(f"No tokens to draw for {out_png.name}")
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `wordcloud`. Install with: pip install wordcloud"
        ) from e

    wc = WordCloud(
        width=2200,
        height=1200,
        background_color="white",
        collocations=False,
        max_words=max(200, len(freq)),
        font_path=font_path,
    ).generate_from_frequencies(freq)
    wc.to_file(str(out_png))

    out_pdf = out_png.with_suffix(".pdf")
    wc.to_image().save(str(out_pdf), "PDF", resolution=300.0)

    # Sidecar metadata for traceability.
    meta_path = out_png.with_suffix(".meta.json")
    meta = {
        "title": title,
        "num_words": len(freq),
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate token-frequency wordclouds from top token JSON files."
    )
    ap.add_argument("--above-json", type=str, required=True)
    ap.add_argument("--below-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=200, help="Use at most top-k rows from each file.")
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument(
        "--min-count-total",
        type=int,
        default=1,
        help="Only keep tokens whose count_total is at least this value.",
    )
    ap.add_argument(
        "--max-count-total",
        type=int,
        default=50,
        help="Cap count_total at this value before drawing sizes.",
    )
    ap.add_argument(
        "--exclude-regex",
        type=str,
        default=r"^<\|.*\|>$",
        help="Regex for tokens to exclude, default filters special tokens like <|im_end|>.",
    )
    ap.add_argument("--keep-whitespace-token", action="store_true")
    ap.add_argument("--font-path", type=str, default=None, help="Optional font path for CJK rendering.")
    args = ap.parse_args()

    above_path = Path(args.above_json)
    below_path = Path(args.below_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    above_items = load_items(above_path)
    below_items = load_items(below_path)

    above_key = choose_default_weight_key(above_path)
    below_key = choose_default_weight_key(below_path)

    above_freq = build_freq(
        items=above_items,
        weight_key=above_key,
        top_k=int(args.top_k),
        min_weight=float(args.min_weight),
        min_count_total=int(args.min_count_total),
        max_count_total=int(args.max_count_total),
        exclude_regex=str(args.exclude_regex),
        keep_whitespace_token=bool(args.keep_whitespace_token),
    )
    below_freq = build_freq(
        items=below_items,
        weight_key=below_key,
        top_k=int(args.top_k),
        min_weight=float(args.min_weight),
        min_count_total=int(args.min_count_total),
        max_count_total=int(args.max_count_total),
        exclude_regex=str(args.exclude_regex),
        keep_whitespace_token=bool(args.keep_whitespace_token),
    )

    above_png = out_dir / (above_path.stem + ".wordcloud.png")
    below_png = out_dir / (below_path.stem + ".wordcloud.png")
    font_path = args.font_path or auto_pick_font(above_items + below_items)
    render_wordcloud(above_freq, above_png, "above", font_path=font_path)
    render_wordcloud(below_freq, below_png, "below", font_path=font_path)

    print(
        json.dumps(
            {
                "above_png": str(above_png),
                "below_png": str(below_png),
                "above_words": len(above_freq),
                "below_words": len(below_freq),
                "above_weight_key": above_key,
                "below_weight_key": below_key,
                "font_path": font_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

