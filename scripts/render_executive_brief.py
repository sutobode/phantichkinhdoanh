"""Render the executive brief with an installed Chromium browser."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "executive_brief.md"
DEFAULT_OUTPUT = ROOT / "docs" / "executive_brief.pdf"


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    body: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            body.append(f"<p>{inline_markup(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            items = "".join(f"<li>{inline_markup(item)}</li>" for item in list_items)
            body.append(f"<ul>{items}</ul>")
            list_items.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush_paragraph()
            flush_list()
            index += 1
            continue

        if line.startswith("|" ) and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if separator.startswith("|") and re.fullmatch(r"[|:\- ]+", separator):
                flush_paragraph()
                flush_list()
                headers = [cell.strip() for cell in line.strip("|").split("|")]
                index += 2
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(
                        [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    )
                    index += 1
                header_html = "".join(f"<th>{inline_markup(cell)}</th>" for cell in headers)
                row_html = "".join(
                    "<tr>" + "".join(f"<td>{inline_markup(cell)}</td>" for cell in row) + "</tr>"
                    for row in rows
                )
                body.append(f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>")
                continue

        if line.startswith("# "):
            flush_paragraph()
            flush_list()
            body.append(f"<h1>{inline_markup(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_paragraph()
            flush_list()
            heading = line[3:]
            heading_id = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            body.append(f'<h2 id="{heading_id}">{inline_markup(heading)}</h2>')
        elif line.startswith("- "):
            flush_paragraph()
            list_items.append(line[2:])
        else:
            flush_list()
            paragraph.append(line)
        index += 1

    flush_paragraph()
    flush_list()
    content = "\n".join(body)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Executive Brief - Customer Retention and Promotion Targeting</title>
<style>
  @page {{ size: A4; margin: 15mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    color: #17202a;
    font: 10.5pt/1.28 Arial, Helvetica, sans-serif;
    margin: 0;
  }}
  h1 {{
    border-bottom: 2px solid #1f5a69;
    color: #173f4a;
    font-size: 19pt;
    line-height: 1.12;
    margin: 0 0 8pt;
    padding-bottom: 5pt;
  }}
  h2 {{
    color: #1f5a69;
    font-size: 13pt;
    line-height: 1.15;
    margin: 10pt 0 4pt;
    page-break-after: avoid;
  }}
  #sensitivity-and-decision-gate {{ page-break-before: always; }}
  p {{ margin: 0 0 5.5pt; orphans: 3; widows: 3; }}
  ul {{ margin: 2pt 0 6pt; padding-left: 17pt; }}
  li {{ margin: 0 0 2pt; }}
  table {{
    border-collapse: collapse;
    font-size: 9.1pt;
    margin: 4pt 0 6pt;
    page-break-inside: avoid;
    width: 100%;
  }}
  th, td {{ border: 1px solid #9aa9ad; padding: 3.5pt 4pt; vertical-align: top; }}
  th {{ background: #dfecef; color: #173f4a; font-weight: 700; text-align: left; }}
  th:not(:first-child), td:not(:first-child) {{ text-align: right; }}
  code {{ background: #eef2f3; font: 9.2pt Consolas, monospace; padding: 0 2pt; }}
  strong {{ color: #111820; }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def find_browser() -> Path:
    configured = os.environ.get("BROWSER_PATH")
    candidates = [
        configured,
        shutil.which("msedge"),
        shutil.which("chrome"),
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("No Chromium browser found; set BROWSER_PATH to its executable.")


def pdf_page_count(pdf_path: Path) -> int:
    data = pdf_path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
        raise ValueError(f"Invalid PDF output: {pdf_path}")
    count = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
    if count < 1:
        raise ValueError("Could not identify any PDF page objects.")
    return count


def render(source: Path, output: Path) -> int:
    browser = find_browser()
    rendered_html = markdown_to_html(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="executive-brief-") as temporary:
        temporary_path = Path(temporary)
        html_path = temporary_path / "executive_brief.html"
        html_path.write_text(rendered_html, encoding="utf-8")
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={temporary_path / 'browser-profile'}",
            f"--print-to-pdf={output.resolve()}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return pdf_page_count(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pages = render(args.source.resolve(), args.output.resolve())
    if pages > 2:
        raise SystemExit(f"Rendered {pages} pages; executive brief must not exceed 2 pages.")
    print(f"Rendered {args.output.resolve()} ({pages} pages)")


if __name__ == "__main__":
    main()
