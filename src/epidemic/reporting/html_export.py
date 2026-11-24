"""Export HTML du rapport, et PDF via le navigateur s'il est installé."""

import subprocess
from pathlib import Path


def markdown_to_html(markdown: str, title: str) -> str:
    """Convertit le sous-ensemble de markdown utilisé par le rapport."""
    body = _blocks(_strip_front_matter(markdown))
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }}
}};
</script>
<script
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
  async></script>
<style>
body {{
  font-family: "Segoe UI", sans-serif;
  max-width: 48rem;
  margin: 2rem auto;
  line-height: 1.5;
  color: #1c1c1c;
  padding: 0 1rem;
}}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; }}
code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
pre {{ background: #f4f4f4; padding: 0.8rem; overflow: auto; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def write_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Imprime le HTML en PDF avec Edge ou Chrome. False si aucun des deux n'est là."""
    browser = _browser()
    if browser is None:
        return False
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=False, timeout=90, capture_output=True)
    return pdf_path.is_file() and pdf_path.stat().st_size > 0


def _browser() -> Path | None:
    candidates = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _strip_front_matter(markdown: str) -> str:
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            return markdown[end + 4 :]
    return markdown


def _blocks(markdown: str) -> str:
    lines = markdown.splitlines()
    html: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            index, block = _fence(lines, index)
            html.append(block)
            continue
        if line.startswith("|"):
            index, block = _table(lines, index)
            html.append(block)
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            html.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
        elif line.startswith("!["):
            html.append(_image(line))
        elif line.strip() == "":
            pass
        else:
            paragraph = [line]
            index += 1
            while (
                index < len(lines)
                and lines[index].strip()
                and not _starts_block(lines[index])
            ):
                paragraph.append(lines[index])
                index += 1
            html.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
            continue
        index += 1
    return "\n".join(html)


def _starts_block(line: str) -> bool:
    return (
        line.startswith("#")
        or line.startswith("|")
        or line.startswith("```")
        or line.startswith("![")
    )


def _fence(lines: list[str], start: int) -> tuple[int, str]:
    index = start + 1
    body: list[str] = []
    while index < len(lines) and not lines[index].startswith("```"):
        body.append(_escape(lines[index]))
        index += 1
    return index + 1, "<pre><code>" + "\n".join(body) + "</code></pre>"


def _table(lines: list[str], start: int) -> tuple[int, str]:
    rows = []
    index = start
    while index < len(lines) and lines[index].startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
        if not all(set(cell) <= set("-: ") for cell in cells):
            rows.append(cells)
        index += 1
    head = rows[0]
    body = rows[1:]
    header = "".join(f"<th>{_inline(cell)}</th>" for cell in head)
    rendered = [f"<tr>{header}</tr>"]
    for row in body:
        rendered.append(
            "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
        )
    return index, "<table>" + "".join(rendered) + "</table>"


def _image(line: str) -> str:
    alt = line.split("]", 1)[0][2:]
    src = line.split("(", 1)[1].rstrip(")")
    return f'<img alt="{_escape(alt)}" src="{_escape(src)}">'


def _inline(text: str) -> str:
    escaped = _escape(text)
    parts = escaped.split("`")
    merged = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            merged.append(f"<code>{part}</code>")
        else:
            bold = part.split("**")
            chunk = []
            for bold_index, piece in enumerate(bold):
                if bold_index % 2 == 1:
                    chunk.append(f"<strong>{piece}</strong>")
                else:
                    chunk.append(piece)
            merged.append("".join(chunk))
    return "".join(merged)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
