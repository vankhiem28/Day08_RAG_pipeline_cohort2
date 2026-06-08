"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import subprocess
from pathlib import Path

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_with_textutil(filepath: Path) -> str | None:
    """Fallback for DOC/DOCX on macOS when MarkItDown is not installed."""
    if filepath.suffix.lower() not in (".doc", ".docx"):
        return None

    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(filepath)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def convert_document(filepath: Path, md: "MarkItDown | None") -> str:
    """Convert a legal document to text without falling back to raw binary reads."""
    if md:
        try:
            content = md.convert(str(filepath)).text_content.strip()
            if content:
                return content
        except Exception as exc:
            print(f"  ! MarkItDown failed for {filepath.name}: {exc}")

    textutil_content = convert_with_textutil(filepath)
    if textutil_content:
        return textutil_content

    if filepath.suffix.lower() == ".pdf":
        return (
            f"> Could not extract text from `{filepath.name}`.\n\n"
            "This PDF needs a PDF text extractor such as MarkItDown/PyPDF, "
            "or OCR if it is a scanned image PDF."
        )

    return (
        f"> Could not convert `{filepath.name}`.\n\n"
        "Install MarkItDown or another document converter, then run this task again."
    )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown() if MarkItDown else None

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            content = convert_document(filepath, md)
            header = f"# {filepath.stem.replace('-', ' ').title()}\n\n"
            output_path.write_text(header + content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + data.get("content_markdown", data.get("content", ""))
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
