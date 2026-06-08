"""Domain-aware chunking for legal, news, and Q&A documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, STANDARDIZED_DIR
from ..models import ChunkRecord, DocumentRecord
from .cleaner_service import CleanerService
from .text_utils import stable_id


class ChunkingService:
    """Create chunks while preserving original content and metadata."""

    LEGAL_ARTICLE_RE = re.compile(r"(?=^Điều\s+\d+[a-zA-Z]?\.)", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        standardized_dir: Path = STANDARDIZED_DIR,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.standardized_dir = standardized_dir
        self.cleaner = CleanerService()

    def load_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        if not self.standardized_dir.exists():
            return documents

        for md_file in sorted(self.standardized_dir.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            doc_type = self._infer_type(md_file)
            metadata = self._metadata_from_markdown(md_file, content, doc_type)
            document_id = stable_id(str(md_file.relative_to(self.standardized_dir)), content[:200])
            documents.append(
                DocumentRecord(
                    document_id=document_id,
                    content=self.cleaner.clean_text(content),
                    metadata=metadata,
                ).to_dict()
            )
        return documents

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for document in documents:
            record = DocumentRecord(
                document_id=document["document_id"],
                content=document["content"],
                metadata=document.get("metadata", {}),
            )
            chunks.extend(chunk.to_dict() for chunk in self.chunk_document(record))
        return chunks

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        doc_type = document.metadata.get("type", "unknown")
        if doc_type == "legal":
            raw_parts = self._split_legal(document.content)
        elif doc_type == "qa":
            raw_parts = self._split_qa(document.content)
        else:
            raw_parts = self._split_news(document.content)

        chunks: list[ChunkRecord] = []
        chunk_index = 0
        for part in raw_parts:
            for bounded in self._split_to_size(part):
                metadata = dict(document.metadata)
                metadata["chunk_index"] = chunk_index
                article_match = re.search(r"Điều\s+\d+[a-zA-Z]?", bounded)
                if article_match:
                    metadata["article"] = article_match.group(0)
                chunk_id = stable_id(document.document_id, str(chunk_index), bounded[:80])
                embedding_text = self.cleaner.build_embedding_text(bounded, metadata)
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        content=bounded,
                        embedding_text=embedding_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
        return chunks

    def _split_legal(self, content: str) -> list[str]:
        parts = [part.strip() for part in self.LEGAL_ARTICLE_RE.split(content) if part.strip()]
        if len(parts) <= 1:
            return self._split_news(content)
        if parts[0].startswith("#") and len(parts[0]) < 300:
            parts[1] = f"{parts[0]}\n\n{parts[1]}"
            parts = parts[1:]
        return parts

    def _split_news(self, content: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content) if p.strip()]
        if not paragraphs:
            return [content.strip()] if content.strip() else []

        parts: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = paragraph
        if current:
            parts.append(current)
        return parts

    def _split_qa(self, content: str) -> list[str]:
        blocks = re.split(r"(?=^Q:|^Câu hỏi:)", content, flags=re.MULTILINE)
        blocks = [block.strip() for block in blocks if block.strip()]
        return blocks or self._split_news(content)

    def _split_to_size(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + int(self.chunk_size * 0.45):
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _infer_type(self, path: Path) -> str:
        parts = set(path.relative_to(self.standardized_dir).parts)
        if "legal" in parts:
            return "legal"
        if "news" in parts:
            return "news"
        if "qa" in parts:
            return "qa"
        return path.parent.name or "unknown"

    def _metadata_from_markdown(self, path: Path, content: str, doc_type: str) -> dict[str, Any]:
        rel_path = str(path.relative_to(self.standardized_dir))
        title = path.stem.replace("-", " ")
        first_heading = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if first_heading:
            title = first_heading.group(1).strip()

        metadata: dict[str, Any] = {
            "source": path.name,
            "path": rel_path,
            "type": doc_type,
            "title": title,
        }
        source_match = re.search(r"\*\*Source:\*\*\s*(.+)", content)
        if source_match:
            metadata["url"] = source_match.group(1).strip()
        return metadata


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        line = json.dumps(record, ensure_ascii=False)
        line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        lines.append(line)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
