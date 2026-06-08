"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import html
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://thanhnien.vn/dien-vien-huu-tin-nghien-ma-tuy-gan-3-nam-moi-ban-ve-nha-su-dung-thuoc-lac-1851517030.htm",
    "https://tienphong.vn/hanh-trinh-phe-ma-tuy-roi-giet-nguoi-cua-ca-si-chau-viet-cuong-post1095287.tpo",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://xaydungchinhsach.chinhphu.vn/khoi-to-bat-tam-giam-long-nhat-son-ngoc-minh-cung-69-bi-can-119260520124509053.htm",
]


class ArticleHTMLParser(HTMLParser):
    """HTML parser tối giản để fallback khi Crawl4AI không chạy được."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self._capture_tag = None
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "p"}:
            self._capture_tag = tag

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == self._capture_tag:
            self._capture_tag = None

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif self._capture_tag:
            self._parts.append(text)

    @property
    def markdown(self) -> str:
        paragraphs = []
        for part in self._parts:
            if len(part) >= 30 and part not in paragraphs:
                paragraphs.append(part)
        return "\n\n".join(paragraphs)


def _validate_article(article: dict) -> dict:
    content = article.get("content_markdown", "").strip()
    if len(content) < 500:
        raise ValueError(
            f"Crawl không đủ nội dung cho URL {article.get('url')}: "
            f"{len(content)} ký tự. Hãy kiểm tra URL, mạng, hoặc cơ chế chống bot."
        )
    return article


def crawl_article_with_urllib(url: str) -> dict:
    """Fallback không cần package ngoài: tải HTML và extract title/paragraph."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        raw_html = response.read().decode("utf-8", errors="ignore")

    parser = ArticleHTMLParser()
    parser.feed(raw_html)
    title = parser.title or url.rstrip("/").split("/")[-1]
    content = f"# {title}\n\n{parser.markdown}"
    return _validate_article({
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    })


def crawl_article_with_jina_reader(url: str) -> dict:
    """Fallback reader service: trả markdown thật từ URL công khai."""
    reader_url = f"https://r.jina.ai/{url}"
    request = Request(
        reader_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/plain",
        },
    )
    with urlopen(request, timeout=45) as response:
        content = response.read().decode("utf-8", errors="ignore").strip()

    title = url.rstrip("/").split("/")[-1]
    for line in content.splitlines()[:8]:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break

    return _validate_article({
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    })


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    crawl4ai_error = None
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            article = {
                "url": url,
                "title": result.metadata.get("title", "Unknown"),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown,
            }
            return _validate_article(article)
    except Exception as exc:
        crawl4ai_error = exc

    jina_error = None
    try:
        return crawl_article_with_jina_reader(url)
    except Exception as exc:
        jina_error = exc

    try:
        return crawl_article_with_urllib(url)
    except Exception as urllib_error:
        raise RuntimeError(
            f"Không crawl được bài báo thật: {url}\n"
            f"- Crawl4AI error: {crawl4ai_error}\n"
            f"- Jina Reader error: {jina_error}\n"
            f"- urllib fallback error: {urllib_error}"
        ) from urllib_error


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
