"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math

from .task4_chunking_indexing import load_or_build_chunks, tokenize

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return _SimpleBM25(tokenized_corpus)


class _SimpleBM25:
    """BM25 tối giản để module không phụ thuộc cài đặt ngoài."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in tokenized_corpus) / max(1, len(tokenized_corpus))
        self.df = {}
        for doc in tokenized_corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.n_docs = len(tokenized_corpus)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc in self.corpus:
            freqs = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            score = 0.0
            doc_len = len(doc) or 1
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global CORPUS
    CORPUS = load_or_build_chunks(with_embeddings=False)
    if not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = list(bm25.get_scores(tokenize(query)))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results = []
    for idx, score in ranked[:max(0, top_k)]:
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(score),
            "metadata": CORPUS[idx].get("metadata", {}),
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
