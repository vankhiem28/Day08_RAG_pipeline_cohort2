# Bài Tập Nhóm - RAG Chatbot

## Mục Tiêu

Xây dựng chatbot RAG trả lời câu hỏi về pháp luật phòng, chống ma túy và các bài báo liên quan tới nghệ sĩ. Sản phẩm nhóm chạy local bằng Streamlit, trả lời có citation, hiển thị source chunks và có conversation memory cho follow-up questions.

Data nhóm hiện lấy từ `invidual/minhchi/data` và đã được copy sang root `data/` để app chạy độc lập, không phụ thuộc trực tiếp vào thư mục bài cá nhân.

## Kiến Trúc Hệ Thống

```text
data/landing từ Minh Chi
  -> data/standardized markdown
  -> src.services.IndexingService
  -> domain-aware chunking + local-hashing-v1 embeddings
  -> data/index/documents.jsonl + data/index/chunks.jsonl
  -> hybrid retrieval: vector cosine + BM25 + RRF + optional rerank
  -> vectorless fallback khi score thấp hoặc exact/legal query
  -> generation có citation
  -> Streamlit chat UI + source display + conversation memory
```

Các module chính:

- `app.py`: giao diện Streamlit.
- `src/chat_pipeline.py`: wrapper conversation memory cho follow-up questions.
- `src/task9_retrieval_pipeline.py`: retrieval pipeline.
- `src/task10_generation.py`: generation có citation.
- `src/services/*`: service dùng chung đã tách từ bài cá nhân `vankhiem`.
- `group_project/evaluation/*`: golden dataset, evaluation script và report.

## Trạng Thái Deliverables

| Hạng mục | Trạng thái |
|---|---|
| Giao diện chat Streamlit | Hoàn thành |
| Trả lời có citation | Hoàn thành |
| Follow-up questions / conversation memory | Hoàn thành |
| Hiển thị source documents | Hoàn thành |
| Data từ Minh Chi | Hoàn thành |
| Local JSONL index | Hoàn thành: 8 documents, 585 chunks |
| Golden dataset evaluation | Hoàn thành: 16 Q&A |
| A/B comparison | Hoàn thành: hybrid + rerank vs hybrid no rerank |
| Evaluation report | Hoàn thành: `group_project/evaluation/results.md` |

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Minh Chi | TBD | Thu thập/crawl data legal và news dùng cho bài nhóm | Hoàn thành |
| Vạn Khiêm | TBD | Tách service RAG, tích hợp Streamlit app, indexing/retrieval/generation | Hoàn thành |
| Nhóm | TBD | Golden dataset, A/B evaluation, báo cáo kết quả | Hoàn thành |

## Hướng Dẫn Chạy

Cài dependency trong virtualenv hiện có:

```bash
venv/bin/python -m pip install -r requirements.txt
```

Rebuild index từ data Minh Chi:

```bash
venv/bin/python -m src.task4_chunking_indexing
```

Chạy chatbot:

```bash
venv/bin/python -m streamlit run app.py
```

Kiểm tra retrieval/generation nhanh:

```bash
venv/bin/python -c "from src.task9_retrieval_pipeline import retrieve; print(retrieve('Điều 28 biện pháp cai nghiện ma túy', top_k=3))"
venv/bin/python -c "from src.task10_generation import generate_with_citation; print(generate_with_citation('Hữu Tín bị đề nghị truy tố tội gì?', top_k=3)['answer'])"
```

Chạy evaluation và xuất báo cáo:

```bash
venv/bin/python group_project/evaluation/eval_pipeline.py
```

Kết quả mới nhất nằm ở:

```text
group_project/evaluation/results.md
```

## Evaluation

Framework mặc định là offline heuristic evaluator để đảm bảo demo chạy được khi không có API key. Script vẫn đo đủ 4 trục yêu cầu:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

A/B configs:

- Config A: hybrid retrieval + rerank.
- Config B: hybrid retrieval không rerank.

Khi có API key/model, có thể chạy thêm DeepEval bằng:

```bash
RAG_EVAL_FRAMEWORK=deepeval venv/bin/python group_project/evaluation/eval_pipeline.py
```
