# Run Guide

Hướng dẫn chạy frontend RAG, PostgreSQL/pgvector và DB viewer.

## 1. Chuẩn Bị Môi Trường

Tạo virtualenv và cài dependency:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Nếu chỉ cần chạy phần đã setup hiện tại, các dependency quan trọng là:

```bash
.venv/bin/python -m pip install "python-dotenv>=1.0.0" "psycopg[binary]>=3.2.0" "streamlit>=1.35.0"
```

## 2. Chạy PostgreSQL + pgvector

Start database:

```bash
docker compose up -d postgres
```

Kiểm tra container:

```bash
docker compose ps
```

Kiểm tra extension và số dòng:

```bash
docker compose exec -T postgres psql -U rag -d rag_lab -c "
select extname, extversion from pg_extension where extname='vector';
select count(*) as documents from documents;
select count(*) as chunks from chunks;
"
```

## 3. Index Dữ Liệu Vào pgvector

Đảm bảo `.env` có:

```env
RAG_VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://rag:rag@127.0.0.1:5432/rag_lab
```

Tạo data mẫu nếu cần:

```bash
.venv/bin/python -m src.sample_data
```

Init schema và index documents/chunks:

```bash
.venv/bin/python -m src.setup_pgvector
```

Kết quả mong đợi:

```text
PostgreSQL/pgvector is ready.
Documents indexed: 9
Chunks indexed: 20
Chunks in PostgreSQL: 20
```

## 4. Chạy Frontend RAG

Start Streamlit:

```bash
.venv/bin/python -m streamlit run app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:8501
```

UI có:

- Chat hỏi đáp RAG
- Answer kèm citation
- Retrieval mode
- Source chunks
- Scores
- Chunk ID
- Nút rebuild index

## 5. Chạy DB Viewer

Start pgweb:

```bash
docker compose up -d pgweb
```

Mở trình duyệt:

```text
http://127.0.0.1:8081
```

Trong pgweb có thể xem:

- `public.documents`
- `public.chunks`
- `content`
- `embedding_text`
- `search_text`
- `metadata`
- `embedding vector(256)`

Query mẫu:

```sql
select
  chunk_id,
  metadata->>'source' as source,
  metadata->>'article' as article,
  left(content, 180) as preview
from chunks
limit 10;
```

Check vector dimension:

```sql
select chunk_id, vector_dims(embedding) as dims
from chunks
limit 5;
```

Search keyword trong DB:

```sql
select
  chunk_id,
  metadata->>'source' as source,
  ts_rank_cd(
    to_tsvector('simple', search_text),
    plainto_tsquery('simple', 'ma tuy')
  ) as score,
  left(content, 180) as preview
from chunks
where to_tsvector('simple', search_text) @@ plainto_tsquery('simple', 'ma tuy')
order by score desc
limit 10;
```

## 6. Test Retrieval Và Generation

Semantic/vector search:

```bash
.venv/bin/python -c "from src.task5_semantic_search import semantic_search; print(semantic_search('cai nghiện ma túy', top_k=3))"
```

Keyword search:

```bash
.venv/bin/python -c "from src.task6_lexical_search import lexical_search; print(lexical_search('Điều 249 ma túy', top_k=3))"
```

Hybrid retrieval:

```bash
.venv/bin/python -c "from src.task9_retrieval_pipeline import retrieve; print(retrieve('Điều 249 Bộ luật Hình sự', top_k=3))"
```

Generation có citation:

```bash
.venv/bin/python -c "from src.task10_generation import generate_with_citation; print(generate_with_citation('Các hình thức cai nghiện ma túy là gì?', top_k=3)['answer'])"
```

## 7. Chạy Test Suite

```bash
.venv/bin/python -m unittest tests.test_individual -v
```

Nếu đã cài pytest:

```bash
.venv/bin/python -m pytest tests/test_individual.py -v
```

## 8. Stop Services

Stop UI Streamlit:

```bash
Control + C
```

Stop DB viewer:

```bash
docker compose stop pgweb
```

Stop PostgreSQL:

```bash
docker compose stop postgres
```

Stop tất cả:

```bash
docker compose down
```

Xóa luôn volume database nếu muốn reset sạch:

```bash
docker compose down -v
```

Sau khi reset volume, chạy lại:

```bash
docker compose up -d postgres
.venv/bin/python -m src.setup_pgvector
```

## 9. Ports

| Service | URL |
|---|---|
| RAG Frontend | `http://127.0.0.1:8501` |
| pgweb DB Viewer | `http://127.0.0.1:8081` |
| PostgreSQL | `127.0.0.1:5432` |

