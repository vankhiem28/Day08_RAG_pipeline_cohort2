# RAG Evaluation Results

## Framework sử dụng

Default run: offline heuristic evaluator with deterministic token-overlap metrics mapped to the four required RAG dimensions. Optional run: set `RAG_EVAL_FRAMEWORK=deepeval` to execute DeepEval metrics when LLM credentials are available.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |
|--------|-----------------------------|------------------------------|-------|
| Faithfulness | 0.997 | 0.997 | +0.000 |
| Answer Relevance | 0.723 | 0.732 | -0.009 |
| Context Recall | 0.856 | 0.867 | -0.011 |
| Context Precision | 0.988 | 0.988 | +0.000 |
| Average | 0.891 | 0.896 | -0.005 |

## A/B Comparison Analysis

**Config A:** hybrid retrieval with RRF merge and local lexical reranking.

**Config B:** hybrid retrieval with RRF merge, without the final reranking pass.

**Kết luận:** Config B có điểm trung bình cao hơn trong lần chạy này. Nếu điểm recall thấp, ưu tiên cải thiện chunking/metadata; nếu precision thấp, ưu tiên reranking và lọc nhiễu từ HTML crawl.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Top Sources |
|---|----------|--------------|-----------|--------|-----------|---------------|-------------|
| 1 | Theo Luật Phòng, chống ma túy 2021, chất ma túy là gì? | 1.000 | 0.494 | 0.708 | 1.000 | mixed | 73luat.md, 73luat.md, 73luat.md |
| 2 | Luật Phòng, chống ma túy 2021 nêu cây có chứa chất ma túy gồm những cây nào? | 1.000 | 0.503 | 0.708 | 1.000 | mixed | 73luat.md, 73luat.md, 73luat.md |
| 3 | Bài viết về Miu Lê ở Cát Hải nêu những chất nào được thu giữ khi khám xét cơ sở lưu trú? | 1.000 | 0.578 | 0.844 | 1.000 | mixed | article_01.md, article_05.md, article_04.md |

## Recommendations

### Cải tiến 1
**Action:** Làm sạch markdown báo chí sau crawl để bỏ menu, footer, quảng cáo và link liên quan.
**Expected impact:** Tăng context precision và giảm câu trả lời trích nhầm nội dung nhiễu.

### Cải tiến 2
**Action:** Bổ sung metadata thủ công cho văn bản pháp luật: số điều, chương, tên văn bản, năm ban hành.
**Expected impact:** Tăng context recall cho câu hỏi pháp luật có trích điều/khoản.

### Cải tiến 3
**Action:** Thử reranker thật như Jina hoặc cross-encoder multilingual khi có API/GPU.
**Expected impact:** Cải thiện thứ tự source chunks cho câu hỏi tin tức nhiều thực thể.
