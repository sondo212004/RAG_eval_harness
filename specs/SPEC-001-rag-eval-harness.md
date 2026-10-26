# SPEC-001: rag-eval-harness — harness đánh giá RAG & benchmark agent

| | |
|---|---|
| Trạng thái | Đã chốt |
| Chủ sở hữu | @sondo |
| Ticket | — (repo cá nhân, public GitHub) |
| Ngày chốt | 2026-10-26 |
| Reviewer bắt buộc | @sondo |

## 1. Vấn đề
Các con số trong CV (bộ RAGAS 62 cặp, benchmark routing 20–100 câu, CI release gate) nằm trong
GitLab công ty, nhà tuyển dụng không kiểm chứng được. Cần một repo cá nhân **viết lại từ đầu,
chỉ dùng tài liệu gốc công khai**, tái lập được bằng một lệnh.

## 2. Mục tiêu & thước đo
- Mục tiêu: người xem clone repo → chạy 1 lệnh → ra bảng điểm RAGAS + latency/token/cost giống README.
- Đo bằng:
  - `make reproduce` chạy xong trên máy sạch, không cần sửa code.
  - CI xanh trên `main`; PR làm metric tụt quá ngưỡng → CI đỏ.
  - README có bảng kết quả sinh tự động từ file kết quả (không gõ tay).

## 3. Ngoài phạm vi
- Code, prompt, schema, bộ Q/A, dữ liệu export, endpoint của công ty/khách hàng.
- UI web/dashboard — chỉ CLI + báo cáo Markdown/JSON.
- RAG production-grade (memory, RBAC, HITL…) — chỉ pipeline tham chiếu đủ để có đối tượng đo.
- Fine-tune model.

## 4. Luồng người dùng
1. Clone repo, copy `.env.example` → `.env`, điền `GEMINI_API_KEY` (chỉ cần khi chạy live).
2. `make fetch` — tải tài liệu gốc theo manifest (URL + sha256).
3. `make ingest` — parse, chunk, embed vào vector store local.
4. `make eval` — chạy bộ Q/A qua adapter (RAG tham chiếu hoặc HTTP), chấm RAGAS, ghi `results/<run_id>.json`.
5. `make bench` — chạy bộ câu routing, ghi latency/token/cost theo route.
6. `make compare` — so run mới nhất với `results/baseline.json`, exit ≠ 0 nếu tụt quá ngưỡng.
7. `make report` — cập nhật bảng kết quả trong README.
8. `make promote RUN=<run_id>` — đặt run làm baseline mới (chủ đích).
9. `make reproduce` = compare + report offline trên kết quả đã commit; `make reproduce-live` = fetch → ingest → eval → bench → compare → report (cần key).
10. CI chạy `compare` trên kết quả đã commit ở mỗi PR; live run bật tay qua `workflow_dispatch`.

## 5. Tiêu chí nghiệm thu (Given/When/Then)
- **AC1** — Given file dataset (mỗi mục có `id`, `question`, `ground_truth`, `gold_contexts`, `domain` ∈ {legal, hardware, om, scada}, `source`, `reviewed`), When chạy loader, Then nạp đủ N mục; mục thiếu field / domain lạ / trùng `id` → lỗi kèm mã ở mục 6.
- **AC2** — Given dataset có mục `reviewed: false`, When chạy eval ở chế độ mặc định, Then các mục đó bị loại và số lượng loại được in ra.
- **AC3** — Given adapter trả `answer` + `contexts` cho từng câu, When chạy eval, Then `results/<run_id>.json` có `faithfulness`, `answer_relevancy`, `context_recall` từng mục, trung bình theo domain và tổng, kèm `dataset_version`, model, thời điểm (lấy từ `Clock`).
- **AC4** — Given baseline và run hiện tại cùng `dataset_version`, When một metric trung bình (tổng hoặc theo domain) giảm ≥ 0.05, Then `compare` exit ≠ 0 và in metric vi phạm.
- **AC5** — Given latency p95 của `bench` ở bất kỳ route nào > 1.2 × p95 cùng route trong baseline, When chạy `compare`, Then exit ≠ 0 và in route vi phạm.
- **AC6** — Given mọi metric trong ngưỡng, When chạy `compare`, Then exit = 0 và in bảng chênh lệch.
- **AC7** — Given bộ câu routing có nhãn `expected_route` ∈ {rag_retrieval, direct_answer}, When chạy `bench`, Then ghi tỉ lệ route đúng, latency p50/p95, token in/out, cost ước tính theo từng route.
- **AC8** — Given không đặt `LANGSMITH_API_KEY`, When chạy eval/bench, Then chạy bình thường và chỉ ghi local; có key thì trace được gửi (kiểm bằng fake client).
- **AC9** — Given file kết quả, When chạy `report`, Then bảng Markdown được chèn vào README giữa `<!-- results:start -->` và `<!-- results:end -->`; chạy lại không đổi nội dung (idempotent).
- **AC10** — Given cây mã nguồn, When chạy test kiến trúc (AST), Then fail nếu module trong `domain/` import `app/`, `infra/` hoặc thư viện I/O.
- **AC11** — Given cây repo, When CI chạy bước kiểm chuỗi cấm, Then fail nếu file được track chứa chuỗi trong danh sách cấm (config, gồm tên khách hàng).
- **AC12** — Given manifest corpus, When chạy `fetch`/`prepare`, Then file sai sha256 bị báo và bỏ qua, không dừng cả lô.
- **AC13** — Given provider được chọn qua config (`gemini` | `openai` | `ollama`), When khởi tạo pipeline, Then dùng đúng adapter; provider lạ → `CONFIG_INVALID`.
- **AC14** — Given run hợp lệ, When chạy `promote RUN=<run_id>`, Then `results/baseline.json` được thay bằng run đó; run `RUN_INCOMPLETE` hoặc không tồn tại → từ chối kèm mã lỗi.
- **AC15** — Given 2 lần ingest cùng corpus + config, When so `dataset_version` và id chunk, Then giống nhau (chunk/embedding tái lập được).

### 5.1 Map AC → test (điền ngay sau khi chốt spec, TRƯỚC khi code)
| AC | Test (file — tên test) | Đã đỏ trước? | Xanh khi nghiệm thu? |
|---|---|---|---|
| AC1 | `tests/domain/test_dataset.py` — `test_load_valid`, `test_missing_field`, `test_unknown_domain`, `test_duplicate_id` | ☑ (ImportError) | ☐ |
| AC2 | `tests/domain/test_dataset.py` — `test_unreviewed_excluded_and_counted` | ☑ (ImportError) | ☐ |
| AC3 | `tests/app/test_run_eval.py` — `test_result_has_per_item_domain_and_total_scores` | ☑ (ImportError) | ☐ |
| AC4 | `tests/domain/test_compare.py` — `test_metric_drop_fails_total`, `test_metric_drop_fails_domain` | ☑ (ImportError) | ☐ |
| AC5 | `tests/domain/test_compare.py` — `test_route_p95_regression_fails` | ☑ (ImportError) | ☐ |
| AC6 | `tests/domain/test_compare.py` — `test_within_threshold_passes_with_diff_table` | ☑ (ImportError) | ☐ |
| AC7 | `tests/app/test_run_bench.py` — `test_route_accuracy_latency_tokens_cost_per_route` + `tests/domain/test_routing.py` | ☑ (ImportError) | ☐ |
| AC8 | `tests/infra/test_tracing.py` — `test_no_key_local_only`, `test_key_sends_to_fake_client` | ☐ | ☐ |
| AC9 | `tests/app/test_report.py` — `test_inserts_between_markers`, `test_idempotent` | ☑ (ImportError) | ☐ |
| AC10 | `tests/test_architecture.py` — `test_domain_has_no_forbidden_imports` | ☐ (chưa quan sát đỏ — có test tự kiểm `test_checker_detects_violation`) | ☐ |
| AC11 | `tests/domain/test_forbidden.py` — `test_detects_forbidden_string_case_insensitive`; `tests/test_forbidden_strings.py` — `test_tracked_files_clean` (danh sách từ env `FORBIDDEN_STRINGS` / `config/forbidden_strings.txt`) | ☑ (ImportError) | ☐ |
| AC12 | `tests/app/test_fetch.py` — `test_checksum_mismatch_skipped_batch_continues` | ☑ (ImportError) | ☐ |
| AC13 | `tests/infra/test_provider_factory.py` — `test_selects_adapter`, `test_unknown_provider_config_invalid` | ☐ | ☐ |
| AC14 | `tests/app/test_promote.py` — `test_promote_valid`, `test_reject_incomplete_or_missing` | ☑ (ImportError) | ☐ |
| AC15 | `tests/app/test_ingest.py` — `test_ingest_reproducible_chunk_ids` | ☐ | ☐ |
| Mục 6 | `tests/app/test_run_eval.py`, `tests/app/test_compare_runs.py`, `tests/app/test_promote.py`, `tests/domain/test_compare.py` — mỗi mã lỗi ≥ 1 test; `CONFIG_MISSING` / `CONFIG_INVALID` ở tầng infra | ☑ (ImportError, trừ infra) | ☐ |

## 6. Case lỗi & biên
| Tình huống | Hành vi mong đợi | Mã lỗi |
|---|---|---|
| Dataset rỗng / thiếu field / domain lạ | Dừng, báo mục + field lỗi | `DATASET_INVALID` |
| `id` trùng | Dừng, liệt kê id trùng | `DATASET_DUPLICATE_ID` |
| Thiếu API key khi chạy live | Dừng trước khi gọi, báo biến thiếu | `CONFIG_MISSING` |
| Provider không hỗ trợ | Dừng | `CONFIG_INVALID` |
| API LLM lỗi / timeout | Retry tối đa 3 lần, quá thì đánh dấu mục `error`, run vẫn hoàn tất | `PROVIDER_ERROR` |
| Baseline không tồn tại | Dừng, báo đường dẫn | `BASELINE_NOT_FOUND` |
| Baseline khác `dataset_version` | Từ chối so sánh | `DATASET_MISMATCH` |
| File corpus sai sha256 / tải lỗi | Bỏ qua file, cảnh báo | `CORPUS_CHECKSUM_MISMATCH` |
| Chạy eval 2 lần cùng `run_id` | Không ghi đè im lặng | `RUN_EXISTS` |
| Tỉ lệ mục `error` > 10% | Run đánh dấu không hợp lệ, `compare` từ chối dùng | `RUN_INCOMPLETE` |
| Domain có trong baseline nhưng mất điểm ở run mới (mọi mục lỗi, tổng lỗi ≤ 10%) | `compare` báo vi phạm `<domain>.missing`, exit ≠ 0 | — |
| `compare` / `promote` với run không tồn tại | Dừng, báo đường dẫn | `RUN_NOT_FOUND` |
| README thiếu marker `results:start/end` | Dừng, không sửa README | `REPORT_MARKERS_MISSING` |

## 7. Ảnh hưởng kỹ thuật
- Repo mới, không đụng hệ thống nào khác.
- Stack: Python + pytest; 3 tầng `domain/` (thuần: tính điểm tổng hợp, so baseline, validate dataset) ← `app/` (use case) ← `infra/` (LLM, vector store, RAGAS, LangSmith, file).
- Contract: adapter HTTP đo chatbot ngoài → OpenAPI 3.1 trong `contracts/` mô tả request/response tối thiểu (`question` → `answer`, `contexts[]`).
- Dependency (đã duyệt 2026-10-26): ragas, langchain-core, langchain-text-splitters, langgraph, google-genai, qdrant-client (local mode), sentence-transformers (bge-m3, kéo torch), flashrank, langsmith, pydantic, beautifulsoup4, pypdf; dev: pytest, ruff.
  - Bổ sung (đã duyệt 2026-10-26): pyyaml (đọc config/pricing), httpx (adapter HTTP), `langchain-community<0.4` (ragas 0.4.3 import lỗi với 0.4.x).
- Danh sách chuỗi cấm (AC11) tối thiểu gồm tên công ty/khách hàng, domain nội bộ và tên sản phẩm nhà cung cấp thấy trong dự án cũ — để trong file config ngoài git hoặc dạng hash, để chính danh sách không làm lộ tên.

## 8. Rủi ro & phương án dự phòng
| Rủi ro | Xác suất | Ảnh hưởng | Cách giảm thiểu |
|---|---|---|---|
| Điểm RAGAS dao động giữa các lần chạy (LLM judge) | Cao | CI đỏ/xanh ngẫu nhiên | temperature 0; CI so trên kết quả đã commit; ngưỡng 0.05 |
| Chi phí / lộ key khi chạy CI | Trung bình | Tốn tiền, lộ key | CI mặc định không gọi LLM; live chỉ qua `workflow_dispatch` + secret |
| Rò rỉ dấu vết công ty (tên khách hàng, dữ liệu export) | Trung bình | Vi phạm NDA | Mục 11 + CI grep chuỗi cấm + review trước khi public |
| URL nguồn chết | Trung bình | Không tái lập được | Manifest ghi sha256; README hướng dẫn thay URL |

## 9. Quyết định đã trả lời
**Vòng 1 (2026-10-26)**
- Đối tượng đo: **cả hai** — RAG tham chiếu nhỏ trong repo + adapter HTTP đo chatbot bất kỳ.
- Provider: **Gemini mặc định**, qua lớp abstraction đổi được OpenAI/Ollama.
- CI gate: **so trên kết quả đã commit**; live run qua `workflow_dispatch` với secret.
- Tracing: **ghi local JSON**; có `LANGSMITH_API_KEY` thì đẩy trace lên thêm.
- Nguồn gốc: **không port gì từ công ty** — code viết lại từ đầu theo cùng phương pháp, dùng lại tài liệu gốc public, bộ Q/A làm mới.

**Vòng 2 (2026-10-26)**
- Stack: **Python + pytest, 3 tầng + test AST**.
- Q/A: **LLM nháp + duyệt tay**, mỗi mục `reviewed: true` + nguồn điều khoản.
- Routing: **`rag_retrieval` vs `direct_answer`**.
- CI gate: **metric RAGAS trung bình giảm ≥ 0.05 (tuyệt đối) hoặc latency p95 tăng > 20%** → fail.
- Corpus: lấy từ file zip đã tải, theo quy tắc mục 11.

**Vòng 3 (2026-10-26)**
- Giữ **4 domain**: legal, hardware, om, scada.
- Đồng ý quy tắc dữ liệu mục 11.
- Không UI web — chỉ CLI + báo cáo Markdown/JSON.

**Vòng 4 (2026-10-26)**
- Domain **hardware để trống** (giữ trong enum, 0 mục Q/A; báo cáo hiển thị "—").
- Phương pháp tham khảo từ các dự án nước cũ (chỉ phương pháp/thông số, không copy code, không mở bộ Q/A cũ).

**Vòng 5 (2026-10-26)**
- Chunking parent/child **3000/200 + 500/50** (ký tự); embedding **`BAAI/bge-m3` local** (SentenceTransformer, 1024 chiều); **Qdrant local mode**, cosine.
- **`gemini-2.5-flash`** cho cả generator và judge RAGAS, temperature 0; giá **0.30 / 2.50 USD per 1M token** in/out, để trong config (không hardcode).
- **Python 3.12 + uv + Makefile** (Makefile chỉ gọi `uv run python -m ...`); Windows chạy qua Git Bash/WSL, README ghi rõ.
- `dataset_version` = 12 ký tự đầu sha256 file dataset; baseline commit ở **`results/baseline.json`**, cập nhật bằng lệnh `make promote RUN=<run_id>` (chủ đích, không tự động).

**Vòng 6 (2026-10-26)**
- Gate latency: p95 **từ `bench`, theo từng route**, > 1.2 × baseline → fail.
- `make reproduce` **offline** (compare + report trên `results/` đã commit, không cần key); `make reproduce-live` chạy fetch → ingest → eval → bench → compare → report.
- Retrieval: fetch_k 30 → lọc score ≥ 0.7 → rerank **FlashRank** (`ms-marco-TinyBERT-L-2-v2`) → top_k 5.

**Vòng 7 (2026-10-26) — người dùng chốt spec**
- Đồng ý AC14 (`promote`), AC15 (ingest tái lập).
- Danh sách chuỗi cấm để trong config ngoài git hoặc dạng hash.
- `prepare` = lấy file từ zip local (theo whitelist mục 11) cho tài liệu không còn URL gốc, kiểm sha256 như `fetch`.

**Vòng 8 (2026-10-26) — làm rõ khi vào tầng app**
- eval và bench **cùng `run_id`, 2 file**: `results/<run_id>.json` (eval) + `results/<run_id>.bench.json`; `make bench RUN=<id>`. `promote` copy cả hai thành `results/baseline.json` + `results/baseline.bench.json`. Thiếu bench → `compare` chỉ so RAGAS. Bench khác `dataset_version` routing → `DATASET_MISMATCH`.
- RAG tham chiếu chọn route bằng **LLM phân loại** (structured output: `rag_retrieval` | `direct_answer`).
- Bộ routing **~40 câu** (20/20), LLM nháp + duyệt tay, mỗi mục `id`, `question`, `expected_route`, `reviewed`.
- Dataset dạng **JSONL**: `datasets/qa.jsonl`, `datasets/routing.jsonl`.

## 9b. Câu hỏi mở
_(không còn)_

## 10. Ước lượng
M — 5–8 ngày; phần lớn thời gian là duyệt tay ~60 cặp Q/A.

## 11. Quy tắc dữ liệu (từ bộ dữ liệu portable — zip local, ngoài git)
**Được dùng** — chỉ tài liệu gốc công khai:
- `public/legal_standards/*.html`, `pdf_raw_data/*` (luật, nghị định, QCVN, TCVN).
- `public/om_manuals/**/*.pdf|*.txt` (manual thiết bị) + `list_machine_and_manual_url.txt` (URL nguồn).
- `public/scada_solutions/SCADA System Fundamentals-R1.pdf` và `pdf_raw_data/zLuan van-…SCADA…pdf` (domain scada).
- Manual, luận văn là tác phẩm có bản quyền (khác văn bản QPPL): **không phân phối lại file** — chỉ manifest URL; gold context trích đoạn ngắn kèm nguồn.

**Cấm dùng / cấm commit** — sản phẩm của hệ thống công ty:
- `postgres/kb_backup.sql`, `qdrant/*.snapshot` (export DB).
- `chunks_for_*/**`, `*_with_bge_embeddings.json` (output pipeline công ty).
- `public/benchmark/chatbot_benchmark_dataset.json` (bộ Q/A công ty — không mở, không tham chiếu).
- `users/`.

**Luật repo:**
- Zip và thư mục giải nén nằm ngoài git (`.gitignore`); repo chỉ chứa script `fetch`/`prepare` + manifest (tên file, URL nguồn, sha256) — người clone tự tải từ URL gốc.
- Không xuất hiện tên khách hàng/công ty trong code, dataset, tên file, commit; CI có bước grep chặn (AC11).
- Chunking và embedding làm lại bằng code của repo.
