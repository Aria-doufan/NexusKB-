# PDF Import Backend Enhancement Design

## Context

NexusKB already accepts uploaded PDF files through the existing vector upload endpoints:

```text
/api/vector/add/single
/api/vector/add/multiple
```

The current backend flow validates the uploaded file in `ChatService`, writes it to a temporary file in `VectorStoreService.get_document()`, dispatches `.pdf` files to `pdf_loader()`, splits the returned LangChain `Document` objects, attaches `user_id` metadata, and stores the chunks in Chroma.

The current PDF loader uses LangChain `PyPDFLoader`. It works for basic PDFs, but the extraction behavior is not explicit in NexusKB code. The backend also lacks clear page limits, clear empty-text errors, PDF-specific tests, and reliable upload failure feedback.

## Goal

Enhance the existing backend PDF import path while preserving the current API contract:

- Use a project-owned `pdfplumber` extraction flow for uploaded PDFs.
- Enforce a 20-page maximum per PDF.
- Return clear failures for oversized, empty, damaged, encrypted, or image-only PDFs.
- Preserve page-level metadata so later retrieval and debugging can identify PDF page sources.
- Keep existing vector upload endpoints, text splitting, and Chroma storage behavior.

## Non-goals

- Do not add OCR for scanned/image-only PDFs.
- Do not add or redesign a frontend upload UI.
- Do not import PDFs into the EnterpriseRAG/Elasticsearch offline benchmark dataset.
- Do not refactor all document format loaders.
- Do not add partial-success semantics for multi-file uploads in this iteration.

## Recommended Approach

Replace `backend/app/utils/file_handler.py` PDF loading internals with a small `pdfplumber`-based implementation inspired by the existing `pdf_text_extractor.py` helper.

The public function shape should remain compatible with the current vector store:

```python
async def pdf_loader(file_path: str, password: str = None) -> list[Document]:
    ...
```

Internally, PDF extraction should be explicit and testable. It should open the PDF with `pdfplumber`, check page count before extracting text, extract each page independently, and return one LangChain `Document` per page with PDF-specific metadata.

## Data Flow

```text
UploadFile
  -> ChatService validates size and type
  -> VectorStoreService.get_document(files=[...], user_id=...)
  -> write upload bytes to a temporary file and close it
  -> get_file_document(temp_path)
  -> pdf_loader(temp_path)
  -> pdfplumber extracts pages, max 20
  -> list[Document] with page metadata
  -> AsyncTextSplitter.split_documents()
  -> attach user_id and original filename metadata
  -> Chroma.add_documents()
  -> temporary file cleanup
```

The existing `/api/vector/add/single` and `/api/vector/add/multiple` endpoints stay unchanged from the caller perspective.

## PDF Document Shape

The loader should produce one `Document` per page that has extractable text:

```python
Document(
    page_content="page text...",
    metadata={
        "source": file_path,
        "page": 1,
        "page_count": 12,
        "content_length": 530,
        "file_type": "pdf",
    },
)
```

Design choices:

- `page` is 1-based to match the page number users see in PDF readers.
- `page_count` records the total number of pages in the original PDF.
- `content_length` records the extracted text length for that page.
- `file_type` is always `pdf` for these documents.
- `source` remains the local path used by the current ingestion flow.

For uploaded files, `VectorStoreService.get_document()` should also attach the original `UploadFile.filename` to every resulting chunk as `filename`. This makes retrieval/debug output more useful than showing only a temporary file path.

## Error Handling

Add PDF-specific exceptions in the loader layer:

- `PdfTextExtractionError`: generic PDF extraction failure.
- `PdfPageLimitError`: page count exceeds the 20-page limit.
- `PdfNoExtractableTextError`: the PDF opens but no page contains extractable text.

Expected behavior:

| Condition | Behavior |
| --- | --- |
| Empty file | Fail with `PdfTextExtractionError` |
| Page count > 20 | Fail with `PdfPageLimitError` |
| Image-only/scanned PDF | Fail with `PdfNoExtractableTextError` |
| Damaged/encrypted/unreadable PDF | Fail with `PdfTextExtractionError` |
| Duplicate MD5 | Skip as existing behavior, not a failure |

For uploads, failures should propagate to the API response instead of being logged and hidden:

- Single-file upload: return HTTP 400 with the PDF error detail.
- Multi-file upload: return HTTP 400 if any uploaded file fails, including the failed filename.
- Local folder ingestion with `files=None`: keep the current best-effort behavior of logging bad files and continuing.

## Temporary File Lifecycle

`VectorStoreService.get_document()` should write uploaded bytes to a temporary file, flush/close the handle before calculating MD5 or loading the file, and delete temporary files in a `finally` block.

This replaces the current scattered cleanup blocks and avoids Windows file-handle issues when reading immediately after writing.

## Dependencies

Add `pdfplumber` to both backend dependency manifests:

- `backend/requirements.txt`
- `backend/pyproject.toml`

No frontend dependencies are needed.

## Testing Plan

### PDF loader tests

Add tests for the loader behavior:

- Normal PDF returns page-level `Document` objects.
- Metadata includes `page`, `page_count`, `content_length`, and `file_type`.
- Page numbers start at 1.
- PDFs over 20 pages raise `PdfPageLimitError`.
- Damaged or empty PDFs raise `PdfTextExtractionError`.
- PDFs with no extractable text raise `PdfNoExtractableTextError`.

The tests may monkeypatch `pdfplumber.open()` instead of generating real PDF binaries if that keeps the test dependency surface smaller.

### Vector upload tests

Add service-level tests for the upload path:

- PDF parse failures in upload mode do not return a successful import.
- Temporary files are closed and cleaned up after success or failure.
- Uploaded filename is attached to resulting document metadata.
- Duplicate MD5 uploads remain skipped without being treated as failures.

### Suggested commands

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py
conda run -n NexusKB pytest backend/tests
```

## Acceptance Criteria

- Uploading a valid text PDF of 20 pages or fewer stores chunks in Chroma.
- Uploaded PDF chunks retain `user_id`, original `filename`, `file_type=pdf`, and page metadata.
- Uploading a PDF with more than 20 pages returns HTTP 400.
- Uploading a scanned/image-only PDF returns HTTP 400 instead of reporting success.
- Temporary upload files are removed after success, duplicate skip, and failure.
- Existing non-PDF upload behavior remains unchanged.
- `CHANGELOG.md` records the backend PDF import design/update.
