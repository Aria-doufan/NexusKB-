# PDF Import Backend Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing PDF upload loader with a project-owned `pdfplumber` flow that enforces a 20-page limit, preserves page metadata, and reports upload failures clearly.

**Architecture:** Keep the current `/api/vector/add/single` and `/api/vector/add/multiple` endpoints. Implement PDF-specific extraction inside `backend/app/utils/file_handler.py`, then make `VectorStoreService.get_document()` close temporary upload files, attach original filenames, and propagate upload-mode failures as HTTP 400 responses while preserving best-effort local-folder ingestion.

**Tech Stack:** FastAPI, LangChain `Document`, Chroma, `pdfplumber`, pytest/anyio, existing `conda run -n NexusKB` test environment.

---

## File Structure

- Modify `backend/requirements.txt`: add the runtime `pdfplumber` dependency.
- Modify `backend/pyproject.toml`: add the same dependency for uv/project installs.
- Modify `backend/app/utils/file_handler.py`: remove `PyPDFLoader` usage for PDFs, add PDF extraction exceptions and `pdfplumber` loader helpers, keep the public `pdf_loader()` signature.
- Modify `backend/app/rag/vector_store.py`: close upload temporary files before reading, centralize cleanup in `finally`, attach uploaded `filename`, and raise HTTP 400 for upload-mode failures.
- Create `backend/tests/test_pdf_loader.py`: unit tests for page metadata, 20-page limit, no text, and extraction errors.
- Create `backend/tests/test_vector_store_pdf_upload.py`: service-level tests for upload failure propagation, temp cleanup, filename metadata, and duplicate skip behavior.
- Modify `CHANGELOG.md`: add an implementation entry under Unreleased after code changes are complete.

---

### Task 1: Add the PDF extraction dependency

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add `pdfplumber` to `backend/requirements.txt`**

Add this line after `pypdf>=6.9.2`:

```text
pdfplumber>=0.11.4
```

Expected surrounding block:

```text
langchain-dashscope>=0.1.8
langchain-openai>=1.2.1
unstructured[all-docs]>=0.22.20
ragas>=0.3.0
datasets>=3.0.0
openai>=1.0.0
elasticsearch>=8.19.0,<9.0.0
```

should become:

```text
langchain-dashscope>=0.1.8
langchain-openai>=1.2.1
unstructured[all-docs]>=0.22.20
ragas>=0.3.0
datasets>=3.0.0
openai>=1.0.0
elasticsearch>=8.19.0,<9.0.0
```

and the earlier dependency block should include:

```text
pypdf>=6.9.2
pdfplumber>=0.11.4
python-magic>=0.4.27
```

- [ ] **Step 2: Add `pdfplumber` to `backend/pyproject.toml`**

Add this dependency immediately after `"pypdf>=6.9.2",`:

```toml
    "pdfplumber>=0.11.4",
```

Expected surrounding block:

```toml
    "python-multipart>=0.0.22",
    "dashscope>=1.25.14",
    "pypdf>=6.9.2",
    "pdfplumber>=0.11.4",
    "python-magic>=0.4.27",
    "python-magic-bin>=0.4.14",
```

- [ ] **Step 3: Verify dependency files contain the new package**

Run:

```powershell
Select-String -Path backend/requirements.txt,backend/pyproject.toml -Pattern "pdfplumber"
```

Expected: one match in each file.

---

### Task 2: Implement the `pdfplumber` PDF loader with tests

**Files:**
- Create: `backend/tests/test_pdf_loader.py`
- Modify: `backend/app/utils/file_handler.py`

- [ ] **Step 1: Write failing PDF loader tests**

Create `backend/tests/test_pdf_loader.py` with this content:

```python
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def patch_pdf(monkeypatch, pages):
    from app.utils import file_handler

    def fake_open(file_path, password=None):
        return FakePdf([FakePage(text) for text in pages])

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)


@pytest.mark.anyio
async def test_pdf_loader_returns_one_document_per_text_page(monkeypatch, tmp_path):
    from app.utils.file_handler import pdf_loader

    pdf_path = tmp_path / "handbook.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    patch_pdf(monkeypatch, ["First page text", "Second page text"])

    documents = await pdf_loader(str(pdf_path))

    assert [doc.page_content for doc in documents] == ["First page text", "Second page text"]
    assert documents[0].metadata == {
        "source": str(pdf_path),
        "page": 1,
        "page_count": 2,
        "content_length": len("First page text"),
        "file_type": "pdf",
    }
    assert documents[1].metadata["page"] == 2
    assert documents[1].metadata["page_count"] == 2
    assert documents[1].metadata["file_type"] == "pdf"


@pytest.mark.anyio
async def test_pdf_loader_uses_one_based_page_numbers_after_skipping_blank_pages(monkeypatch, tmp_path):
    from app.utils.file_handler import pdf_loader

    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    patch_pdf(monkeypatch, [None, "  Useful page text  "])

    documents = await pdf_loader(str(pdf_path))

    assert len(documents) == 1
    assert documents[0].page_content == "Useful page text"
    assert documents[0].metadata["page"] == 2
    assert documents[0].metadata["page_count"] == 2


@pytest.mark.anyio
async def test_pdf_loader_rejects_pdfs_over_twenty_pages(monkeypatch, tmp_path):
    from app.utils.file_handler import PdfPageLimitError, pdf_loader

    pdf_path = tmp_path / "large.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    patch_pdf(monkeypatch, [f"page {index}" for index in range(21)])

    with pytest.raises(PdfPageLimitError) as excinfo:
        await pdf_loader(str(pdf_path))

    assert "PDF page count 21 exceeds limit 20" in str(excinfo.value)


@pytest.mark.anyio
async def test_pdf_loader_rejects_pdf_with_no_extractable_text(monkeypatch, tmp_path):
    from app.utils.file_handler import PdfNoExtractableTextError, pdf_loader

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    patch_pdf(monkeypatch, [None, "", "   "])

    with pytest.raises(PdfNoExtractableTextError) as excinfo:
        await pdf_loader(str(pdf_path))

    assert "PDF contains no extractable text" in str(excinfo.value)


@pytest.mark.anyio
async def test_pdf_loader_rejects_empty_file(tmp_path):
    from app.utils.file_handler import PdfTextExtractionError, pdf_loader

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    with pytest.raises(PdfTextExtractionError) as excinfo:
        await pdf_loader(str(pdf_path))

    assert "PDF content is empty" in str(excinfo.value)


@pytest.mark.anyio
async def test_pdf_loader_wraps_pdfplumber_errors(monkeypatch, tmp_path):
    from app.utils import file_handler
    from app.utils.file_handler import PdfTextExtractionError, pdf_loader

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_open(file_path, password=None):
        raise RuntimeError("cannot parse")

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    with pytest.raises(PdfTextExtractionError) as excinfo:
        await pdf_loader(str(pdf_path))

    assert "PDF text extraction failed: cannot parse" in str(excinfo.value)
```

- [ ] **Step 2: Run the new loader tests and verify they fail**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py -q
```

Expected: FAIL because `pdfplumber`, `PdfPageLimitError`, or `PdfNoExtractableTextError` is not implemented yet.

- [ ] **Step 3: Implement the PDF loader**

Modify `backend/app/utils/file_handler.py` as follows:

1. Replace the current loader import line:

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
```

with:

```python
import pdfplumber
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
```

2. Add these classes and helper near the top of the file after imports:

```python
PDF_MAX_PAGES = 20


class PdfTextExtractionError(Exception):
    pass


class PdfPageLimitError(PdfTextExtractionError):
    pass


class PdfNoExtractableTextError(PdfTextExtractionError):
    pass


def _load_pdf_documents(file_path: str, password: str | None = None, max_pages: int = PDF_MAX_PAGES) -> list[Document]:
    abs_file_path = get_abstract_path(file_path) if not os.path.isabs(file_path) else file_path

    if not os.path.exists(abs_file_path):
        raise PdfTextExtractionError(f"PDF file does not exist: {abs_file_path}")
    if not os.path.isfile(abs_file_path):
        raise PdfTextExtractionError(f"PDF path is not a file: {abs_file_path}")
    if os.path.getsize(abs_file_path) == 0:
        raise PdfTextExtractionError("PDF content is empty")

    try:
        with pdfplumber.open(abs_file_path, password=password) as pdf:
            page_count = len(pdf.pages)
            if page_count > max_pages:
                raise PdfPageLimitError(f"PDF page count {page_count} exceeds limit {max_pages}")

            documents: list[Document] = []
            for index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue

                text = page_text.strip()
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": abs_file_path,
                            "page": index,
                            "page_count": page_count,
                            "content_length": len(text),
                            "file_type": "pdf",
                        },
                    )
                )

            if not documents:
                raise PdfNoExtractableTextError("PDF contains no extractable text")

            return documents
    except PdfTextExtractionError:
        raise
    except Exception as exc:
        raise PdfTextExtractionError(f"PDF text extraction failed: {exc}") from exc
```

3. Replace the existing `pdf_loader()` implementation with:

```python
async def pdf_loader(file_path: str, password: str = None) -> list[Document]:
    """
    加载PDF文件内容
    :param file_path: PDF文件路径
    :param password: PDF密码（如果有）
    :return: PDF文件内容
    """
    return await asyncio.to_thread(_load_pdf_documents, file_path, password, PDF_MAX_PAGES)
```

- [ ] **Step 4: Run the loader tests and verify they pass**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py -q
```

Expected: PASS for all tests in `test_pdf_loader.py`.

---

### Task 3: Propagate upload failures and preserve uploaded filenames

**Files:**
- Create: `backend/tests/test_vector_store_pdf_upload.py`
- Modify: `backend/app/rag/vector_store.py`

- [ ] **Step 1: Write failing vector upload tests**

Create `backend/tests/test_vector_store_pdf_upload.py` with this content:

```python
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes = b"pdf bytes"):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class FakeSplitter:
    async def split_documents(self, documents):
        return documents


class FakeVectorStore:
    def __init__(self):
        self.added_documents = []

    def add_documents(self, documents):
        self.added_documents.extend(documents)


def build_service(monkeypatch, duplicate=False):
    from app.rag.vector_store import VectorStoreService

    service = object.__new__(VectorStoreService)
    service.spliter = FakeSplitter()
    service.vectors_store = FakeVectorStore()

    async def check_md5_hex(md5_hex):
        return duplicate

    async def save_md5_hex(md5_hex):
        service.saved_md5 = md5_hex

    service.check_md5_hex = check_md5_hex
    service.save_md5_hex = save_md5_hex
    return service


@pytest.mark.anyio
async def test_upload_failure_raises_http_400_and_cleans_temp_file(monkeypatch):
    from app.rag.vector_store import VectorStoreService
    from app.utils.file_handler import PdfNoExtractableTextError

    service = build_service(monkeypatch)
    observed_paths = []

    async def fake_get_file_document(file_path):
        observed_paths.append(file_path)
        assert Path(file_path).exists()
        raise PdfNoExtractableTextError("PDF contains no extractable text")

    service.get_file_document = fake_get_file_document

    with pytest.raises(HTTPException) as excinfo:
        await VectorStoreService.get_document(service, files=[FakeUploadFile("scan.pdf")], user_id="user-1")

    assert excinfo.value.status_code == 400
    assert "scan.pdf" in excinfo.value.detail
    assert "PDF contains no extractable text" in excinfo.value.detail
    assert observed_paths
    assert not Path(observed_paths[0]).exists()


@pytest.mark.anyio
async def test_upload_success_adds_user_id_and_original_filename_metadata(monkeypatch):
    from app.rag.vector_store import VectorStoreService

    service = build_service(monkeypatch)
    observed_paths = []

    async def fake_get_file_document(file_path):
        observed_paths.append(file_path)
        return [
            Document(
                page_content="PDF text",
                metadata={"file_type": "pdf", "page": 1, "page_count": 1},
            )
        ]

    service.get_file_document = fake_get_file_document

    await VectorStoreService.get_document(service, files=[FakeUploadFile("handbook.pdf")], user_id="user-1")

    assert len(service.vectors_store.added_documents) == 1
    metadata = service.vectors_store.added_documents[0].metadata
    assert metadata["user_id"] == "user-1"
    assert metadata["filename"] == "handbook.pdf"
    assert metadata["file_type"] == "pdf"
    assert observed_paths
    assert not Path(observed_paths[0]).exists()


@pytest.mark.anyio
async def test_duplicate_upload_skips_without_error_and_cleans_temp_file(monkeypatch):
    from app.rag.vector_store import VectorStoreService

    service = build_service(monkeypatch, duplicate=True)
    called_loader = False
    observed_paths = []

    async def fake_get_file_document(file_path):
        nonlocal called_loader
        called_loader = True
        return []

    async def fake_check_md5_hex(md5_hex):
        return True

    original_unlink = os.unlink

    def recording_unlink(file_path):
        observed_paths.append(file_path)
        original_unlink(file_path)

    service.get_file_document = fake_get_file_document
    service.check_md5_hex = fake_check_md5_hex
    monkeypatch.setattr("app.rag.vector_store.os.unlink", recording_unlink)

    await VectorStoreService.get_document(service, files=[FakeUploadFile("duplicate.pdf")], user_id="user-1")

    assert called_loader is False
    assert service.vectors_store.added_documents == []
    assert observed_paths
    assert not Path(observed_paths[0]).exists()
```

- [ ] **Step 2: Run the upload tests and verify they fail**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: FAIL because current `get_document()` swallows upload errors, does not attach `filename`, and does not centralize cleanup in `finally`.

- [ ] **Step 3: Update imports in `vector_store.py`**

In `backend/app/rag/vector_store.py`, add `HTTPException` to imports:

```python
from fastapi import HTTPException
```

Keep existing imports otherwise.

- [ ] **Step 4: Replace `get_document()` with upload-aware cleanup and failure propagation**

Replace the entire `async def get_document(...)` method in `backend/app/rag/vector_store.py` with:

```python
    async def get_document(self, files: list = None, user_id: str = None):
        """
        处理文档并将其转为向量存入向量数据库
        :param files: 上传的文件列表，如果为None则从数据文件夹读取
        :param user_id: 用户ID，用于标记文档的所有者
        """
        file_entries: list[tuple[str, str | None]] = []
        upload_mode = bool(files)

        if files:
            for file in files:
                temp_file = await asyncio.to_thread(
                    tempfile.NamedTemporaryFile,
                    delete=False,
                    suffix=os.path.splitext(file.filename)[1],
                )
                try:
                    content = await file.read()
                    await asyncio.to_thread(temp_file.write, content)
                    await asyncio.to_thread(temp_file.flush)
                    await asyncio.to_thread(temp_file.close)
                except Exception:
                    await asyncio.to_thread(temp_file.close)
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                    raise

                file_entries.append((temp_file.name, file.filename))
        else:
            allowed_file_path: tuple[str] = await listdir_allowed_type(
                chroma_config['data_path'],
                tuple(chroma_config['allow_knowledge_file_types'])
            )
            file_entries = [(file_path, None) for file_path in allowed_file_path]

        for file_path, original_filename in file_entries:
            display_name = original_filename or file_path
            try:
                md5_hex = await get_file_md5_hex(file_path)
                if await self.check_md5_hex(md5_hex):
                    logger.info(f"【向量数据库】文件 {display_name} 的md5值 {md5_hex} 已存在，跳过")
                    continue

                document: list[Document] = await self.get_file_document(file_path)
                if not document:
                    raise ValueError(f"文件 {display_name} 加载内容为空")

                document = await self.spliter.split_documents(document)
                if not document:
                    raise ValueError(f"文件 {display_name} 切分内容为空")

                for doc in document:
                    if user_id:
                        doc.metadata['user_id'] = user_id
                    if original_filename:
                        doc.metadata['filename'] = original_filename

                await asyncio.to_thread(self.vectors_store.add_documents, document)
                await self.save_md5_hex(md5_hex)
                logger.info(f"【向量数据库】文件 {display_name} 的md5值 {md5_hex} 已保存")
            except Exception as e:
                logger.error(f"【向量数据库】文件 {display_name} 处理时出错: {e}")
                if upload_mode:
                    raise HTTPException(status_code=400, detail=f"文件 {display_name} 处理失败: {e}") from e
                continue
            finally:
                if upload_mode and os.path.exists(file_path):
                    os.unlink(file_path)
```

- [ ] **Step 5: Run the upload tests and verify they pass**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: PASS for all tests in `test_vector_store_pdf_upload.py`.

---

### Task 4: Run combined PDF import tests and fix integration issues

**Files:**
- Verify: `backend/tests/test_pdf_loader.py`
- Verify: `backend/tests/test_vector_store_pdf_upload.py`
- Modify only if needed: `backend/app/utils/file_handler.py`, `backend/app/rag/vector_store.py`

- [ ] **Step 1: Run both new test files together**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: PASS.

- [ ] **Step 2: If any `anyio` backend tries Trio and fails, constrain tests to asyncio**

If the test output includes a missing Trio backend error, add this fixture near the top of each new test file after the path setup:

```python
@pytest.fixture
def anyio_backend():
    return "asyncio"
```

Then rerun:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: PASS.

- [ ] **Step 3: If `pdfplumber` is not installed in the active environment, install/update dependencies through the project environment**

Run:

```powershell
conda run -n NexusKB pip install -r backend/requirements.txt
```

Expected: completes successfully and installs `pdfplumber` if missing.

Then rerun:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: PASS.

---

### Task 5: Update the changelog for the implemented backend change

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add an implementation entry under `2026-06-16 - Unreleased`**

Add this bullet immediately after the existing PDF design bullet:

```markdown
- Implemented backend PDF import hardening with `pdfplumber` extraction, a 20-page limit, page-level metadata, upload filename metadata, upload failure propagation, and temporary-file cleanup tests.
```

Expected top of changelog:

```markdown
## 2026-06-16 - Unreleased

- Added `docs/superpowers/specs/2026-06-17-pdf-import-design.md` to define the planned backend PDF import enhancement with `pdfplumber`, a 20-page limit, page metadata, and upload failure handling.
- Implemented backend PDF import hardening with `pdfplumber` extraction, a 20-page limit, page-level metadata, upload filename metadata, upload failure propagation, and temporary-file cleanup tests.
```

- [ ] **Step 2: Verify the changelog entry exists**

Run:

```powershell
Select-String -Path CHANGELOG.md -Pattern "backend PDF import hardening"
```

Expected: one match.

---

### Task 6: Run regression tests for the touched backend area

**Files:**
- Verify: `backend/tests/test_pdf_loader.py`
- Verify: `backend/tests/test_vector_store_pdf_upload.py`
- Verify: existing backend tests where practical

- [ ] **Step 1: Run focused tests**

Run:

```powershell
conda run -n NexusKB pytest backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full backend test suite**

Run:

```powershell
conda run -n NexusKB pytest backend/tests -q
```

Expected: PASS, or report any unrelated pre-existing failures with the failing test names and error summaries.

- [ ] **Step 3: Inspect the working tree**

Run:

```powershell
git status --short
```

Expected: modified/created files are limited to:

```text
 M CHANGELOG.md
 M backend/app/rag/vector_store.py
 M backend/app/utils/file_handler.py
 M backend/pyproject.toml
 M backend/requirements.txt
?? backend/tests/test_pdf_loader.py
?? backend/tests/test_vector_store_pdf_upload.py
?? docs/superpowers/plans/2026-06-17-pdf-import-backend-enhancement.md
```

The existing untracked `pdf_text_extractor.py` may also appear and should not be staged unless the user explicitly asks.

- [ ] **Step 4: Do not commit unless the user explicitly asks**

If the user asks for a commit, stage only the relevant files:

```powershell
git add CHANGELOG.md backend/app/rag/vector_store.py backend/app/utils/file_handler.py backend/pyproject.toml backend/requirements.txt backend/tests/test_pdf_loader.py backend/tests/test_vector_store_pdf_upload.py docs/superpowers/plans/2026-06-17-pdf-import-backend-enhancement.md docs/superpowers/specs/2026-06-17-pdf-import-design.md
```

Then create a new commit with a HEREDOC-style message appropriate for the active shell. Do not stage `pdf_text_extractor.py` unless explicitly requested.

---

## Self-Review Against Spec

- Spec requirement: project-owned `pdfplumber` extraction flow. Covered by Task 1 and Task 2.
- Spec requirement: 20-page limit. Covered by Task 2 tests and `PDF_MAX_PAGES` implementation.
- Spec requirement: clear failures for empty, damaged, encrypted, oversized, and image-only PDFs. Covered by Task 2 tests and exception hierarchy; encrypted/damaged PDFs are handled by wrapped `pdfplumber` errors.
- Spec requirement: page-level metadata. Covered by Task 2 tests and loader implementation.
- Spec requirement: uploaded original filename metadata. Covered by Task 3 tests and `VectorStoreService.get_document()` metadata assignment.
- Spec requirement: upload failures return HTTP 400 instead of false success. Covered by Task 3 tests and upload-mode `HTTPException` propagation.
- Spec requirement: temporary files closed before reading and cleaned after success, duplicate skip, and failure. Covered by Task 3 implementation and tests.
- Spec requirement: local folder ingestion remains best-effort. Covered by Task 3 implementation using `upload_mode` to only raise for uploads.
- Spec requirement: dependencies and changelog. Covered by Task 1 and Task 5.

No placeholders remain; all new function names and exception names match across tests and implementation steps.
