import importlib.machinery
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def fake_pdfplumber_module(monkeypatch):
    module = types.ModuleType("pdfplumber")
    module.__spec__ = importlib.machinery.ModuleSpec("pdfplumber", loader=None)
    module.open = None
    monkeypatch.setitem(sys.modules, "pdfplumber", module)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdf:
    def __init__(self, page_texts):
        self.pages = [FakePage(text) for text in page_texts]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def write_pdf_placeholder(tmp_path: Path) -> str:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nplaceholder")
    return str(pdf_path)


@pytest.mark.anyio
async def test_pdf_loader_returns_document_per_text_page_with_metadata(tmp_path, monkeypatch):
    from app.utils import file_handler

    pdf_path = write_pdf_placeholder(tmp_path)

    def fake_open(file_path, password=None):
        assert file_path == pdf_path
        assert password is None
        return FakePdf([" First page text ", "Second page text"])

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    documents = await file_handler.pdf_loader(pdf_path)

    assert [document.page_content for document in documents] == [
        "First page text",
        "Second page text",
    ]
    assert [document.metadata for document in documents] == [
        {
            "source": pdf_path,
            "page": 1,
            "page_count": 2,
            "content_length": len("First page text"),
            "file_type": "pdf",
        },
        {
            "source": pdf_path,
            "page": 2,
            "page_count": 2,
            "content_length": len("Second page text"),
            "file_type": "pdf",
        },
    ]


@pytest.mark.anyio
async def test_pdf_loader_preserves_one_based_page_numbers_when_blank_pages_are_skipped(tmp_path, monkeypatch):
    from app.utils import file_handler

    pdf_path = write_pdf_placeholder(tmp_path)

    def fake_open(file_path, password=None):
        return FakePdf(["Page one", "   ", None, "Page four"])

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    documents = await file_handler.pdf_loader(pdf_path)

    assert [document.page_content for document in documents] == ["Page one", "Page four"]
    assert [document.metadata["page"] for document in documents] == [1, 4]
    assert [document.metadata["page_count"] for document in documents] == [4, 4]


@pytest.mark.anyio
async def test_pdf_loader_raises_page_limit_error_over_twenty_pages(tmp_path, monkeypatch):
    from app.utils import file_handler

    pdf_path = write_pdf_placeholder(tmp_path)

    def fake_open(file_path, password=None):
        return FakePdf(["text"] * 21)

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    with pytest.raises(file_handler.PdfPageLimitError, match="PDF page count 21 exceeds limit 20"):
        await file_handler.pdf_loader(pdf_path)


@pytest.mark.anyio
async def test_pdf_loader_raises_no_extractable_text_error(tmp_path, monkeypatch):
    from app.utils import file_handler

    pdf_path = write_pdf_placeholder(tmp_path)

    def fake_open(file_path, password=None):
        return FakePdf(["", "   ", None])

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    with pytest.raises(file_handler.PdfNoExtractableTextError, match="PDF contains no extractable text"):
        await file_handler.pdf_loader(pdf_path)


@pytest.mark.anyio
async def test_pdf_loader_raises_text_extraction_error_for_empty_file(tmp_path):
    from app.utils import file_handler

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    with pytest.raises(file_handler.PdfTextExtractionError, match="PDF content is empty"):
        await file_handler.pdf_loader(str(pdf_path))


@pytest.mark.anyio
async def test_pdf_loader_wraps_pdfplumber_errors(tmp_path, monkeypatch):
    from app.utils import file_handler

    pdf_path = write_pdf_placeholder(tmp_path)

    def fake_open(file_path, password=None):
        raise ValueError("cannot parse")

    monkeypatch.setattr(file_handler.pdfplumber, "open", fake_open)

    with pytest.raises(
        file_handler.PdfTextExtractionError,
        match="PDF text extraction failed: cannot parse",
    ):
        await file_handler.pdf_loader(pdf_path)
