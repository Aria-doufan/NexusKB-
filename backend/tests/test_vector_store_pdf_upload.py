import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes = b"%PDF-1.7\nplaceholder"):
        self.filename = filename
        self._content = content
        self.read_calls = 0

    async def read(self):
        self.read_calls += 1
        return self._content


class ReadFailUploadFile(FakeUploadFile):
    async def read(self):
        self.read_calls += 1
        raise OSError("simulated read failure")


class FakeSplitter:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    async def split_documents(self, documents):
        self.calls.append(documents)
        return self.documents


class FakeVectorStore:
    def __init__(self):
        self.added_documents = []

    def add_documents(self, documents):
        self.added_documents.extend(documents)


class FailingVectorStore(FakeVectorStore):
    def add_documents(self, documents):
        raise RuntimeError("vector database unavailable")


class FakeBM25Retriever:
    created_instances = []

    def __init__(self, documents, k):
        self.documents = documents
        self.k = k

    @classmethod
    def from_documents(cls, documents, k):
        instance = cls(documents=documents, k=k)
        cls.created_instances.append(instance)
        return instance


async def _async_noop(*args, **kwargs):
    return None


def build_fake_service(split_documents=None):
    from app.rag.vector_store import VectorStoreService

    service = object.__new__(VectorStoreService)
    service.spliter = FakeSplitter(split_documents or [])
    service.vectors_store = FakeVectorStore()
    service.saved_md5 = []

    async def save_md5_hex(md5_hex):
        service.saved_md5.append(md5_hex)

    service.save_md5_hex = save_md5_hex
    return service


@pytest.mark.anyio
async def test_upload_pdf_parse_failure_raises_400_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    upload = FakeUploadFile("broken.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "broken-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        raise vector_store.PdfTextExtractionError("PDF text extraction failed: cannot parse")

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    with pytest.raises(HTTPException) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert exc_info.value.status_code == 400
    assert "broken.pdf" in exc_info.value.detail
    assert "PDF text extraction failed: cannot parse" in exc_info.value.detail
    assert temp_paths
    assert not os.path.exists(temp_paths[0])
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_empty_loader_result_raises_400_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service(split_documents=[Document(page_content="chunk")])
    upload = FakeUploadFile("empty-loader.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "empty-loader-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        return []

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    with pytest.raises(HTTPException) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert exc_info.value.status_code == 400
    assert "empty-loader.pdf" in exc_info.value.detail
    assert "文件加载内容为空" in exc_info.value.detail
    assert temp_paths
    assert not os.path.exists(temp_paths[0])
    assert service.spliter.calls == []
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_empty_splitter_result_raises_400_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service(split_documents=[])
    upload = FakeUploadFile("empty-splitter.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "empty-splitter-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        return [Document(page_content="raw")]

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    with pytest.raises(HTTPException) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert exc_info.value.status_code == 400
    assert "empty-splitter.pdf" in exc_info.value.detail
    assert "文件切分内容为空" in exc_info.value.detail
    assert temp_paths
    assert not os.path.exists(temp_paths[0])
    assert len(service.spliter.calls) == 1
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_save_md5_failure_is_not_mapped_to_400_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    split_document = Document(page_content="chunk", metadata={"page": 1})
    service = build_fake_service(split_documents=[split_document])
    upload = FakeUploadFile("save-md5-fails.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "save-md5-fails-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        return [Document(page_content="raw")]

    async def save_md5_hex(md5_hex):
        raise RuntimeError("md5 store unavailable")

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document
    service.save_md5_hex = save_md5_hex

    with pytest.raises(RuntimeError, match="md5 store unavailable") as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert not isinstance(exc_info.value, HTTPException)
    assert temp_paths
    assert not os.path.exists(temp_paths[0])
    assert len(service.vectors_store.added_documents) == 1
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_empty_upload_list_does_not_ingest_local_folder(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    listed_local_folder = False

    async def listdir_allowed_type(*args, **kwargs):
        nonlocal listed_local_folder
        listed_local_folder = True
        return ["local.pdf"]

    monkeypatch.setattr(vector_store, "listdir_allowed_type", listdir_allowed_type)

    await service.get_document(files=[], user_id="user-1")

    assert not listed_local_folder
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_multi_file_upload_stops_before_staging_second_file_and_cleans_first_temp(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    first_upload = FakeUploadFile("first-broken.pdf")
    second_upload = FakeUploadFile("second-not-staged.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "first-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        raise vector_store.PdfTextExtractionError("first file failed before second upload was staged")

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    with pytest.raises(HTTPException) as exc_info:
        await service.get_document(files=[first_upload, second_upload], user_id="user-1")

    assert exc_info.value.status_code == 400
    assert "first-broken.pdf" in exc_info.value.detail
    assert first_upload.read_calls == 1
    assert second_upload.read_calls == 0
    assert len(temp_paths) == 1
    assert not os.path.exists(temp_paths[0])
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_staging_read_failure_is_not_mapped_to_400_and_cleans_temp(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    upload = ReadFailUploadFile("unreadable.pdf")
    temp_paths = []
    real_named_temporary_file = vector_store.tempfile.NamedTemporaryFile

    def tracking_named_temporary_file(*args, **kwargs):
        temp_file = real_named_temporary_file(*args, **kwargs)
        temp_paths.append(temp_file.name)
        return temp_file

    async def fail_if_processing_starts(file_path):
        pytest.fail("staging failure should not reach md5 processing")

    monkeypatch.setattr(vector_store.tempfile, "NamedTemporaryFile", tracking_named_temporary_file)
    monkeypatch.setattr(vector_store, "get_file_md5_hex", fail_if_processing_starts)

    with pytest.raises(OSError) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert not isinstance(exc_info.value, HTTPException)
    assert "simulated read failure" in str(exc_info.value)
    assert upload.read_calls == 1
    assert temp_paths
    assert all(not os.path.exists(temp_path) for temp_path in temp_paths)
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_staging_temp_write_failure_is_not_mapped_to_400_and_cleans_temp(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    upload = FakeUploadFile("write-fails.pdf")
    temp_paths = []
    real_named_temporary_file = vector_store.tempfile.NamedTemporaryFile

    class WriteFailingTempFile:
        def __init__(self, wrapped_file):
            self._wrapped_file = wrapped_file
            self.name = wrapped_file.name

        def write(self, content):
            raise OSError("simulated temp write failure")

        def flush(self):
            return self._wrapped_file.flush()

        def close(self):
            return self._wrapped_file.close()

    def write_failing_named_temporary_file(*args, **kwargs):
        temp_file = real_named_temporary_file(*args, **kwargs)
        temp_paths.append(temp_file.name)
        return WriteFailingTempFile(temp_file)

    async def fail_if_processing_starts(file_path):
        pytest.fail("server staging failure should not reach md5 processing")

    monkeypatch.setattr(vector_store.tempfile, "NamedTemporaryFile", write_failing_named_temporary_file)
    monkeypatch.setattr(vector_store, "get_file_md5_hex", fail_if_processing_starts)

    with pytest.raises(OSError, match="simulated temp write failure"):
        await service.get_document(files=[upload], user_id="user-1")

    assert upload.read_calls == 1
    assert temp_paths
    assert all(not os.path.exists(temp_path) for temp_path in temp_paths)
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_staging_temp_creation_failure_is_not_mapped_to_400(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service()
    upload = FakeUploadFile("temp-create-fails.pdf")

    def create_failing_named_temporary_file(*args, **kwargs):
        raise OSError("simulated temp creation failure")

    async def fail_if_processing_starts(file_path):
        pytest.fail("server staging failure should not reach md5 processing")

    monkeypatch.setattr(vector_store.tempfile, "NamedTemporaryFile", create_failing_named_temporary_file)
    monkeypatch.setattr(vector_store, "get_file_md5_hex", fail_if_processing_starts)

    with pytest.raises(OSError) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert not isinstance(exc_info.value, HTTPException)
    assert "simulated temp creation failure" in str(exc_info.value)
    assert upload.read_calls == 0
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("flush", "simulated temp flush failure"),
        ("close", "simulated temp close failure"),
    ],
)
async def test_upload_staging_flush_or_close_failure_is_not_mapped_to_400_and_cleans_temp(
    monkeypatch,
    operation,
    message,
):
    from app.rag import vector_store

    service = build_fake_service()
    upload = FakeUploadFile(f"{operation}-fails.pdf")
    temp_paths = []
    real_named_temporary_file = vector_store.tempfile.NamedTemporaryFile

    class OperationFailingTempFile:
        def __init__(self, wrapped_file):
            self._wrapped_file = wrapped_file
            self.name = wrapped_file.name

        def write(self, content):
            return self._wrapped_file.write(content)

        def flush(self):
            if operation == "flush":
                raise OSError(message)
            return self._wrapped_file.flush()

        def close(self):
            self._wrapped_file.close()
            if operation == "close":
                raise OSError(message)
            return None

    def operation_failing_named_temporary_file(*args, **kwargs):
        temp_file = real_named_temporary_file(*args, **kwargs)
        temp_paths.append(temp_file.name)
        return OperationFailingTempFile(temp_file)

    async def fail_if_processing_starts(file_path):
        pytest.fail("server staging failure should not reach md5 processing")

    monkeypatch.setattr(vector_store.tempfile, "NamedTemporaryFile", operation_failing_named_temporary_file)
    monkeypatch.setattr(vector_store, "get_file_md5_hex", fail_if_processing_starts)

    with pytest.raises(OSError) as exc_info:
        await service.get_document(files=[upload], user_id="user-1")

    assert not isinstance(exc_info.value, HTTPException)
    assert message in str(exc_info.value)
    assert upload.read_calls == 1
    assert temp_paths
    assert all(not os.path.exists(temp_path) for temp_path in temp_paths)
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []


@pytest.mark.anyio
async def test_upload_vector_store_add_failure_is_not_mapped_to_400_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    split_document = Document(page_content="chunk", metadata={"page": 1})
    service = build_fake_service(split_documents=[split_document])
    service.vectors_store = FailingVectorStore()
    upload = FakeUploadFile("source.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "source-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        return [Document(page_content="raw", metadata={"source": file_path})]

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    with pytest.raises(RuntimeError, match="vector database unavailable"):
        await service.get_document(files=[upload], user_id="user-123")

    assert service.saved_md5 == []
    assert temp_paths
    assert not os.path.exists(temp_paths[0])


@pytest.mark.anyio
async def test_upload_success_attaches_metadata_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    split_document = Document(page_content="chunk", metadata={"page": 1})
    service = build_fake_service(split_documents=[split_document])
    upload = FakeUploadFile("source.pdf")
    temp_paths = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "source-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def get_file_document(file_path):
        assert os.path.exists(file_path)
        return [Document(page_content="raw", metadata={"source": file_path})]

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    await service.get_document(files=[upload], user_id="user-123")

    assert len(service.vectors_store.added_documents) == 1
    stored_document = service.vectors_store.added_documents[0]
    assert stored_document.metadata["user_id"] == "user-123"
    assert stored_document.metadata["filename"] == "source.pdf"
    assert stored_document.metadata["page"] == 1
    assert service.saved_md5 == ["source-md5"]
    assert temp_paths
    assert not os.path.exists(temp_paths[0])


@pytest.mark.anyio
async def test_upload_uppercase_pdf_extension_uses_pdf_loader_and_preserves_filename(monkeypatch):
    from app.rag import vector_store

    split_document = Document(page_content="chunk", metadata={"page": 1})
    service = build_fake_service(split_documents=[split_document])
    upload = FakeUploadFile("REPORT.PDF")
    temp_paths = []
    pdf_loader_calls = []

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        assert file_path.endswith(".pdf")
        assert not file_path.endswith(".PDF")
        return "report-md5"

    async def check_md5_hex(md5_hex):
        return False

    async def fake_pdf_loader(file_path):
        pdf_loader_calls.append(file_path)
        assert os.path.exists(file_path)
        assert file_path.endswith(".pdf")
        return [Document(page_content="raw", metadata={"source": file_path})]

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    monkeypatch.setattr(vector_store, "pdf_loader", fake_pdf_loader)
    service.check_md5_hex = check_md5_hex

    await service.get_document(files=[upload], user_id="user-123")

    assert len(pdf_loader_calls) == 1
    assert pdf_loader_calls == temp_paths
    assert len(service.vectors_store.added_documents) == 1
    stored_document = service.vectors_store.added_documents[0]
    assert stored_document.metadata["user_id"] == "user-123"
    assert stored_document.metadata["filename"] == "REPORT.PDF"
    assert stored_document.metadata["page"] == 1
    assert service.saved_md5 == ["report-md5"]
    assert temp_paths
    assert not os.path.exists(temp_paths[0])


@pytest.mark.anyio
async def test_duplicate_md5_upload_skips_loader_add_and_cleans_temp_file(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service(split_documents=[Document(page_content="chunk")])
    upload = FakeUploadFile("duplicate.pdf")
    temp_paths = []
    loader_called = False

    async def fake_get_file_md5_hex(file_path):
        temp_paths.append(file_path)
        assert os.path.exists(file_path)
        return "duplicate-md5"

    async def check_md5_hex(md5_hex):
        assert md5_hex == "duplicate-md5"
        return True

    async def get_file_document(file_path):
        nonlocal loader_called
        loader_called = True
        return [Document(page_content="raw")]

    monkeypatch.setattr(vector_store, "get_file_md5_hex", fake_get_file_md5_hex)
    service.check_md5_hex = check_md5_hex
    service.get_file_document = get_file_document

    await service.get_document(files=[upload], user_id="user-123")

    assert not loader_called
    assert service.spliter.calls == []
    assert service.vectors_store.added_documents == []
    assert service.saved_md5 == []
    assert temp_paths
    assert not os.path.exists(temp_paths[0])


@pytest.mark.anyio
async def test_bm25_retriever_skips_bad_local_pdf_and_uses_valid_local_document(monkeypatch):
    from app.rag import vector_store

    split_document = Document(page_content="valid local chunk", metadata={"source": "valid.txt"})
    service = build_fake_service(split_documents=[split_document])
    FakeBM25Retriever.created_instances = []

    async def listdir_allowed_type(*args, **kwargs):
        return ["bad.pdf", "valid.txt"]

    async def get_file_document(file_path):
        if file_path == "bad.pdf":
            raise vector_store.PdfTextExtractionError("PDF text extraction failed: scanned PDF has no text")
        return [Document(page_content="valid local raw", metadata={"source": file_path})]

    monkeypatch.setattr(vector_store, "listdir_allowed_type", listdir_allowed_type)
    monkeypatch.setattr(vector_store, "BM25Retriever", FakeBM25Retriever)
    service.get_file_document = get_file_document

    retriever = await service.get_bm25_retriever()

    assert retriever is FakeBM25Retriever.created_instances[0]
    assert retriever.documents == [split_document]
    assert retriever.k == vector_store.chroma_config["k"]
    assert len(service.spliter.calls) == 1
    assert service.spliter.calls[0][0].page_content == "valid local raw"


@pytest.mark.anyio
async def test_bm25_retriever_returns_none_when_all_local_files_fail(monkeypatch):
    from app.rag import vector_store

    service = build_fake_service(split_documents=[Document(page_content="unused chunk")])

    async def listdir_allowed_type(*args, **kwargs):
        return ["bad.pdf", "also-bad.txt"]

    async def get_file_document(file_path):
        raise vector_store.PdfTextExtractionError(f"failed to load {file_path}")

    class FailingBM25Retriever:
        @classmethod
        def from_documents(cls, documents, k):
            pytest.fail("BM25 retriever should not be built when every local file fails")

    monkeypatch.setattr(vector_store, "listdir_allowed_type", listdir_allowed_type)
    monkeypatch.setattr(vector_store, "BM25Retriever", FailingBM25Retriever)
    service.get_file_document = get_file_document

    retriever = await service.get_bm25_retriever()

    assert retriever is None
    assert service.spliter.calls == []
