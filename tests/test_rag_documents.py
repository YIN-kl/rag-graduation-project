from pathlib import Path

from langchain_core.documents import Document

import rag


def test_companion_text_file_is_skipped_when_pdf_exists(tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()
    pdf_path = folder / "差旅费用报销规定.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    companion_path = folder / "差旅费用报销规定-文本版.md"
    companion_path.write_text("# companion", encoding="utf-8")

    assert rag._is_companion_text_only_file(companion_path) is True


def test_pdf_falls_back_to_companion_text_when_extraction_is_poor(monkeypatch, tmp_path):
    pdf_path = tmp_path / "差旅费用报销规定.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    companion_path = tmp_path / "差旅费用报销规定-文本版.md"
    companion_path.write_text("# 差旅费用报销规定", encoding="utf-8")

    class FakePdfLoader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="????????????????", metadata={})]

    import langchain_community.document_loaders as loaders

    monkeypatch.setattr(loaders, "PyPDFLoader", FakePdfLoader)
    monkeypatch.setattr(
        rag,
        "_load_text_document_file",
        lambda path: [Document(page_content="中文整理版内容", metadata={})],
    )

    docs = rag._load_document_file(pdf_path)

    assert len(docs) == 1
    assert "中文整理版内容" in docs[0].page_content
    assert "文本整理版" in docs[0].page_content


def test_load_documents_indexes_pdf_and_skips_companion_markdown(monkeypatch, tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()
    pdf_path = folder / "差旅费用报销规定.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    companion_path = folder / "差旅费用报销规定-文本版.md"
    companion_path.write_text("# companion", encoding="utf-8")

    monkeypatch.setattr(
        rag,
        "_load_document_file",
        lambda path: [Document(page_content=path.name, metadata={})],
    )

    docs = rag.load_documents(str(folder))

    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "差旅费用报销规定.pdf"
