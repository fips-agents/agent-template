"""Tests for fipsagents.server.parser."""


from __future__ import annotations

import sys
import types

import pytest

from fipsagents.baseagent.config import ParserConfig, PdfParserConfig
from fipsagents.server.parser import (
    DoclingParser,
    FileParser,
    NullParser,
    ParseOutcome,
    PlaintextParser,
    _is_plaintext_mime,
    create_parser,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsPlaintextMime:
    @pytest.mark.parametrize("mime", [
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    ])
    def test_recognises_text_shaped(self, mime):
        assert _is_plaintext_mime(mime) is True

    @pytest.mark.parametrize("mime", [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/png",
        "image/jpeg",
        "application/octet-stream",
    ])
    def test_rejects_binary(self, mime):
        assert _is_plaintext_mime(mime) is False


class TestParseOutcome:
    def test_completed_factory(self):
        out = ParseOutcome.completed("hello")
        assert out.status == "completed"
        assert out.text == "hello"
        assert out.error is None

    def test_skipped_factory(self):
        out = ParseOutcome.skipped("not my type")
        assert out.status == "skipped"
        assert out.text is None
        assert out.error == "not my type"

    def test_skipped_no_reason(self):
        out = ParseOutcome.skipped()
        assert out.status == "skipped"
        assert out.error is None

    def test_failed_factory(self):
        out = ParseOutcome.failed("boom")
        assert out.status == "failed"
        assert out.text is None
        assert out.error == "boom"


# ---------------------------------------------------------------------------
# NullParser
# ---------------------------------------------------------------------------


class TestNullParser:
    @pytest.mark.asyncio
    async def test_skips_everything(self):
        parser = NullParser()
        out = await parser.parse(
            b"hello", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "skipped"

    @pytest.mark.asyncio
    async def test_skips_binary(self):
        parser = NullParser()
        out = await parser.parse(
            b"\x00\x01", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"


# ---------------------------------------------------------------------------
# PlaintextParser
# ---------------------------------------------------------------------------


class TestPlaintextParser:
    @pytest.mark.asyncio
    async def test_decodes_utf8(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b"hello world", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "hello world"

    @pytest.mark.asyncio
    async def test_decodes_unicode(self):
        parser = PlaintextParser()
        body = "café — résumé".encode("utf-8")
        out = await parser.parse(
            body, mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "café — résumé"

    @pytest.mark.asyncio
    async def test_replaces_invalid_bytes(self):
        parser = PlaintextParser()
        # 0xff is invalid UTF-8 mid-stream — should become U+FFFD.
        out = await parser.parse(
            b"hello\xffworld", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert "hello" in out.text
        assert "world" in out.text

    @pytest.mark.asyncio
    async def test_skips_pdf(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b"%PDF-1.4", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"
        assert "not text-shaped" in out.error

    @pytest.mark.asyncio
    async def test_handles_json(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b'{"hello": "world"}',
            mime_type="application/json",
            filename="x.json",
        )
        assert out.status == "completed"
        assert "hello" in out.text


# ---------------------------------------------------------------------------
# DoclingParser
# ---------------------------------------------------------------------------


class _FakeDoclingDoc:
    def __init__(self, text: str) -> None:
        self._text = text

    def export_to_markdown(self) -> str:
        return self._text


class _FakeDoclingResult:
    def __init__(self, text: str) -> None:
        self.document = _FakeDoclingDoc(text)


class _FakeConverter:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[str] = []

    def convert(self, path: str) -> _FakeDoclingResult:
        self.calls.append(path)
        return _FakeDoclingResult(self._text)


class TestDoclingParser:
    @pytest.mark.asyncio
    async def test_falls_through_to_plaintext(self):
        # Plaintext shouldn't even invoke docling.
        parser = DoclingParser()
        out = await parser.parse(
            b"hi there", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "hi there"
        # Converter never lazy-loaded.
        assert parser._converter is None

    @pytest.mark.asyncio
    async def test_invokes_converter_for_pdf(self):
        parser = DoclingParser()
        fake = _FakeConverter("# Heading\n\nbody text")
        parser._converter = fake  # bypass the lazy import

        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "completed"
        assert out.text == "# Heading\n\nbody text"
        # Confirm the temp path docling saw kept the .pdf suffix.
        assert len(fake.calls) == 1
        assert fake.calls[0].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_failed_converter_returns_failed_outcome(self):
        parser = DoclingParser()

        class _Boom:
            def convert(self, path):
                raise RuntimeError("docling exploded")

        parser._converter = _Boom()

        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "failed"
        assert "RuntimeError" in out.error
        assert "docling exploded" in out.error

    @pytest.mark.asyncio
    async def test_empty_text_marks_skipped(self):
        parser = DoclingParser()
        parser._converter = _FakeConverter("")
        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"
        assert "empty text" in out.error

    @pytest.mark.asyncio
    async def test_extension_inferred_from_mime_when_filename_lacks_one(self):
        parser = DoclingParser()
        fake = _FakeConverter("body")
        parser._converter = fake

        await parser.parse(
            b"data", mime_type="application/pdf", filename="noextension",
        )
        assert fake.calls[0].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_lazy_import_raises_clear_error(self, monkeypatch):
        """Touching ._ensure_converter without docling installed should
        raise an ImportError that mentions the [files] extra."""
        parser = DoclingParser()
        # Simulate docling missing even if the host happens to have it.
        monkeypatch.setitem(sys.modules, "docling.document_converter", None)
        with pytest.raises(ImportError, match=r"\[files\] extra"):
            parser._ensure_converter()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_disabled_returns_null(self):
        parser = create_parser(enabled=False)
        assert isinstance(parser, NullParser)

    def test_enabled_with_docling_returns_docling(self, monkeypatch):
        # Pretend docling is importable.
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: True,
        )
        parser = create_parser(enabled=True)
        assert isinstance(parser, DoclingParser)

    def test_enabled_without_docling_returns_plaintext(self, monkeypatch):
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: False,
        )
        parser = create_parser(enabled=True)
        assert isinstance(parser, PlaintextParser)

    def test_real_environment_resolves_to_some_parser(self):
        """Smoke test against the actual environment — docling installed
        or not, factory must return a usable FileParser."""
        parser = create_parser(enabled=True)
        assert isinstance(parser, FileParser)


# ---------------------------------------------------------------------------
# Pipeline option propagation (issue #146)
# ---------------------------------------------------------------------------


class _FakeInputFormat:
    PDF = "pdf-marker"


class _FakePdfPipelineOptions:
    def __init__(self, *, do_ocr, do_table_structure):
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure


class _FakePdfFormatOption:
    def __init__(self, *, pipeline_options):
        self.pipeline_options = pipeline_options


class _RecordingDocumentConverter:
    instances: list["_RecordingDocumentConverter"] = []

    def __init__(self, format_options=None):
        self.format_options = format_options
        type(self).instances.append(self)


@pytest.fixture
def fake_docling(monkeypatch):
    """Inject minimal docling stand-ins so DoclingParser._ensure_converter
    can run without the [files] extra installed.

    Resets the per-test instance log so each test starts clean.
    """
    docling_pkg = types.ModuleType("docling")
    datamodel_pkg = types.ModuleType("docling.datamodel")
    base_models = types.ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = _FakeInputFormat
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = _FakePdfPipelineOptions
    document_converter = types.ModuleType("docling.document_converter")
    document_converter.DocumentConverter = _RecordingDocumentConverter
    document_converter.PdfFormatOption = _FakePdfFormatOption

    monkeypatch.setitem(sys.modules, "docling", docling_pkg)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel_pkg)
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.base_models", base_models,
    )
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.pipeline_options", pipeline_options,
    )
    monkeypatch.setitem(
        sys.modules, "docling.document_converter", document_converter,
    )
    _RecordingDocumentConverter.instances = []
    yield _RecordingDocumentConverter
    _RecordingDocumentConverter.instances = []


class TestPipelineOptions:
    def test_default_config_flips_do_ocr_off(self):
        """Framework default is do_ocr=False (issue #146)."""
        cfg = ParserConfig()
        assert cfg.pdf.do_ocr is False
        assert cfg.pdf.do_table_structure is True

    def test_no_config_keeps_no_args_constructor(self, fake_docling):
        """Backward-compat path: DoclingParser() with no config builds
        DocumentConverter() with no args, leaving Docling's own defaults
        in place."""
        parser = DoclingParser()
        parser._ensure_converter()
        assert len(fake_docling.instances) == 1
        assert fake_docling.instances[0].format_options is None

    @pytest.mark.parametrize("do_ocr", [True, False])
    @pytest.mark.parametrize("do_table_structure", [True, False])
    def test_pdf_options_propagate_to_converter(
        self, fake_docling, do_ocr, do_table_structure,
    ):
        cfg = ParserConfig(
            pdf=PdfParserConfig(
                do_ocr=do_ocr, do_table_structure=do_table_structure,
            ),
        )
        parser = DoclingParser(parser_config=cfg)
        parser._ensure_converter()

        assert len(fake_docling.instances) == 1
        format_options = fake_docling.instances[0].format_options
        assert format_options is not None
        assert _FakeInputFormat.PDF in format_options

        pdf_format_option = format_options[_FakeInputFormat.PDF]
        options = pdf_format_option.pipeline_options
        assert options.do_ocr is do_ocr
        assert options.do_table_structure is do_table_structure

    def test_factory_forwards_parser_config(self, monkeypatch, fake_docling):
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: True,
        )
        cfg = ParserConfig(pdf=PdfParserConfig(do_ocr=True))
        parser = create_parser(enabled=True, parser_config=cfg)
        assert isinstance(parser, DoclingParser)

        # Lazy-load to confirm the config reaches DocumentConverter.
        parser._ensure_converter()
        assert len(fake_docling.instances) == 1
        options = fake_docling.instances[0].format_options[_FakeInputFormat.PDF]
        assert options.pipeline_options.do_ocr is True

    def test_factory_without_config_omits_format_options(
        self, monkeypatch, fake_docling,
    ):
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: True,
        )
        parser = create_parser(enabled=True)
        assert isinstance(parser, DoclingParser)
        parser._ensure_converter()
        assert fake_docling.instances[0].format_options is None
