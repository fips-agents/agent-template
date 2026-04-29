"""Tests for fipsagents.server.chunker."""

from __future__ import annotations

import pytest

from fipsagents.server import chunker as chunker_mod
from fipsagents.server.chunker import (
    Chunk,
    NullChunker,
    RecursiveTokenChunker,
    _greedy_assemble,
    _hard_split_by_tokens,
    _recursive_units,
    _split_paragraphs,
    _split_sentences,
    _take_overlap,
    count_tokens,
    create_chunker,
)


# ---------------------------------------------------------------------------
# Token counting + tiktoken fallback
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_short_text_is_positive(self):
        assert count_tokens("hello world") > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("one")
        long = count_tokens("one two three four five six seven eight nine ten")
        assert long > short

    def test_fallback_uses_char_over_four(self, monkeypatch):
        """When tiktoken is unavailable, count is len(text)//4 (min 1)."""
        # Force the fallback path by stubbing the encoding probe.
        chunker_mod._reset_tiktoken_cache_for_tests()
        monkeypatch.setattr(chunker_mod, "_get_tiktoken_encoding", lambda: None)
        # 40 chars / 4 = 10
        text = "a" * 40
        assert count_tokens(text) == 10
        # Single char still rounds to at least 1
        assert count_tokens("a") == 1

    def test_fallback_off_by_at_most_factor_two_on_english(self, monkeypatch):
        """Sanity-check that the heuristic is in the right order of magnitude.

        We do not assert ±20% here because the exact ratio depends on
        whether tiktoken is installed in the test environment. Instead
        we assert both counts are positive and within a factor-of-two
        envelope, which is the operative claim in the ADR.
        """
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump."
        )
        chunker_mod._reset_tiktoken_cache_for_tests()
        real = count_tokens(text)

        chunker_mod._reset_tiktoken_cache_for_tests()
        monkeypatch.setattr(chunker_mod, "_get_tiktoken_encoding", lambda: None)
        fallback = count_tokens(text)

        assert real > 0 and fallback > 0
        ratio = max(real, fallback) / min(real, fallback)
        assert ratio < 2.0, (
            f"tiktoken={real} fallback={fallback} ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Splitter primitives
# ---------------------------------------------------------------------------


class TestSplitParagraphs:
    def test_single_paragraph(self):
        assert _split_paragraphs("one paragraph here") == ["one paragraph here"]

    def test_blank_line_separator(self):
        assert _split_paragraphs("first\n\nsecond") == ["first", "second"]

    def test_multiple_blank_lines(self):
        assert _split_paragraphs("first\n\n\n\nsecond") == ["first", "second"]

    def test_strips_whitespace(self):
        assert _split_paragraphs("  first  \n\n  second  ") == ["first", "second"]

    def test_drops_empty(self):
        assert _split_paragraphs("\n\n\n\n") == []


class TestSplitSentences:
    def test_period_followed_by_capital(self):
        out = _split_sentences("First sentence. Second sentence.")
        assert out == ["First sentence.", "Second sentence."]

    def test_question_mark(self):
        out = _split_sentences("Are you here? I am here.")
        assert out == ["Are you here?", "I am here."]

    def test_exclamation(self):
        out = _split_sentences("Stop! Go now.")
        assert out == ["Stop!", "Go now."]

    def test_no_split_on_lowercase_following(self):
        # Conservative: lowercase after period (e.g. abbreviations)
        # is *not* a split point.
        out = _split_sentences("e.g. this is one sentence.")
        assert out == ["e.g. this is one sentence."]

    def test_handles_quoted_continuation(self):
        out = _split_sentences('She said "hi". "Bye" was the reply.')
        assert len(out) == 2


class TestHardSplitByTokens:
    def test_short_text_returns_single_unit(self):
        assert _hard_split_by_tokens("short text", 100) == ["short text"]

    def test_splits_long_text_at_word_boundary(self):
        text = " ".join(["word"] * 200)
        chunks = _hard_split_by_tokens(text, max_tokens=20)
        assert len(chunks) > 1
        # No chunk should slice mid-word
        for c in chunks:
            assert "word" in c
            assert not c.startswith(" ") and not c.endswith(" ")

    def test_empty_input(self):
        assert _hard_split_by_tokens("", 100) == []

    def test_single_word_over_cap_accepts_overshoot(self):
        # Extremely long single word — we'd rather over-shoot than slice
        # into UTF-8 codepoints.
        word = "a" * 1000
        out = _hard_split_by_tokens(word, max_tokens=10)
        assert out == [word]


class TestTakeOverlap:
    def test_zero_target_returns_empty(self):
        units, total = _take_overlap(["a", "b", "c"], 0)
        assert units == []
        assert total == 0

    def test_walks_from_end(self):
        units, total = _take_overlap(["alpha", "beta", "gamma"], 10)
        assert units[-1] == "gamma"
        assert total > 0

    def test_empty_units(self):
        assert _take_overlap([], 100) == ([], 0)


# ---------------------------------------------------------------------------
# Recursive units
# ---------------------------------------------------------------------------


class TestRecursiveUnits:
    def test_short_text_under_cap_passes_through(self):
        assert _recursive_units("hello world", 100) == ["hello world"]

    def test_paragraphs_split_first(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        units = _recursive_units(text, max_tokens=5)
        # Each paragraph is short enough on its own
        assert len(units) == 3

    def test_long_paragraph_splits_to_sentences(self):
        para = " ".join([f"Sentence number {i}." for i in range(20)])
        units = _recursive_units(para, max_tokens=10)
        # Sentence-level splitting kicks in
        assert len(units) > 1
        for u in units:
            # Each unit is roughly one or two sentences worth
            assert count_tokens(u) <= 30

    def test_huge_sentence_falls_back_to_word_split(self):
        sent = " ".join(["word"] * 500) + "."
        units = _recursive_units(sent, max_tokens=20)
        assert len(units) > 1


# ---------------------------------------------------------------------------
# Greedy assembler
# ---------------------------------------------------------------------------


class TestGreedyAssemble:
    def test_empty_units(self):
        assert _greedy_assemble(
            [], chunk_size_tokens=100, chunk_overlap_tokens=10,
        ) == []

    def test_single_unit_under_cap(self):
        chunks = _greedy_assemble(
            ["one paragraph"],
            chunk_size_tokens=100, chunk_overlap_tokens=10,
        )
        assert len(chunks) == 1
        assert chunks[0].content == "one paragraph"

    def test_packs_multiple_units(self):
        units = [f"para {i}" for i in range(5)]
        chunks = _greedy_assemble(
            units, chunk_size_tokens=100, chunk_overlap_tokens=0,
        )
        # All packed into one chunk
        assert len(chunks) == 1
        for unit in units:
            assert unit in chunks[0].content

    def test_emits_multiple_chunks_when_capped(self):
        # 50 paragraphs at ~3 tokens each = 150 tokens; cap at 30 → ~5 chunks
        units = [f"para number {i}" for i in range(50)]
        chunks = _greedy_assemble(
            units, chunk_size_tokens=30, chunk_overlap_tokens=0,
        )
        assert len(chunks) > 1

    def test_overlap_appears_in_consecutive_chunks(self):
        units = [f"unit-{i:03d} body content here" for i in range(20)]
        chunks = _greedy_assemble(
            units,
            chunk_size_tokens=30,
            chunk_overlap_tokens=10,
        )
        assert len(chunks) >= 2
        # End of chunk N should appear at the start of chunk N+1.
        for n in range(len(chunks) - 1):
            tail = chunks[n].content.split("\n\n")[-1]
            assert tail in chunks[n + 1].content

    def test_overlap_larger_than_cap_collapses_to_zero(self):
        # No deduplication / loop issues when overlap >= chunk_size.
        units = [f"unit {i}" for i in range(10)]
        chunks = _greedy_assemble(
            units, chunk_size_tokens=10, chunk_overlap_tokens=100,
        )
        assert len(chunks) >= 1
        # Total content length is bounded; no infinite-overlap blowup.
        total = sum(len(c.content) for c in chunks)
        assert total < 10_000

    def test_token_count_populated(self):
        chunks = _greedy_assemble(
            ["hello world"],
            chunk_size_tokens=100, chunk_overlap_tokens=0,
        )
        assert chunks[0].token_count > 0


# ---------------------------------------------------------------------------
# RecursiveTokenChunker
# ---------------------------------------------------------------------------


class TestRecursiveTokenChunker:
    @pytest.mark.asyncio
    async def test_empty_text_returns_no_chunks(self):
        chunker = RecursiveTokenChunker()
        assert await chunker.chunk("") == []
        assert await chunker.chunk("   \n\n  ") == []

    @pytest.mark.asyncio
    async def test_short_text_single_chunk(self):
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk("Hello world. This is a short doc.")
        assert len(out) == 1
        assert "Hello world" in out[0].content
        assert out[0].token_count > 0

    @pytest.mark.asyncio
    async def test_long_doc_produces_multiple_chunks(self):
        # Build ~3000 tokens of text via repeated paragraphs.
        para = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump.\n\n"
        )
        text = para * 60  # ~3K-4K tokens
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk(
            text,
            chunk_size_tokens=400,
            chunk_overlap_tokens=50,
        )
        assert len(out) >= 4
        # No chunk much exceeds the cap (allow some slack for boundary
        # snapping at paragraph breaks).
        for chunk in out:
            assert chunk.token_count < 600

    @pytest.mark.asyncio
    async def test_overlap_is_respected(self):
        para = " ".join(f"sentence-{i}." for i in range(200))
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk(
            para, chunk_size_tokens=80, chunk_overlap_tokens=20,
        )
        assert len(out) >= 2
        # Verify each adjacent pair shares a non-trivial substring at
        # the seam — proves overlap is happening.
        overlap_seen = 0
        for n in range(len(out) - 1):
            tail_tokens = out[n].content.split()[-3:]
            head = out[n + 1].content
            if any(tok in head for tok in tail_tokens):
                overlap_seen += 1
        assert overlap_seen >= 1

    @pytest.mark.asyncio
    async def test_chunks_are_non_empty(self):
        text = "para one.\n\npara two.\n\npara three.\n\npara four."
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk(text, chunk_size_tokens=100)
        for chunk in out:
            assert chunk.content.strip(), "chunk content should not be empty"

    @pytest.mark.asyncio
    async def test_invalid_chunk_size_raises(self):
        chunker = RecursiveTokenChunker()
        with pytest.raises(ValueError):
            await chunker.chunk("abc", chunk_size_tokens=0)
        with pytest.raises(ValueError):
            await chunker.chunk("abc", chunk_size_tokens=-10)

    @pytest.mark.asyncio
    async def test_invalid_overlap_raises(self):
        chunker = RecursiveTokenChunker()
        with pytest.raises(ValueError):
            await chunker.chunk("abc", chunk_overlap_tokens=-1)

    @pytest.mark.asyncio
    async def test_metadata_default_empty_dict(self):
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk("Hello world.")
        assert out[0].metadata == {}

    @pytest.mark.asyncio
    async def test_unicode_passes_through(self):
        text = "héllo wörld. 日本語のテキスト. مرحبا بالعالم."
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk(text)
        assert len(out) == 1
        assert "héllo" in out[0].content
        assert "日本語" in out[0].content
        assert "مرحبا" in out[0].content

    @pytest.mark.asyncio
    async def test_huge_single_paragraph_split_via_word_boundaries(self):
        # No paragraph or sentence breaks at all — just a wall of words.
        # Forces the hard-split fallback.
        text = " ".join(["word"] * 2000)
        chunker = RecursiveTokenChunker()
        out = await chunker.chunk(
            text,
            chunk_size_tokens=200,
            chunk_overlap_tokens=20,
        )
        assert len(out) > 1
        for chunk in out:
            assert "word" in chunk.content


# ---------------------------------------------------------------------------
# NullChunker + factory
# ---------------------------------------------------------------------------


class TestNullChunker:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self):
        chunker = NullChunker()
        out = await chunker.chunk("anything at all goes here")
        assert out == []


class TestCreateChunker:
    def test_disabled_returns_null(self):
        assert isinstance(create_chunker(enabled=False), NullChunker)

    def test_enabled_returns_recursive(self):
        assert isinstance(create_chunker(enabled=True), RecursiveTokenChunker)

    def test_default_is_enabled(self):
        assert isinstance(create_chunker(), RecursiveTokenChunker)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


class TestChunk:
    def test_default_metadata_is_empty_dict(self):
        c = Chunk(content="hi")
        assert c.metadata == {}
        assert c.token_count == 0

    def test_default_metadata_is_not_shared(self):
        # Regression guard against the classic mutable-default bug.
        a = Chunk(content="a")
        b = Chunk(content="b")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata
