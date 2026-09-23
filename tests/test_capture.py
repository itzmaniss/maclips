"""S1 capture: link classification, leaderboard stripping, output shape.

No network and no browser: the Playwright-driving function isn't touched
here, only the pure helpers it's built from.
"""
from __future__ import annotations

from maclips import capture as c


def test_slugify():
    assert c.slugify("Reach — $5,000 Weekly Double Coverage!") == "reach-5-000-weekly-double-coverage"
    assert c.slugify("   ") == "campaign"


def test_google_doc_and_notion_links_are_docs():
    assert c.classify_link("https://docs.google.com/document/d/abc123/edit?usp=sharing") == "doc"
    assert c.classify_link("https://mediamaxxing.notion.site/lovable-clipping") == "doc"


def test_non_allowlisted_links_are_undecided():
    """Only Google Docs and Notion are fetched. Everything else — source
    footage, socials, and any other domain (even a plausible requirements
    microsite like reachclipping.com) — is left for a human to decide on."""
    for url in [
        "https://www.youtube.com/@someone",
        "https://www.tiktok.com/@someone/video/123",
        "https://kick.com/someone",
        "https://www.twitch.tv/someone",
        "https://www.dropbox.com/scl/fo/abc",
        "https://drive.google.com/drive/folders/abc",
        "https://discord.gg/abc",
        "https://example.com/watermark.png",
        "https://reachclipping.com/DedicatedDoubleCoverage",
    ]:
        assert c.classify_link(url) == "undecided", url


def test_google_doc_export_url():
    url = "https://docs.google.com/document/d/1bNXCOGVCFcVedkgcsziFRTpqIGG3Z3z8FFsteSS0qp4/edit?usp=sharing"
    assert c.google_doc_export_url(url) == (
        "https://docs.google.com/document/d/1bNXCOGVCFcVedkgcsziFRTpqIGG3Z3z8FFsteSS0qp4/export?format=txt"
    )
    assert c.google_doc_export_url("https://example.com") is None


def test_leaderboard_block_is_stripped_between_its_markers():
    text = (
        "Content requirements\n"
        "Include brand logo\n"
        "Top clippers\n"
        "Zia $1.4k\n"
        "Alive $633.23\n"
        "10M views across every approved clip, running total by day.\n"
        "Aug 24\n"
    )
    stripped = c.strip_leaderboard(text)
    assert "Zia" not in stripped
    assert "Alive" not in stripped
    assert "Content requirements" in stripped
    assert "views across every approved clip" in stripped


def test_leaderboard_missing_end_marker_gives_up_after_the_cap():
    lines = ["Top clippers"] + [f"creator{i} $1.00" for i in range(100)]
    stripped = c.strip_leaderboard("\n".join(lines))
    # No end marker ever appears, so only the capped window is treated as the
    # leaderboard; everything past it is kept rather than silently eaten.
    assert "creator39" not in stripped
    assert "creator40" in stripped
    assert "creator99" in stripped


def test_text_without_a_leaderboard_is_unchanged():
    text = "Budget\n$0 / $5,000 remaining\nContent requirements\nInclude brand logo\n"
    assert c.strip_leaderboard(text) == text


def test_build_capture_text_wraps_fetched_docs_as_untrusted_and_lists_undecided():
    doc = c.FetchedDoc(
        url="https://docs.google.com/document/d/x/edit",
        text="full rules here",
        fetched_at="2026-09-23T12:00:00+00:00",
    )
    text = c.build_capture_text(
        "https://contentrewards.com/discover/x",
        "Campaign Name\nDescription here.",
        fetched=[doc],
        undecided=["https://www.youtube.com/@brand"],
    )
    assert "Source URL: https://contentrewards.com/discover/x" in text
    assert "Campaign Name" in text
    assert "https://www.youtube.com/@brand" in text
    assert "NOT fetched" in text

    # The fetched doc must be inside a clearly marked untrusted block, with
    # its own source URL and fetch time, and the on-page text must sit
    # entirely outside that block.
    start = text.index("=== UNTRUSTED THIRD-PARTY CONTENT ===")
    end = text.index("=== END UNTRUSTED THIRD-PARTY CONTENT ===")
    block = text[start:end]
    assert "Source URL: https://docs.google.com/document/d/x/edit" in block
    assert "Fetched at: 2026-09-23T12:00:00+00:00" in block
    assert "untrusted" in block.lower()
    assert "instructions" in block.lower()
    assert "full rules here" in block
    assert "Campaign Name" not in block
    assert text.index("Campaign Name") < start
