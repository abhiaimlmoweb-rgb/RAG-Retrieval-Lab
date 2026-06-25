"""Tests for web document naming and loaders."""

from loaders.web_naming import url_document_slug


def test_url_document_slug_unique_for_different_urls():
    a = url_document_slug("https://example.com/docs/very/long/path/segment/alpha")
    b = url_document_slug("https://example.com/docs/very/long/path/segment/beta")
    assert a != b
    assert a.endswith(".html")
    assert b.endswith(".html")


def test_url_document_slug_stable_for_same_url():
    url = "https://docs.example.com/guide/install.html"
    assert url_document_slug(url) == url_document_slug(url)


def test_truncated_paths_still_unique():
    long_a = "https://host.com/" + "a" * 200
    long_b = "https://host.com/" + "b" * 200
    assert url_document_slug(long_a) != url_document_slug(long_b)
