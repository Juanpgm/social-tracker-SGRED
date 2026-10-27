"""Limits and refusals of the feed parser: size, depth, a hostile prefix, namespaces."""

import pytest

from social_tracker.collector.feed import MAX_DEPTH, MAX_FEED_BYTES, FeedError, parse_feed

EMPTY_RSS = b'<rss version="2.0"><channel></channel></rss>'


def nested_title(levels: int) -> bytes:
    """A feed whose title holds `levels` nested elements: the whole tree is `levels + 4` deep."""
    inner = b"<a>" * levels + b"</a>" * levels
    return b"<rss><channel><item><title>" + inner + b"</title></item></channel></rss>"


def test_a_document_exactly_at_the_size_limit_is_read_and_one_byte_more_is_refused():
    exact = EMPTY_RSS + b" " * (MAX_FEED_BYTES - len(EMPTY_RSS))
    assert len(exact) == MAX_FEED_BYTES
    assert parse_feed(exact).entries == ()
    with pytest.raises(FeedError) as caught:
        parse_feed(exact + b" ")
    assert "too large" in caught.value.message


def test_nesting_exactly_at_the_depth_limit_is_read_and_one_level_more_is_refused():
    assert len(parse_feed(nested_title(MAX_DEPTH - 4)).entries) == 1
    with pytest.raises(FeedError) as caught:
        parse_feed(nested_title(MAX_DEPTH - 3))
    assert "nested" in caught.value.message


@pytest.mark.parametrize("empty", [b"", b"   ", b"\n\t \r\n"])
def test_an_empty_document_is_reported_as_empty(empty):
    with pytest.raises(FeedError) as caught:
        parse_feed(empty)
    assert "empty" in caught.value.message


def test_a_document_type_declaration_is_refused_even_when_bad_bytes_come_first():
    # The strict pass fails inside the comment, so the document reaches the lenient pass; the
    # declaration after it must still be refused there.
    prefix = b'<?xml version="1.0"?><!-- \xff\xfe -->'
    doctype = b"<!DOCTYPE rss [<!ELEMENT rss ANY>]>"
    body = b'<rss version="2.0"><channel><item><title>x</title></item></channel></rss>'
    with pytest.raises(FeedError) as caught:
        parse_feed(prefix + doctype + body)
    assert "document type" in caught.value.message
    assert len(parse_feed(prefix + body).entries) == 1  # the same document without it is fine


def test_an_atom_author_is_the_name_not_the_whole_author_block():
    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>t</title><id>i</id>'
        "<author><name>Carla</name><email>carla@example.org</email>"
        "<uri>https://example.org/carla</uri></author></entry></feed>"
    )
    assert parse_feed(xml.encode()).entries[0].author == "Carla"


def test_a_feed_element_outside_the_atom_namespace_is_not_an_atom_feed():
    with pytest.raises(FeedError) as caught:
        parse_feed(b"<feed><entry><title>t</title></entry></feed>")
    assert "RSS or Atom" in caught.value.message
