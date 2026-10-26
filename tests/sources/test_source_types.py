"""Per-type configuration rules."""

import time

import pytest

from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import parse_source
from tests.sources.factories import EMBEDDED_CREDENTIALS_URL, payload, without

# ---- news_rss -------------------------------------------------------------------------


def test_news_rss_defaults():
    src = parse_source(payload("news_rss"))
    assert src.poll_interval_s == 600
    assert src.keywords is None


@pytest.mark.parametrize("value", [120, 121, 86_400])
def test_news_rss_poll_interval_accepts_min_and_above(value: int):
    assert parse_source(payload("news_rss", poll_interval_s=value)).poll_interval_s == value


@pytest.mark.parametrize("value", [119, 0, -1, -600, None, "600", 600.0, True])
def test_news_rss_poll_interval_rejects(value):
    with pytest.raises(ValidationError, match="poll_interval_s"):
        parse_source(payload("news_rss", poll_interval_s=value))


def test_news_rss_requires_feed_url():
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(without(payload("news_rss"), "feed_url"))


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.org/feed",
        "rtmp://example.org/live",
        "example.org/feed",
        "https://",
        "http:///path",
        "https://:8080/x",
        EMBEDDED_CREDENTIALS_URL,
        "https://exa mple.org/feed",
        "https://example.org/fe\ned",
        "https://example.org/" + "a" * 2100,
        "file://localhost/etc/passwd",
        "javascript://example.org/%0aalert(1)",
        "https://[]:8080/x",
        "https://[",
        "https://example.org/a\x00b",
        "https://example.org/a\x07b",
        "https://example.org/a\x7fb",
        None,
        42,
    ],
)
def test_news_rss_rejects_bad_urls(url):
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url=url))


@pytest.mark.parametrize(
    "url",
    ["http://example.org/rss", "HTTPS://Example.org:8443/a?b=c#d", "https://[::1]:8080/feed"],
)
def test_news_rss_accepts_http_and_https(url: str):
    assert parse_source(payload("news_rss", feed_url=url)).feed_url == url


def test_url_length_boundary():
    prefix = "https://example.org/"
    at_limit = prefix + "a" * (2048 - len(prefix))
    assert parse_source(payload("news_rss", feed_url=at_limit)).feed_url == at_limit
    with pytest.raises(ValidationError, match="longer than"):
        parse_source(payload("news_rss", feed_url=at_limit + "a"))


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_url_is_reported_as_blank(blank: str):
    with pytest.raises(ValidationError, match="blank"):
        parse_source(payload("news_rss", feed_url=blank))


def test_url_check_on_one_megabyte_input_is_fast():
    start = time.perf_counter()
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url="https://" + "a" * 1_000_000))
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url=" " * 1_000_000))
    assert time.perf_counter() - start < 2.0


def test_keywords_are_trimmed_and_optional():
    src = parse_source(payload("news_rss", keywords=[" fire ", "flood"]))
    assert src.keywords == ["fire", "flood"]


@pytest.mark.parametrize("keywords", [[""], ["  "], [1], "fire", ["a" * 101], ["k"] * 51, {"a": 1}])
def test_keywords_reject_bad_values(keywords):
    with pytest.raises(ValidationError, match="keywords"):
        parse_source(payload("news_rss", keywords=keywords))


def test_keywords_accept_bounds():
    assert parse_source(payload("news_rss", keywords=["k"] * 50)).keywords == ["k"] * 50
    assert parse_source(payload("news_rss", keywords=["a" * 100])).keywords == ["a" * 100]
    assert parse_source(payload("news_rss", keywords=[])).keywords == []


# ---- instagram ------------------------------------------------------------------------


def test_instagram_defaults():
    src = parse_source(payload("instagram"))
    assert (src.include_stories, src.include_posts, src.poll_interval_s) == (True, True, 1800)
    assert src.secret_ref == "ig_main"


@pytest.mark.parametrize("handle", ["a", "a" * 30, "A.b_c9", "0", "._"])
def test_instagram_handle_accepts(handle: str):
    assert parse_source(payload("instagram", handle=handle)).handle == handle


@pytest.mark.parametrize(
    "handle",
    ["", "a" * 31, "a b", "a-b", "a@b", "@a", "café", "a\n", "\na", None, 5, "a/b"],
)
def test_instagram_handle_rejects(handle):
    with pytest.raises(ValidationError, match="handle"):
        parse_source(payload("instagram", handle=handle))


def test_instagram_requires_handle():
    with pytest.raises(ValidationError, match="handle"):
        parse_source(without(payload("instagram"), "handle"))


@pytest.mark.parametrize("value", [900, 901, 100_000])
def test_instagram_poll_interval_accepts(value: int):
    assert parse_source(payload("instagram", poll_interval_s=value)).poll_interval_s == value


@pytest.mark.parametrize("value", [899, 0, -900, None, "900"])
def test_instagram_poll_interval_rejects(value):
    with pytest.raises(ValidationError, match="poll_interval_s"):
        parse_source(payload("instagram", poll_interval_s=value))


def test_instagram_requires_secret_ref():
    with pytest.raises(ValidationError, match="secret_ref"):
        parse_source(without(payload("instagram"), "secret_ref"))


@pytest.mark.parametrize("ref", [None, "", "   ", "has space", "a" * 129, 5, "tab\tref", "x\n"])
def test_instagram_secret_ref_rejects(ref):
    with pytest.raises(ValidationError, match="secret_ref"):
        parse_source(payload("instagram", secret_ref=ref))


def test_secret_ref_accepts_bounds():
    assert parse_source(payload("instagram", secret_ref="a")).secret_ref == "a"
    assert parse_source(payload("instagram", secret_ref="a" * 128)).secret_ref == "a" * 128


@pytest.mark.parametrize("flag", ["include_stories", "include_posts"])
def test_instagram_flags_can_be_disabled_and_must_be_bool(flag: str):
    assert getattr(parse_source(payload("instagram", **{flag: False})), flag) is False
    with pytest.raises(ValidationError, match=flag):
        parse_source(payload("instagram", **{flag: "yes"}))


@pytest.mark.parametrize(
    "key", ["include_lives", "lives", "live", "includeLives", "INCLUDE_LIVES", "live_capture"]
)
@pytest.mark.parametrize("value", [True, False])
def test_instagram_lives_are_rejected_with_a_clear_message(key: str, value: bool):
    with pytest.raises(ValidationError, match="Instagram Lives"):
        parse_source(payload("instagram", **{key: value}))


def test_secret_ref_is_optional_for_other_types():
    assert parse_source(payload("news_rss")).secret_ref is None
    assert parse_source(payload("news_rss", secret_ref="shared_ref")).secret_ref == "shared_ref"


# ---- live_stream ----------------------------------------------------------------------


def test_live_stream_defaults():
    src = parse_source(payload("live_stream"))
    assert (src.chunk_seconds, src.language, src.mode) == (30, "es", "streaming")


@pytest.mark.parametrize("value", [10, 11, 59, 60])
def test_live_stream_chunk_seconds_accepts(value: int):
    assert parse_source(payload("live_stream", chunk_seconds=value)).chunk_seconds == value


@pytest.mark.parametrize("value", [9, 61, 0, -10, None, "30", 30.0, True])
def test_live_stream_chunk_seconds_rejects(value):
    with pytest.raises(ValidationError, match="chunk_seconds"):
        parse_source(payload("live_stream", chunk_seconds=value))


@pytest.mark.parametrize(
    "url",
    ["http://e.org/s", "https://e.org/s", "rtmp://e.org/live/key1", "icecast://e.org:8000/mount"],
)
def test_live_stream_accepts_streaming_schemes(url: str):
    assert parse_source(payload("live_stream", stream_url=url)).stream_url == url


@pytest.mark.parametrize(
    "url", ["", "  ", "file:///x", "javascript:x", "ftp://e.org/s", "rtsp://e.org/s", "e.org", None]
)
def test_live_stream_rejects_bad_urls(url):
    with pytest.raises(ValidationError, match="stream_url"):
        parse_source(payload("live_stream", stream_url=url))


def test_live_stream_requires_stream_url():
    with pytest.raises(ValidationError, match="stream_url"):
        parse_source(without(payload("live_stream"), "stream_url"))


@pytest.mark.parametrize("lang", ["", "  ", "x" * 17, None, 5])
def test_live_stream_language_rejects(lang):
    with pytest.raises(ValidationError, match="language"):
        parse_source(payload("live_stream", language=lang))


def test_live_stream_language_override():
    assert parse_source(payload("live_stream", language="en")).language == "en"


# ---- recorded_audio -------------------------------------------------------------------


def test_recorded_audio_defaults_with_feed_url():
    src = parse_source(payload("recorded_audio"))
    assert src.poll_interval_s == 1800
    assert src.watch_dir is None


def test_recorded_audio_accepts_watch_dir_alone():
    src = parse_source(without(payload("recorded_audio", watch_dir="/data/audio"), "feed_url"))
    assert src.watch_dir == "/data/audio"
    assert src.feed_url is None


def test_recorded_audio_rejects_both():
    with pytest.raises(ValidationError, match="exactly one"):
        parse_source(payload("recorded_audio", watch_dir="/data/audio"))


def test_recorded_audio_rejects_neither():
    with pytest.raises(ValidationError, match="exactly one"):
        parse_source(without(payload("recorded_audio"), "feed_url"))


def test_recorded_audio_explicit_nulls_count_as_neither():
    with pytest.raises(ValidationError, match="exactly one"):
        parse_source(payload("recorded_audio", feed_url=None, watch_dir=None))


@pytest.mark.parametrize("blank", ["", "  "])
def test_recorded_audio_blank_watch_dir_rejected(blank: str):
    with pytest.raises(ValidationError, match="watch_dir"):
        parse_source(without(payload("recorded_audio", watch_dir=blank), "feed_url"))


@pytest.mark.parametrize("blank", ["", "  "])
def test_recorded_audio_blank_feed_url_rejected(blank: str):
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("recorded_audio", feed_url=blank))


@pytest.mark.parametrize("bad", ["/data/\x00x", "/data/a\nb", "/data/\x1fx", "/data/\x7fx", 5])
def test_recorded_audio_watch_dir_rejects_control_chars_and_non_str(bad):
    with pytest.raises(ValidationError, match="watch_dir"):
        parse_source(without(payload("recorded_audio", watch_dir=bad), "feed_url"))


def test_watch_dir_length_boundary_and_space_allowed():
    ok = "/" + "d" * 4095
    got = parse_source(without(payload("recorded_audio", watch_dir=ok), "feed_url"))
    assert got.watch_dir == ok
    spaced = "/data/my audio"
    got = parse_source(without(payload("recorded_audio", watch_dir=spaced), "feed_url"))
    assert got.watch_dir == spaced
    with pytest.raises(ValidationError, match="longer than"):
        parse_source(without(payload("recorded_audio", watch_dir=ok + "d"), "feed_url"))


def test_recorded_audio_feed_url_scheme_checked():
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("recorded_audio", feed_url="file:///x.mp3"))


@pytest.mark.parametrize("value", [1, 1800])
def test_recorded_audio_poll_interval_accepts(value: int):
    assert parse_source(payload("recorded_audio", poll_interval_s=value)).poll_interval_s == value


@pytest.mark.parametrize("value", [0, -1, None])
def test_recorded_audio_poll_interval_rejects(value):
    with pytest.raises(ValidationError, match="poll_interval_s"):
        parse_source(payload("recorded_audio", poll_interval_s=value))
