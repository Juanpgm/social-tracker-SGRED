"""Boundaries of the HTML reader and of the item identity rules."""

import pytest

from social_tracker.collector.normalize import MAX_HTML_CHARS, strip_html
from tests.collector.test_normalize import only_item


@pytest.mark.parametrize(
    ("html", "text"),
    [
        ("<iframe>hidden<script>x</script>still hidden</iframe>visible", "visible"),
        ("<template>hidden<style>x</style>still hidden</template>visible", "visible"),
        ("<noscript>a<script>x</script>b<style>y</style>c</noscript>visible", "visible"),
        ("<script>a<script>b</script>c</script>shown", "cshown"),  # as a browser reads it
        ("before</script>after", "beforeafter"),
        ("one</style></style>two", "onetwo"),
        ("intro<p>one</p>", "intro\n\none"),
        ("AT&T", "AT&T"),
        ("x &amp", "x &"),
        ("tail <b", "tail"),
    ],
)
def test_strip_html_handles_nesting_stray_end_tags_and_unfinished_input(html, text):
    assert strip_html(html) == text


def test_the_html_reader_looks_at_a_bounded_amount_of_input():
    assert len(strip_html("a" * (MAX_HTML_CHARS + 500))) == MAX_HTML_CHARS


def test_a_guid_of_exactly_the_id_limit_is_kept_and_a_longer_one_is_hashed():
    assert only_item(guid="g" * 512).external_id == "g" * 512
    assert only_item(guid="g" * 513).external_id.startswith("h:")


def test_a_synthesized_id_is_a_forty_digit_hash():
    external_id = only_item(guid=None).external_id
    assert external_id.startswith("h:")
    assert len(external_id) == 42
