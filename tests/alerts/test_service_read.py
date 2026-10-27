"""Reading alerts through the service: the list with its filters and pages (AL-4), the detail with
its members and trail, and the counters."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertFilter, Severity
from social_tracker.classification.domain import Label
from social_tracker.shared.errors import NotFoundError, ValidationError
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED

RUMOR, DELIBERATE, RECYCLED = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
)
FIRST = PUBLISHED + timedelta(minutes=1)


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready("src_gov")
    yield world
    world.close()


def fill(w, count=5):
    for n in range(count):
        w.put(
            f"n{n}",
            confidence=round(0.5 + n / 20, 2),
            fetched=FIRST + timedelta(hours=n),
            source="src_gov" if n % 2 else "src_news",
        )
    w.refresh()


def listed(w, flt=None, **page):
    return w.service.list(flt, **page)


def test_the_list_is_newest_first_and_says_how_many_there_are(w):
    fill(w)
    page = listed(w)
    assert [a.confidence for a in page.alerts] == [0.7, 0.65, 0.6, 0.55, 0.5]
    assert (page.total, page.limit, page.offset) == (5, 25, 0)
    assert (page.has_next, page.has_previous) == (False, False)


@pytest.mark.parametrize(
    ("limit", "offset", "expected", "has_next", "has_previous"),
    [
        (2, 0, [0.7, 0.65], True, False),
        (2, 2, [0.6, 0.55], True, True),
        (2, 4, [0.5], False, True),
        (5, 0, [0.7, 0.65, 0.6, 0.55, 0.5], False, False),
        (4, 0, [0.7, 0.65, 0.6, 0.55], True, False),
        (1, 4, [0.5], False, True),
        (2, 5, [], False, True),
        (2, 100, [], False, True),
    ],
)
def test_pages_and_their_neighbours(w, limit, offset, expected, has_next, has_previous):
    fill(w)
    page = listed(w, limit=limit, offset=offset)
    assert [a.confidence for a in page.alerts] == expected
    assert (page.has_next, page.has_previous) == (has_next, has_previous)
    assert page.total == 5


def test_an_empty_store_is_an_empty_page(w):
    page = listed(w)
    assert (page.alerts, page.total, page.has_next, page.has_previous) == ((), 0, False, False)
    counts = w.service.counts()
    assert (counts.active, counts.new, counts.withdrawn) == (0, 0, 0)


@pytest.mark.parametrize(
    ("limit", "offset"), [(0, 0), (-1, 0), (201, 0), (5, -1), (True, 0), ("5", 0)]
)
def test_a_page_out_of_range_is_refused(w, limit, offset):
    with pytest.raises(ValidationError):
        listed(w, limit=limit, offset=offset)


def test_the_filters_narrow_the_list(w):
    fill(w)
    assert len(listed(w, AlertFilter(source_id="src_gov")).alerts) == 2
    assert listed(w, AlertFilter(severity=Severity.HIGH)).total == 1
    assert listed(w, AlertFilter(severity=Severity.HIGH, source_id="src_gov")).total == 0
    flt = AlertFilter(since=FIRST + timedelta(hours=1), until=FIRST + timedelta(hours=3))
    assert [a.confidence for a in listed(w, flt).alerts] == [0.65, 0.6, 0.55]
    assert listed(w, AlertFilter(tag="earthquake")).total == 5
    assert listed(w, AlertFilter(tag="prediction")).total == 5
    assert listed(w, AlertFilter(tag="obfuscated_text")).total == 0


def test_a_filter_by_status_and_label(w):
    w.put("x", label=DELIBERATE, confidence=0.9)
    w.put("y", label=RECYCLED, confidence=0.9)
    w.refresh()
    alert = listed(w, AlertFilter(label="recycled_content")).alerts[0]
    w.service.set_status(alert.id, "seen", actor="ana", reason="looked")
    assert listed(w, AlertFilter(status="seen")).total == 1
    assert listed(w, AlertFilter(status="new")).total == 1
    assert listed(w, AlertFilter(status="seen", label="deliberate_disinformation")).total == 0


def test_the_list_does_not_show_withdrawn_alerts_unless_asked(w):
    w.put("x", confidence=0.9)
    w.refresh()
    with w.db.transaction() as conn:
        conn.execute("UPDATE classifications SET label = 'no_risk'")
    w.refresh()
    assert listed(w).total == 0
    assert listed(w, AlertFilter(status="withdrawn")).total == 1
    counts = w.service.counts()
    assert (counts.active, counts.withdrawn) == (0, 1)


# ---- the detail ----


def test_the_detail_has_the_members_the_trail_and_the_sources(w):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    w.put("a", text=text, confidence=0.9, label=DELIBERATE)
    w.put("b", text=text, confidence=0.6, source="src_gov")
    w.refresh()
    alert = listed(w).alerts[0]
    w.service.set_status(alert.id, "seen", actor="ana", reason="looked at it")
    detail = w.service.detail(alert.id)
    assert detail.alert.id == alert.id
    assert len(detail.members) == 2
    assert {m.source_id for m in detail.members} == {"src_news", "src_gov"}
    assert [e.kind.value for e in detail.events] == ["created", "merged", "status_changed"]
    assert detail.removed_sources == frozenset()
    assert detail.alert.status.value == "seen"


@pytest.mark.parametrize("bad", ["alt_" + "0" * 32, "", "x", "' OR 1=1 --", "A" * 100])
def test_the_detail_of_an_unknown_alert_is_not_found(w, bad):
    with pytest.raises(NotFoundError):
        w.service.detail(bad)


def test_the_counters_split_new_from_the_rest(w):
    fill(w)
    first = listed(w).alerts[0]
    w.service.set_status(first.id, "seen", actor="ana", reason="looked")
    counts = w.service.counts()
    assert (counts.active, counts.new) == (5, 4)
    assert dict(counts.by_status) == {"new": 4, "seen": 1}
    assert sum(n for _, n in counts.new_by_severity) == 4
