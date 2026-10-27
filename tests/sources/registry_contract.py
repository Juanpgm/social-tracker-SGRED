"""Behavioural contract of `SourceRegistry`, run against in-memory and SQLite repositories.

Every test takes an `Env` (the registry plus the repositories behind it) so the same scenarios
can also be replayed against deliberately wrong registries (see the meta-tests).
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import pytest

from social_tracker.control.domain import (
    CollectionState,
    SourceState,
    source_analysis_key,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import (
    ConflictError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.domain import SOURCE_TYPES, Source, parse_source
from social_tracker.sources.ports import SourceRepository
from social_tracker.sources.service import (
    SOURCE_ADDED,
    SOURCE_DELETED,
    SOURCE_DISABLED,
    SOURCE_EDITED,
    SOURCE_ENABLED,
    SOURCES_IMPORTED,
    ImportConflictError,
    ImportReport,
    SourceChange,
    SourceRegistry,
)
from social_tracker.sources.yaml_io import SourceFileError, dump_sources
from tests.sources.factories import CANARY, T0, payload

START = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)


class FlakyControl:
    """Delegates to a real control repository but fails the n-th call of one method."""

    def __init__(self, inner: ControlRepository, method: str, on_call: int = 1) -> None:
        self._inner = inner
        self._method = method
        self._on_call = on_call
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._method:
            return attr

        def flaky(*args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == self._on_call:
                raise StorageError("simulated storage failure")
            return attr(*args, **kwargs)

        return flaky


@dataclass
class Env:
    registry: SourceRegistry
    sources: SourceRepository
    control: ControlRepository
    uow: UnitOfWork
    clock: FixedClock
    registry_cls: type[SourceRegistry]

    def registry_with(
        self, *, control: Any = None, clock: Any = None, sources: Any = None
    ) -> SourceRegistry:
        return self.registry_cls(
            sources or self.sources, control or self.control, self.uow, clock or self.clock
        )


def data(source_id: str = "src_one", name: str = "One", type_: str = "news_rss", **kw: Any):
    full = payload(type_, id=source_id, name=name, **kw)
    full.pop("created_at")
    full.pop("updated_at")
    return full


def events(env: Env, source_id: str | None = None) -> list[tuple[str, str | None, dict[str, Any]]]:
    return [
        (e.kind, e.source_id, json.loads(e.detail))
        for e in env.control.list_control_events(source_id)
    ]


def kinds(env: Env) -> list[str]:
    return [kind for kind, _, _ in events(env)]


def export_of(*sources: Source) -> str:
    return dump_sources(sources)


def race(calls: list[Callable[[], Any]]) -> list[tuple[str, Any]]:
    barrier = threading.Barrier(len(calls))
    results: list[tuple[str, Any]] = [("unset", None)] * len(calls)

    def run(i: int) -> None:
        barrier.wait(10)
        try:
            results[i] = ("ok", calls[i]())
        except Exception as exc:  # noqa: BLE001 - the outcome is what the test inspects
            results[i] = ("err", exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results


class RegistryContract:
    def make_env(self, registry_cls: type[SourceRegistry] = SourceRegistry) -> Env:
        raise NotImplementedError

    @pytest.fixture
    def env(self) -> Env:
        return self.make_env()

    # ---- start state ----

    def test_a_fresh_registry_has_no_sources_and_no_events(self, env: Env):
        assert env.registry.list() == []
        assert env.control.list_control_events() == []
        assert env.registry.export_sources() == "sources: []\n"

    # ---- add ----

    @pytest.mark.parametrize("type_", SOURCE_TYPES)
    def test_add_stores_each_source_type_with_clock_timestamps_and_disabled(
        self, env: Env, type_: str
    ):
        source = env.registry.add(data(type_=type_))
        assert source.type == type_
        assert (source.created_at, source.updated_at) == (START, START)
        assert source.enabled is False
        assert env.registry.get("src_one") == source
        assert env.sources.get("src_one") == source

    def test_add_writes_exactly_one_audit_event(self, env: Env):
        env.registry.add(data())
        stored = env.control.list_control_events()
        assert [(e.kind, e.source_id, e.at) for e in stored] == [(SOURCE_ADDED, "src_one", START)]
        assert json.loads(stored[0].detail) == {"type": "news_rss"}

    def test_add_ignores_timestamps_supplied_by_the_caller(self, env: Env):
        source = env.registry.add({**data(), "created_at": "2001-01-01T00:00:00+00:00"})
        assert source.created_at == START
        assert source.updated_at == START

    @pytest.mark.parametrize("bad", [None, [], "text", 5, [("id", "src_one")]])
    def test_add_rejects_data_that_is_not_a_mapping(self, env: Env, bad: Any):
        with pytest.raises(ValidationError):
            env.registry.add(bad)
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    def test_add_validation_failure_stores_nothing_and_names_the_field(self, env: Env):
        with pytest.raises(ValidationError) as caught:
            env.registry.add(data(poll_interval_s=1))
        assert caught.value.fields == ("poll_interval_s",)
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    def test_add_rejects_inline_secrets_without_echoing_them(self, env: Env):
        with pytest.raises(ValidationError) as caught:
            env.registry.add({**data(), "token": CANARY})
        assert CANARY not in caught.value.message
        assert "token" not in caught.value.message
        assert env.registry.list() == []

    def test_add_with_a_duplicate_id_conflicts_without_echoing_the_name(self, env: Env):
        env.registry.add(data("src_one", "Unique Name"))
        before = env.control.list_control_events()
        with pytest.raises(ConflictError) as caught:
            env.registry.add(data("src_one", "Other Name"))
        assert "Other Name" not in caught.value.message
        assert [s.name for s in env.registry.list()] == ["Unique Name"]
        assert env.control.list_control_events() == before

    @pytest.mark.parametrize("clash", ["same name", "SAME NAME", "  Same Name  "])
    def test_add_name_uniqueness_is_case_and_edge_space_insensitive(self, env: Env, clash: str):
        env.registry.add(data("src_one", "Same Name"))
        with pytest.raises(ConflictError):
            env.registry.add(data("src_two", clash))
        assert [s.id for s in env.registry.list()] == ["src_one"]
        assert kinds(env) == [SOURCE_ADDED]

    def test_conflict_messages_do_not_echo_names(self, env: Env):
        env.registry.add(data("src_one", f"Name {CANARY}"))
        with pytest.raises(ConflictError) as caught:
            env.registry.add(data("src_two", f"NAME {CANARY}"))
        assert CANARY not in caught.value.message

    def test_a_no_break_space_inside_a_name_makes_it_a_different_name(self, env: Env):
        env.registry.add(data("src_one", "Same Name"))
        env.registry.add(data("src_two", "Same Name"))
        assert len(env.registry.list()) == 2

    def test_add_name_uniqueness_applies_unicode_normalisation(self, env: Env):
        env.registry.add(data("src_one", "Café"))  # precomposed e-acute
        with pytest.raises(ConflictError):
            env.registry.add(data("src_two", "Café"))  # e + combining acute

    def test_a_failing_audit_event_rolls_back_the_new_source(self, env: Env):
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).add(data())
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    # ---- get / list ----

    def test_get_unknown_source_is_not_found(self, env: Env):
        with pytest.raises(NotFoundError):
            env.registry.get("src_missing")

    @pytest.mark.parametrize("bad", ["", "ab", "Has Space", "UPPER_ID", None, 5, "x" * 65])
    def test_get_rejects_ids_that_are_not_slugs(self, env: Env, bad: Any):
        with pytest.raises(ValidationError) as caught:
            env.registry.get(bad)
        assert caught.value.fields == ("id",)

    def test_list_is_sorted_by_id(self, env: Env):
        for sid in ("src_c", "src_a", "src_b"):
            env.registry.add(data(sid, sid.upper()))
        assert [s.id for s in env.registry.list()] == ["src_a", "src_b", "src_c"]

    # ---- edit ----

    def test_edit_changes_fields_stamps_updated_at_and_audits_the_changed_names(self, env: Env):
        added = env.registry.add(data())
        env.clock.advance(HOUR)
        result = env.registry.edit("src_one", {"name": "Renamed", "credibility": 5})
        assert isinstance(result, SourceChange)
        assert result.changed is True
        assert (result.source.name, result.source.credibility) == ("Renamed", 5)
        assert result.source.created_at == added.created_at == START
        assert result.source.updated_at == START + HOUR
        assert env.sources.get("src_one") == result.source
        assert events(env)[-1] == (SOURCE_EDITED, "src_one", {"changed": ["credibility", "name"]})
        assert len(events(env)) == 2

    def test_edit_with_identical_values_is_a_noop_that_leaves_updated_at_and_events_alone(
        self, env: Env
    ):
        added = env.registry.add(data())
        env.clock.advance(HOUR)
        result = env.registry.edit("src_one", {"name": "One", "credibility": 3})
        assert result.changed is False
        assert result.source == added
        assert env.sources.get("src_one").updated_at == START
        assert kinds(env) == [SOURCE_ADDED]

    def test_edit_with_no_changes_is_a_noop(self, env: Env):
        env.registry.add(data())
        env.clock.advance(HOUR)
        assert env.registry.edit("src_one", {}).changed is False
        assert kinds(env) == [SOURCE_ADDED]

    def test_edit_can_change_type_specific_fields_and_clear_optional_ones(self, env: Env):
        env.registry.add(data(keywords=["a", "b"]))
        env.clock.advance(HOUR)
        env.registry.edit("src_one", {"keywords": None, "poll_interval_s": 900})
        stored = env.sources.get("src_one")
        assert stored.keywords is None
        assert stored.poll_interval_s == 900

    def test_edit_rename_to_another_sources_name_conflicts_and_changes_nothing(self, env: Env):
        env.registry.add(data("src_one", f"One {CANARY}"))
        env.registry.add(data("src_two", "Two"))
        before = (env.registry.list(), env.control.list_control_events())
        with pytest.raises(ConflictError) as caught:
            env.registry.edit("src_two", {"name": f"  one {CANARY} "})
        assert CANARY not in caught.value.message
        assert (env.registry.list(), env.control.list_control_events()) == before

    def test_edit_case_only_rename_of_the_same_source_is_allowed(self, env: Env):
        env.registry.add(data("src_one", "one"))
        env.clock.advance(HOUR)
        result = env.registry.edit("src_one", {"name": "ONE"})
        assert result.changed is True
        assert env.sources.get("src_one").name == "ONE"
        assert env.sources.get("src_one").updated_at == START + HOUR

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("id", "src_new"),
            ("type", "instagram"),
            ("enabled", True),
            ("created_at", "2001-01-01T00:00:00+00:00"),
            ("updated_at", "2001-01-01T00:00:00+00:00"),
        ],
    )
    def test_edit_refuses_immutable_and_managed_fields(self, env: Env, key: str, value: Any):
        env.registry.add(data())
        before = env.registry.get("src_one")
        with pytest.raises(ValidationError) as caught:
            env.registry.edit("src_one", {key: value})
        assert caught.value.fields == (key,)
        assert env.registry.get("src_one") == before
        assert kinds(env) == [SOURCE_ADDED]

    def test_edit_rejects_unknown_keys_without_echoing_them(self, env: Env):
        env.registry.add(data())
        with pytest.raises(ValidationError) as caught:
            env.registry.edit("src_one", {f"x-{CANARY}": CANARY})
        assert CANARY not in caught.value.message
        assert caught.value.fields == ()

    @pytest.mark.parametrize("bad", [None, [], "name", 5])
    def test_edit_rejects_changes_that_are_not_a_mapping(self, env: Env, bad: Any):
        env.registry.add(data())
        with pytest.raises(ValidationError):
            env.registry.edit("src_one", bad)

    def test_edit_rejects_non_string_keys(self, env: Env):
        env.registry.add(data())
        with pytest.raises(ValidationError):
            env.registry.edit("src_one", {1: "x"})

    def test_edit_invalid_value_names_the_field_and_changes_nothing(self, env: Env):
        env.registry.add(data())
        before = env.registry.get("src_one")
        with pytest.raises(ValidationError) as caught:
            env.registry.edit("src_one", {"feed_url": "ftp://example.org/x"})
        assert caught.value.fields == ("feed_url",)
        assert env.registry.get("src_one") == before

    def test_edit_cannot_introduce_a_secret_in_a_url(self, env: Env):
        env.registry.add(data())
        with pytest.raises(ValidationError) as caught:
            env.registry.edit("src_one", {"feed_url": f"https://example.org/?token={CANARY}"})
        assert CANARY not in caught.value.message

    def test_edit_unknown_source_is_not_found(self, env: Env):
        with pytest.raises(NotFoundError):
            env.registry.edit("src_missing", {"name": "X"})

    def test_edit_with_a_clock_before_created_at_fails_and_writes_nothing(self, env: Env):
        env.registry.add(data())
        earlier = FixedClock(START - HOUR)
        with pytest.raises(ValidationError):
            env.registry_with(clock=earlier).edit("src_one", {"name": "Late"})
        assert env.registry.get("src_one").name == "One"
        assert kinds(env) == [SOURCE_ADDED]

    def test_a_failing_audit_event_rolls_back_an_edit(self, env: Env):
        env.registry.add(data())
        env.clock.advance(HOUR)
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).edit("src_one", {"name": "Renamed"})
        stored = env.sources.get("src_one")
        assert (stored.name, stored.updated_at) == ("One", START)
        assert kinds(env) == [SOURCE_ADDED]

    # ---- enable / disable ----

    def test_enable_then_disable_change_the_flag_stamp_updated_at_and_audit(self, env: Env):
        env.registry.add(data())
        env.clock.advance(HOUR)
        on = env.registry.enable("src_one")
        assert (on.changed, on.source.enabled, on.source.updated_at) == (True, True, START + HOUR)
        env.clock.advance(HOUR)
        off = env.registry.disable("src_one")
        assert (off.changed, off.source.enabled) == (True, False)
        assert off.source.updated_at == START + 2 * HOUR
        assert [(k, s) for k, s, _ in events(env)] == [
            (SOURCE_ADDED, "src_one"),
            (SOURCE_ENABLED, "src_one"),
            (SOURCE_DISABLED, "src_one"),
        ]

    def test_redundant_enable_and_disable_are_noops_without_events(self, env: Env):
        env.registry.add(data())
        env.clock.advance(HOUR)
        assert env.registry.disable("src_one").changed is False
        env.registry.enable("src_one")
        count = len(events(env))
        env.clock.advance(HOUR)
        again = env.registry.enable("src_one")
        assert again.changed is False
        assert again.source.updated_at == START + HOUR
        assert env.sources.get("src_one").updated_at == START + HOUR
        assert len(events(env)) == count

    @pytest.mark.parametrize("operation", ["enable", "disable"])
    def test_enable_and_disable_unknown_source_is_not_found(self, env: Env, operation: str):
        with pytest.raises(NotFoundError):
            getattr(env.registry, operation)("src_missing")
        assert env.control.list_control_events() == []

    def test_a_failing_audit_event_rolls_back_enable(self, env: Env):
        env.registry.add(data())
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).enable("src_one")
        assert env.sources.get("src_one").enabled is False

    def test_enable_and_edit_do_not_touch_each_others_fields(self, env: Env):
        env.registry.add(data())
        env.registry.enable("src_one")
        env.registry.edit("src_one", {"name": "Renamed"})
        stored = env.sources.get("src_one")
        assert (stored.enabled, stored.name) == (True, "Renamed")

    # ---- delete ----

    def test_delete_removes_the_source_and_its_state_but_keeps_the_audit_trail(self, env: Env):
        env.registry.add(data())
        env.control.save_source_state(SourceState("src_one", CollectionState.IDLE))
        env.registry.delete("src_one")
        assert env.registry.list() == []
        assert env.control.get_source_state("src_one") is None
        assert [k for k, _, _ in events(env, "src_one")] == [SOURCE_ADDED, SOURCE_DELETED]
        assert events(env)[-1][2] == {"type": "news_rss", "state": "idle"}

    def test_deleting_a_collecting_source_is_allowed_and_recorded(self, env: Env):
        env.registry.add(data())
        env.control.save_source_state(SourceState("src_one", CollectionState.COLLECTING))
        env.registry.delete("src_one")
        assert env.control.get_source_state("src_one") is None
        assert events(env)[-1] == (
            SOURCE_DELETED,
            "src_one",
            {"type": "news_rss", "state": "collecting"},
        )

    def test_delete_without_a_state_row_records_a_null_state(self, env: Env):
        env.registry.add(data())
        env.registry.delete("src_one")
        assert events(env)[-1][2] == {"type": "news_rss", "state": None}

    def test_delete_unknown_source_is_not_found_and_audits_nothing(self, env: Env):
        with pytest.raises(NotFoundError):
            env.registry.delete("src_missing")
        assert env.control.list_control_events() == []

    def test_delete_twice_the_second_is_not_found(self, env: Env):
        env.registry.add(data())
        env.registry.delete("src_one")
        with pytest.raises(NotFoundError):
            env.registry.delete("src_one")
        assert kinds(env) == [SOURCE_ADDED, SOURCE_DELETED]

    def test_delete_does_not_disturb_other_sources(self, env: Env):
        env.registry.add(data("src_one", "One"))
        env.registry.add(data("src_two", "Two"))
        env.control.save_source_state(SourceState("src_two", CollectionState.IDLE))
        env.registry.delete("src_one")
        assert [s.id for s in env.registry.list()] == ["src_two"]
        assert env.control.get_source_state("src_two") is not None

    def test_the_name_of_a_deleted_source_can_be_reused(self, env: Env):
        env.registry.add(data("src_one", "Reused"))
        env.registry.delete("src_one")
        env.registry.add(data("src_two", "Reused"))
        assert [s.id for s in env.registry.list()] == ["src_two"]

    def test_delete_switches_off_the_per_source_analysis_flag(self, env: Env):
        env.registry.add(data())
        env.control.set_setting(source_analysis_key("src_one"), "true")
        env.registry.delete("src_one")
        assert env.control.get_setting(source_analysis_key("src_one")) == "false"
        env.registry.add(data())
        assert env.control.get_setting(source_analysis_key("src_one")) == "false"

    def test_delete_does_not_create_an_analysis_flag_that_never_existed(self, env: Env):
        env.registry.add(data())
        env.registry.delete("src_one")
        assert env.control.get_setting(source_analysis_key("src_one")) is None

    def test_a_failing_state_cleanup_rolls_back_the_source_delete(self, env: Env):
        env.registry.add(data())
        env.control.save_source_state(SourceState("src_one", CollectionState.IDLE))
        flaky = FlakyControl(env.control, "delete_source_state")
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).delete("src_one")
        assert [s.id for s in env.registry.list()] == ["src_one"]
        assert env.control.get_source_state("src_one") is not None
        assert kinds(env) == [SOURCE_ADDED]

    def test_a_failing_audit_event_rolls_back_the_delete_and_the_state_cleanup(self, env: Env):
        env.registry.add(data())
        env.control.save_source_state(
            SourceState("src_one", CollectionState.BACKOFF, 1, START + HOUR)
        )
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).delete("src_one")
        assert [s.id for s in env.registry.list()] == ["src_one"]
        assert env.control.get_source_state("src_one") is not None
        assert kinds(env) == [SOURCE_ADDED]

    # ---- import / export ----

    def test_import_into_an_empty_registry_adds_everything_and_audits_each_plus_a_summary(
        self, env: Env
    ):
        a = parse_source(payload("news_rss", id="src_a", name="A"))
        b = parse_source(payload("instagram", id="src_b", name="B"))
        report = env.registry.import_sources(export_of(b, a))
        assert report == ImportReport(("src_a", "src_b"), (), (), dry_run=False)
        assert env.registry.list() == [a, b]
        assert events(env) == [
            (SOURCE_ADDED, "src_a", {"type": "news_rss", "via": "import"}),
            (SOURCE_ADDED, "src_b", {"type": "instagram", "via": "import"}),
            (SOURCES_IMPORTED, None, {"added": 2, "updated": 0, "unchanged": 0}),
        ]

    def test_import_is_idempotent_a_second_run_changes_and_audits_nothing(self, env: Env):
        text = export_of(
            parse_source(payload("news_rss", id="src_a", name="A")),
            parse_source(payload("live_stream", id="src_b", name="B")),
        )
        env.registry.import_sources(text)
        count = len(events(env))
        env.clock.advance(HOUR)
        report = env.registry.import_sources(text)
        assert report == ImportReport((), (), ("src_a", "src_b"), dry_run=False)
        assert len(events(env)) == count
        assert {s.updated_at for s in env.registry.list()} == {T0}

    def test_import_updates_an_existing_source_keeping_created_at_and_stamping_updated_at(
        self, env: Env
    ):
        existing = env.registry.add(data("src_a", "Old"))
        env.clock.advance(HOUR)
        incoming = existing.with_changes(name="New", credibility=4)
        report = env.registry.import_sources(export_of(incoming))
        assert report == ImportReport((), ("src_a",), (), dry_run=False)
        stored = env.sources.get("src_a")
        assert (stored.name, stored.credibility) == ("New", 4)
        assert (stored.created_at, stored.updated_at) == (START, START + HOUR)
        assert events(env)[-2:] == [
            (SOURCE_EDITED, "src_a", {"changed": ["credibility", "name"], "via": "import"}),
            (SOURCES_IMPORTED, None, {"added": 0, "updated": 1, "unchanged": 0}),
        ]

    def test_import_replaces_the_whole_entry_so_omitted_fields_fall_back_to_defaults(
        self, env: Env
    ):
        env.registry.add(data("src_a", "A"))
        env.registry.enable("src_a")
        hand_written = (
            "sources:\n- id: src_a\n  type: news_rss\n  name: A\n"
            "  feed_url: https://example.org/rss.xml\n"
        )
        report = env.registry.import_sources(hand_written)
        assert report.updated == ("src_a",)
        assert env.sources.get("src_a").enabled is False

    def test_import_fills_missing_timestamps_from_the_clock_and_keeps_given_ones(self, env: Env):
        hand_written = (
            "sources:\n- id: src_new\n  type: news_rss\n  name: N\n"
            "  feed_url: https://example.org/rss.xml\n"
        )
        env.registry.import_sources(hand_written)
        assert env.sources.get("src_new").created_at == START
        given = parse_source(payload("news_rss", id="src_old", name="O"))
        env.clock.advance(HOUR)
        env.registry.import_sources(export_of(given))
        assert env.sources.get("src_old").created_at == given.created_at

    def test_import_leaves_sources_that_are_not_in_the_file_alone(self, env: Env):
        env.registry.add(data("src_keep", "Keep"))
        env.registry.import_sources(export_of(parse_source(payload(id="src_new", name="New"))))
        assert [s.id for s in env.registry.list()] == ["src_keep", "src_new"]

    def test_import_of_an_empty_list_does_nothing(self, env: Env):
        report = env.registry.import_sources("sources: []\n")
        assert report == ImportReport((), (), (), dry_run=False)
        assert env.control.list_control_events() == []

    def test_import_allows_a_name_released_earlier_in_the_same_file(self, env: Env):
        env.registry.add(data("src_a", "Alpha"))
        renamed = env.registry.get("src_a").with_changes(name="Gamma")
        newcomer = parse_source(payload("news_rss", id="src_b", name="alpha"))
        report = env.registry.import_sources(export_of(renamed, newcomer))
        assert (report.added, report.updated) == (("src_b",), ("src_a",))

    def test_import_rejects_a_name_swap_as_a_conflict_and_applies_nothing(self, env: Env):
        a = env.registry.add(data("src_a", "Alpha"))
        b = env.registry.add(data("src_b", "Beta"))
        text = export_of(a.with_changes(name="Beta"), b.with_changes(name="Alpha"))
        with pytest.raises(ImportConflictError):
            env.registry.import_sources(text)
        assert [s.name for s in env.registry.list()] == ["Alpha", "Beta"]

    def test_import_is_atomic_when_a_later_entry_is_invalid(self, env: Env):
        text = (
            export_of(parse_source(payload(id="src_a", name="A"))).rstrip("\n")
            + "\n- id: src_b\n  type: news_rss\n  name: B\n  feed_url: ftp://nope\n"
        )
        with pytest.raises(SourceFileError) as caught:
            env.registry.import_sources(text)
        assert [(p.index, p.fields) for p in caught.value.problems] == [(1, ("feed_url",))]
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    def test_import_is_atomic_and_reports_every_conflict_together(self, env: Env):
        existing = env.registry.add(data("src_taken", "Taken"))
        other = env.registry.add(data("src_other", "Other"))
        text = export_of(
            parse_source(payload(id="src_fine", name="Fine")),
            parse_source(payload(id="src_x", name="TAKEN")),
            parse_source(payload("instagram", id="src_other", name="Other")),
            parse_source(payload(id="src_y", name="existing-clash-free")),
        )
        before = (env.registry.list(), env.control.list_control_events())
        with pytest.raises(ImportConflictError) as caught:
            env.registry.import_sources(text)
        by_sorted_id = sorted((p.index, p.code, p.fields, p.message) for p in caught.value.problems)
        assert [(i, c, f) for i, c, f, _ in by_sorted_id] == [
            (1, "conflict", ("type",)),  # the export sorts entries by id: src_other is second
            (2, "conflict", ("name",)),
        ]
        assert caught.value.code == "conflict"
        assert isinstance(caught.value, ConflictError)
        assert (env.registry.list(), env.control.list_control_events()) == before
        assert existing in env.registry.list()
        assert other in env.registry.list()

    def test_conflict_messages_never_echo_names(self, env: Env):
        env.registry.add(data("src_taken", f"Name {CANARY}"))
        clash = parse_source(payload(id="src_x", name=f"name {CANARY}"))
        with pytest.raises(ImportConflictError) as caught:
            env.registry.import_sources(export_of(clash))
        assert CANARY not in repr(caught.value.problems) + caught.value.message

    def test_import_rolls_back_every_write_when_a_later_audit_event_fails(self, env: Env):
        text = export_of(
            parse_source(payload(id="src_a", name="A")),
            parse_source(payload(id="src_b", name="B")),
        )
        flaky = FlakyControl(env.control, "append_control_event", on_call=2)
        with pytest.raises(StorageError):
            env.registry_with(control=flaky).import_sources(text)
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    def test_import_dry_run_classifies_without_writing_or_auditing(self, env: Env):
        existing = env.registry.add(data("src_a", "A"))
        before_events = env.control.list_control_events()
        text = export_of(
            existing.with_changes(name="Changed"),
            parse_source(payload(id="src_b", name="B")),
        )
        report = env.registry.import_sources(text, dry_run=True)
        assert report == ImportReport(("src_b",), ("src_a",), (), dry_run=True)
        assert env.registry.list() == [existing]
        assert env.control.list_control_events() == before_events

    def test_import_dry_run_still_rejects_invalid_files_and_conflicts(self, env: Env):
        env.registry.add(data("src_a", "A"))
        with pytest.raises(SourceFileError):
            env.registry.import_sources("sources: [1]", dry_run=True)
        with pytest.raises(ImportConflictError):
            env.registry.import_sources(
                export_of(parse_source(payload(id="src_z", name="a"))), dry_run=True
            )

    @pytest.mark.parametrize("bad", ["", "- a", "sources: 1", "sources:\n- &x {}\n- *x"])
    def test_import_rejects_bad_documents_and_applies_nothing(self, env: Env, bad: str):
        with pytest.raises(SourceFileError):
            env.registry.import_sources(bad)
        assert env.registry.list() == []
        assert env.control.list_control_events() == []

    def test_import_does_not_echo_secrets_in_errors(self, env: Env):
        text = (
            "sources:\n- id: src_a\n  type: news_rss\n  name: A\n"
            f"  feed_url: https://example.org/rss?token={CANARY}\n  password: {CANARY}\n"
        )
        with pytest.raises(SourceFileError) as caught:
            env.registry.import_sources(text)
        assert CANARY not in repr(caught.value.problems) + caught.value.message

    def test_export_then_import_into_a_fresh_registry_reproduces_every_source(self, env: Env):
        for type_, sid in zip(SOURCE_TYPES, ("src_a", "src_b", "src_c", "src_d"), strict=True):
            env.registry.add(data(sid, sid.upper(), type_))
        env.registry.enable("src_b")
        text = env.registry.export_sources()
        other = self.make_env()
        other.registry.import_sources(text)
        assert other.registry.list() == env.registry.list()
        assert other.registry.export_sources() == text

    def test_export_is_sorted_and_never_contains_secret_values(self, env: Env):
        env.registry.add(data("src_z", "Z"))
        env.registry.add(data("src_b", "B", "instagram"))
        text = env.registry.export_sources()
        assert text.index("src_b") < text.index("src_z")
        assert "secret_ref: ig_main" in text

    # ---- concurrency ----

    def test_two_threads_adding_the_same_name_one_wins_and_one_conflicts(self, env: Env):
        for trial in range(8):
            fresh = self.make_env()
            results = race(
                [
                    partial(fresh.registry.add, data("src_one", f"Contested {trial}")),
                    partial(fresh.registry.add, data("src_two", f"CONTESTED {trial}")),
                ]
            )
            outcomes = sorted(kind for kind, _ in results)
            assert outcomes == ["err", "ok"], results
            error = next(value for kind, value in results if kind == "err")
            assert isinstance(error, ConflictError)
            assert len(fresh.registry.list()) == 1
            assert kinds(fresh) == [SOURCE_ADDED]

    def test_two_threads_adding_the_same_id_one_wins_and_one_conflicts(self, env: Env):
        for _ in range(8):
            fresh = self.make_env()
            results = race(
                [
                    partial(fresh.registry.add, data("src_one", "First")),
                    partial(fresh.registry.add, data("src_one", "Second")),
                ]
            )
            assert sorted(kind for kind, _ in results) == ["err", "ok"], results
            assert isinstance(next(v for k, v in results if k == "err"), ConflictError)
            assert kinds(fresh) == [SOURCE_ADDED]

    def test_many_threads_adding_distinct_sources_all_succeed_with_one_event_each(self, env: Env):
        calls = [
            (lambda i=i: env.registry.add(data(f"src_{i:02d}", f"Name {i}"))) for i in range(10)
        ]
        results = race(calls)
        assert [kind for kind, _ in results] == ["ok"] * 10, results
        assert len(env.registry.list()) == 10
        assert kinds(env) == [SOURCE_ADDED] * 10

    def test_concurrent_enable_calls_audit_exactly_one_change(self, env: Env):
        env.registry.add(data())
        results = race([lambda: env.registry.enable("src_one") for _ in range(6)])
        assert [kind for kind, _ in results] == ["ok"] * 6
        assert sorted(value.changed for _, value in results) == [False] * 5 + [True]
        assert kinds(env).count(SOURCE_ENABLED) == 1

    def test_add_and_delete_race_never_leaves_an_event_without_matching_state(self, env: Env):
        env.registry.add(data())
        results = race(
            [
                lambda: env.registry.delete("src_one"),
                lambda: env.registry.edit("src_one", {"name": "Renamed"}),
            ]
        )
        assert all(kind in ("ok", "err") for kind, _ in results)
        remaining = env.registry.list()
        recorded = kinds(env)
        assert recorded.count(SOURCE_DELETED) == (0 if remaining else 1)
        for kind, value in results:
            if kind == "err":
                assert isinstance(value, NotFoundError)
