"""Reusable contract for any `ControlRepository` implementation (fake now, SQLite in 1c).

Rules every implementation must honour (and that this suite pins):
- Settings validation lives IN the repository: `validate_setting` / `validate_setting_key`
  from the control domain are applied on every write and key lookup. A SQLite adapter must
  do the same; the database is not trusted to enforce it.
- `global_analysis_enabled` accepts only "true" or "false".
- `save_source_state` does NOT check that the source exists: sources and control state live
  in separate repositories with no cross-repository foreign key.
- The event log is append-only and returned in append order, whatever the timestamps or kinds.
"""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.control.domain import (
    MAX_SETTING_KEY,
    MAX_SETTING_VALUE,
    SETTING_ALERT_MIN_CONFIDENCE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    ControlEvent,
    SourceState,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.errors import ValidationError

# The public surface of the port, read from the Protocol itself so it cannot drift.
PORT_SURFACE = frozenset(
    name
    for name, member in vars(ControlRepository).items()
    if not name.startswith("_") and callable(member)
)

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def event(kind: str = "paused", source_id: str | None = None, minutes: int = 0) -> ControlEvent:
    return ControlEvent(at=AT + timedelta(minutes=minutes), kind=kind, source_id=source_id)


class ControlRepositoryContract:
    # Extra public names an implementation may expose. Empty for every implementation today:
    # the SQLite adapter has no `close()` because the `Database` it is handed owns the
    # connection and its lifecycle. A subclass that widens this must justify each name and
    # show it cannot alter the event log.
    ALLOWED_EXTRA_PUBLIC: frozenset[str] = frozenset()

    def make_repo(self) -> ControlRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> ControlRepository:
        return self.make_repo()

    # ---- settings ----

    def test_global_analysis_defaults_to_false(self, repo: ControlRepository):
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"

    def test_a_fresh_repository_does_not_see_another_ones_settings(self):
        first = self.make_repo()
        first.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
        second = self.make_repo()
        assert second.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"

    def test_unknown_setting_is_none(self, repo: ControlRepository):
        assert repo.get_setting("unknown_key") is None

    def test_set_then_get_and_overwrite(self, repo: ControlRepository):
        repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true"
        repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "false")
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"

    def test_settings_are_independent(self, repo: ControlRepository):
        repo.set_setting("a_key", "1")
        repo.set_setting("b_key", "2")
        assert (repo.get_setting("a_key"), repo.get_setting("b_key")) == ("1", "2")

    def test_unicode_and_empty_values_round_trip(self, repo: ControlRepository):
        repo.set_setting("note", "café 新聞")
        repo.set_setting("blank", "")
        assert repo.get_setting("note") == "café 新聞"
        assert repo.get_setting("blank") == ""

    @pytest.mark.parametrize("key", ["", "   ", "k" * 101])
    def test_invalid_key_is_rejected_on_set_and_get(self, repo: ControlRepository, key: str):
        with pytest.raises(ValidationError):
            repo.set_setting(key, "v")
        with pytest.raises(ValidationError):
            repo.get_setting(key)

    @pytest.mark.parametrize(
        "key",
        ["\x00", "\x00k", "k\x00", "k\x00k", "k\x1fk", "k\x7fk", "k\x85k", "k\x9fk", "k\tk"],
    )
    def test_control_characters_in_a_setting_key_are_rejected(
        self, repo: ControlRepository, key: str
    ):
        with pytest.raises(ValidationError):
            repo.set_setting(key, "v")
        with pytest.raises(ValidationError):
            repo.get_setting(key)

    def test_invalid_value_is_rejected_and_keeps_the_old_one(self, repo: ControlRepository):
        repo.set_setting("k_one", "ok")
        with pytest.raises(ValidationError):
            repo.set_setting("k_one", None)  # type: ignore[arg-type]
        assert repo.get_setting("k_one") == "ok"

    def test_setting_key_boundaries(self, repo: ControlRepository):
        edge = "k" * MAX_SETTING_KEY
        repo.set_setting(edge, "v")
        assert repo.get_setting(edge) == "v"
        with pytest.raises(ValidationError):
            repo.set_setting(edge + "k", "v")

    @pytest.mark.parametrize("size", [MAX_SETTING_VALUE - 1, MAX_SETTING_VALUE])
    def test_setting_value_up_to_the_limit_round_trips(self, repo: ControlRepository, size: int):
        repo.set_setting("big_value", "v" * size)
        assert repo.get_setting("big_value") == "v" * size

    def test_setting_value_over_the_limit_is_rejected_and_keeps_the_old_one(
        self, repo: ControlRepository
    ):
        repo.set_setting("big_value", "small")
        with pytest.raises(ValidationError):
            repo.set_setting("big_value", "v" * (MAX_SETTING_VALUE + 1))
        assert repo.get_setting("big_value") == "small"

    def test_non_string_key_is_rejected(self, repo: ControlRepository):
        with pytest.raises(ValidationError):
            repo.set_setting(5, "v")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            repo.get_setting(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["true", "false"])
    def test_global_analysis_accepts_only_true_or_false(self, repo: ControlRepository, value: str):
        repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, value)
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == value

    def test_global_analysis_rejects_other_values_and_keeps_the_old_one(
        self, repo: ControlRepository
    ):
        repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
        for value in ["", "TRUE", "False", "1", "0", "yes", " true"]:
            with pytest.raises(ValidationError):
                repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, value)
            assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true"

    def test_alert_threshold_defaults_to_one_half(self, repo: ControlRepository):
        assert repo.get_setting(SETTING_ALERT_MIN_CONFIDENCE) == "0.5"

    @pytest.mark.parametrize("value", ["0", "0.0", "0.5", "0.75", "1", "1.0", "0.000001"])
    def test_alert_threshold_accepts_plain_decimals_in_range(
        self, repo: ControlRepository, value: str
    ):
        repo.set_setting(SETTING_ALERT_MIN_CONFIDENCE, value)
        assert repo.get_setting(SETTING_ALERT_MIN_CONFIDENCE) == value

    @pytest.mark.parametrize("value", ["", "abc", "nan", "inf", "-0.1", "1.01", "0.5e0", " 0.5"])
    def test_alert_threshold_rejects_anything_else_and_keeps_the_old_value(
        self, repo: ControlRepository, value: str
    ):
        repo.set_setting(SETTING_ALERT_MIN_CONFIDENCE, "0.9")
        with pytest.raises(ValidationError):
            repo.set_setting(SETTING_ALERT_MIN_CONFIDENCE, value)
        assert repo.get_setting(SETTING_ALERT_MIN_CONFIDENCE) == "0.9"

    # ---- source state ----

    def test_missing_source_state_is_none(self, repo: ControlRepository):
        assert repo.get_source_state("src_one") is None

    def test_save_and_get_round_trip_including_backoff_schedule(self, repo: ControlRepository):
        saved = SourceState("src_one", CollectionState.BACKOFF, 2, AT + timedelta(seconds=120))
        repo.save_source_state(saved)
        got = repo.get_source_state("src_one")
        assert got == saved
        assert got.next_run_at.utcoffset() == timedelta(0)  # type: ignore[union-attr]

    def test_save_is_an_upsert(self, repo: ControlRepository):
        # Every save replaces EVERY column of the row, whichever column changes. The sequence
        # moves each of state, consecutive_failures and next_run_at on its own and together,
        # including back to NULL, and compares the whole state after each save.
        t1, t2 = AT + timedelta(minutes=10), AT + timedelta(minutes=45)
        sequence = [
            SourceState("src_one", CollectionState.IDLE),
            SourceState("src_one", CollectionState.COLLECTING),
            SourceState("src_one", CollectionState.COLLECTING, 3),
            SourceState("src_one", CollectionState.COLLECTING, 4),
            SourceState("src_one", CollectionState.BACKOFF, 4, t1),
            SourceState("src_one", CollectionState.BACKOFF, 4, t2),
            SourceState("src_one", CollectionState.BACKOFF, 1, t2),
            SourceState("src_one", CollectionState.FAILED, 5),
            SourceState("src_one", CollectionState.DISABLED),
            SourceState("src_one", CollectionState.IDLE, 2),
        ]
        for state in sequence:
            repo.save_source_state(state)
            assert repo.get_source_state("src_one") == state
        assert repo.get_source_state("src_two") is None  # a save never touches another source

    def test_sources_are_independent(self, repo: ControlRepository):
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.save_source_state(SourceState("src_two", CollectionState.FAILED, 5))
        assert repo.get_source_state("src_one").state is CollectionState.IDLE  # type: ignore[union-attr]
        assert repo.get_source_state("src_two").consecutive_failures == 5  # type: ignore[union-attr]
        assert repo.get_source_state("src_three") is None

    def test_get_source_state_with_blank_id_is_none(self, repo: ControlRepository):
        assert repo.get_source_state("") is None

    def test_saving_state_for_an_unregistered_source_is_allowed(self, repo: ControlRepository):
        # No cross-repository foreign key: control state may exist before/without the source.
        ghost = SourceState("src_ghost", CollectionState.IDLE)
        repo.save_source_state(ghost)
        assert repo.get_source_state("src_ghost") == ghost

    # ---- delete_source_state (SR-9 cleanup, called by the service when a source is deleted) ----

    def test_delete_source_state_removes_only_that_state(self, repo: ControlRepository):
        keep = SourceState("src_two", CollectionState.FAILED, 5)
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.save_source_state(keep)
        assert repo.delete_source_state("src_one") is None
        assert repo.get_source_state("src_one") is None
        assert repo.get_source_state("src_two") == keep

    def test_delete_source_state_is_idempotent(self, repo: ControlRepository):
        repo.delete_source_state("src_never_saved")  # absent: a no-op, not an error
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.delete_source_state("src_one")
        repo.delete_source_state("src_one")
        repo.delete_source_state("")
        assert repo.get_source_state("src_one") is None

    def test_delete_source_state_keeps_the_audit_trail_and_settings(self, repo: ControlRepository):
        repo.set_setting("note", "kept")
        repo.append_control_event(event("source_deleted", "src_one"))
        repo.append_control_event(event("paused"))
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.delete_source_state("src_one")
        assert [e.kind for e in repo.list_control_events("src_one")] == ["source_deleted"]
        assert len(repo.list_control_events()) == 2
        assert repo.get_setting("note") == "kept"

    def test_a_state_can_be_saved_again_after_it_was_deleted(self, repo: ControlRepository):
        repo.save_source_state(SourceState("src_one", CollectionState.FAILED, 5))
        repo.delete_source_state("src_one")
        fresh = SourceState("src_one", CollectionState.IDLE)
        repo.save_source_state(fresh)
        assert repo.get_source_state("src_one") == fresh

    def test_saving_state_does_not_touch_settings_or_events(self, repo: ControlRepository):
        repo.append_control_event(event("a_kind"))
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        assert len(repo.list_control_events()) == 1
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"

    # ---- control events (append-only) ----

    def test_no_events_initially(self, repo: ControlRepository):
        assert repo.list_control_events() == []

    def test_events_are_listed_in_append_order(self, repo: ControlRepository):
        first, second, third = (
            event("a_kind", "src_one"),
            event("b_kind"),
            event("c_kind", "src_two"),
        )
        for e in (first, second, third):
            repo.append_control_event(e)
        assert repo.list_control_events() == [first, second, third]

    def test_append_order_wins_over_timestamp_order_and_kind_order(self, repo: ControlRepository):
        # Out-of-order timestamps, repeated timestamps and kinds in non-alphabetical order:
        # sorting by `at` or by `kind` must produce a different list than append order.
        appended = [
            event("zulu", minutes=10),
            event("mike", minutes=0),
            event("alpha", minutes=10),
            event("bravo", minutes=5),
        ]
        for e in appended:
            repo.append_control_event(e)
        assert repo.list_control_events() == appended

    def test_same_timestamp_events_keep_append_order(self, repo: ControlRepository):
        appended = [event("c_kind"), event("a_kind"), event("b_kind")]
        for e in appended:
            repo.append_control_event(e)
        assert repo.list_control_events() == appended

    def test_filtered_listing_keeps_append_order_too(self, repo: ControlRepository):
        appended = [
            event("zulu", "src_one", minutes=9),
            event("noise", "src_two", minutes=1),
            event("alpha", "src_one", minutes=2),
            event("mike", "src_one", minutes=2),
        ]
        for e in appended:
            repo.append_control_event(e)
        assert repo.list_control_events("src_one") == [appended[0], appended[2], appended[3]]

    def test_an_empty_source_filter_is_not_the_same_as_no_filter(self, repo: ControlRepository):
        repo.append_control_event(event("global_kind"))
        repo.append_control_event(event("scoped_kind", "src_one"))
        assert repo.list_control_events("") == []
        assert len(repo.list_control_events(None)) == 2
        assert len(repo.list_control_events()) == 2

    def test_events_can_be_filtered_by_source(self, repo: ControlRepository):
        mine, other, global_ = (
            event("a_kind", "src_one"),
            event("b_kind", "src_two"),
            event("g_kind"),
        )
        for e in (mine, other, global_):
            repo.append_control_event(e)
        assert repo.list_control_events("src_one") == [mine]
        assert repo.list_control_events("src_missing") == []

    def test_identical_events_are_both_kept(self, repo: ControlRepository):
        repo.append_control_event(event())
        repo.append_control_event(event())
        assert len(repo.list_control_events()) == 2

    def test_unicode_detail_round_trips(self, repo: ControlRepository):
        e = ControlEvent(at=AT, kind="note", detail="café 新聞")
        repo.append_control_event(e)
        assert repo.list_control_events() == [e]

    def test_mutating_the_returned_list_does_not_alter_the_log(self, repo: ControlRepository):
        repo.append_control_event(event())
        repo.list_control_events().clear()
        assert len(repo.list_control_events()) == 1

    def test_the_implementation_exposes_nothing_beyond_the_port(self, repo: ControlRepository):
        # An allow-list, not a blacklist of scary names: `purge_events`, `truncate_log` or any
        # other public addition fails here, whatever it is called.
        public = {name for name in dir(repo) if not name.startswith("_")}
        assert public - PORT_SURFACE - self.ALLOWED_EXTRA_PUBLIC == set()

    def test_the_port_surface_is_what_the_contract_expects(self):
        assert {
            "get_setting",
            "set_setting",
            "get_source_state",
            "save_source_state",
            "delete_source_state",
            "append_control_event",
            "list_control_events",
        } == PORT_SURFACE

    def test_a_long_log_is_never_truncated(self, repo: ControlRepository):
        appended = [event(f"kind_{i}", minutes=i) for i in range(250)]
        for e in appended:
            repo.append_control_event(e)
        assert repo.list_control_events() == appended

    def test_events_never_change_or_disappear_as_the_log_grows(self, repo: ControlRepository):
        appended: list[ControlEvent] = []
        for i in range(60):
            e = event(f"kind_{i}", source_id="src_one" if i % 2 else None, minutes=60 - i)
            repo.append_control_event(e)
            appended.append(e)
            if i % 7 == 0:
                assert repo.list_control_events() == appended
        assert repo.list_control_events() == appended

    def test_other_writes_never_touch_the_log(self, repo: ControlRepository):
        e = event("a_kind", "src_one")
        repo.append_control_event(e)
        repo.set_setting("some_key", "v")
        repo.save_source_state(SourceState("src_one", CollectionState.COLLECTING))
        assert repo.list_control_events() == [e]

    def test_appending_a_non_event_is_rejected(self, repo: ControlRepository):
        with pytest.raises(ValidationError):
            repo.append_control_event({"kind": "x"})  # type: ignore[arg-type]
        assert repo.list_control_events() == []

    def test_concurrent_appends_are_all_recorded(self, repo: ControlRepository):
        total = 40
        barrier = threading.Barrier(total)

        def worker(i: int):
            barrier.wait(timeout=5)
            repo.append_control_event(event(f"kind_{i}", minutes=i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(total)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        kinds = {e.kind for e in repo.list_control_events()}
        assert kinds == {f"kind_{i}" for i in range(total)}
