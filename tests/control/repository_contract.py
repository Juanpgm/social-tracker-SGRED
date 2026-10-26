"""Reusable contract for any `ControlRepository` implementation (fake now, SQLite in 1c)."""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.control.domain import (
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    ControlEvent,
    SourceState,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.errors import ValidationError

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def event(kind: str = "paused", source_id: str | None = None, minutes: int = 0) -> ControlEvent:
    return ControlEvent(at=AT + timedelta(minutes=minutes), kind=kind, source_id=source_id)


class ControlRepositoryContract:
    def make_repo(self) -> ControlRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> ControlRepository:
        return self.make_repo()

    # ---- settings ----

    def test_global_analysis_defaults_to_false(self, repo: ControlRepository):
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"

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

    def test_invalid_value_is_rejected_and_keeps_the_old_one(self, repo: ControlRepository):
        repo.set_setting("k_one", "ok")
        with pytest.raises(ValidationError):
            repo.set_setting("k_one", None)  # type: ignore[arg-type]
        assert repo.get_setting("k_one") == "ok"

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
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.save_source_state(SourceState("src_one", CollectionState.COLLECTING))
        assert repo.get_source_state("src_one").state is CollectionState.COLLECTING  # type: ignore[union-attr]

    def test_sources_are_independent(self, repo: ControlRepository):
        repo.save_source_state(SourceState("src_one", CollectionState.IDLE))
        repo.save_source_state(SourceState("src_two", CollectionState.FAILED, 5))
        assert repo.get_source_state("src_one").state is CollectionState.IDLE  # type: ignore[union-attr]
        assert repo.get_source_state("src_two").consecutive_failures == 5  # type: ignore[union-attr]
        assert repo.get_source_state("src_three") is None

    def test_get_source_state_with_blank_id_is_none(self, repo: ControlRepository):
        assert repo.get_source_state("") is None

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

    def test_the_port_exposes_no_way_to_change_or_remove_events(self, repo: ControlRepository):
        public = {name for name in dir(repo) if not name.startswith("_")}
        mutators = {
            n
            for n in public
            if "event" in n
            and n.split("_")[0] in {"update", "delete", "remove", "clear", "set", "edit"}
        }
        assert mutators == set()

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
