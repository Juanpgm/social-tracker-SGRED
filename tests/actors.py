"""Shared actor/reason pair for tests that call audited service operations."""

ACTOR = "operator"
WHY = "scheduled maintenance"
WHO = {"actor": ACTOR, "reason": WHY}
