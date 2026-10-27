"""The in-memory escalation repository satisfies its contract (it is the reference)."""

from tests.fakes.classification_repositories import InMemoryClassificationRepository
from tests.fakes.escalation_repository import InMemoryEscalationRepository
from tests.fakes.item_repository import InMemoryItemRepository
from tests.llm.repository_contracts import EscalationRepositoryContract, Rig


class TestInMemoryEscalationRepository(EscalationRepositoryContract):
    def make_rig(self) -> Rig:
        classifications, items = InMemoryClassificationRepository(), InMemoryItemRepository()
        return Rig(InMemoryEscalationRepository(classifications, items), classifications, items)
