"""The in-memory classification ports satisfy their contracts (they are the reference)."""

from tests.classification.repository_contracts import (
    ClassificationRepositoryContract,
    SignatureRepositoryContract,
    TopicConfigRepositoryContract,
)
from tests.fakes.classification_repositories import (
    InMemoryClassificationRepository,
    InMemorySignatureRepository,
    InMemoryTopicConfigRepository,
)


class TestInMemoryClassificationRepository(ClassificationRepositoryContract):
    def make_repo(self):
        return InMemoryClassificationRepository()


class TestInMemorySignatureRepository(SignatureRepositoryContract):
    def make_repo(self):
        return InMemorySignatureRepository()


class TestInMemoryTopicConfigRepository(TopicConfigRepositoryContract):
    def make_repo(self):
        return InMemoryTopicConfigRepository()
