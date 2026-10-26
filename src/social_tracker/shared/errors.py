"""Error taxonomy. Every error carries a stable machine-readable `code`."""


class DomainError(Exception):
    """Base class for all domain errors; catch this to handle any of them."""

    code: str = "domain_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class ValidationError(DomainError):
    code = "validation_error"


class ConflictError(DomainError):
    code = "conflict"


class NotFoundError(DomainError):
    code = "not_found"


class InvalidTransition(DomainError):
    code = "invalid_transition"


class StorageError(DomainError):
    code = "storage_error"
