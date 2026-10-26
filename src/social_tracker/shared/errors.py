"""Error taxonomy. Every error carries a stable machine-readable `code`."""


class DomainError(Exception):
    """Base class for all domain errors; catch this to handle any of them."""

    code: str = "domain_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class ValidationError(DomainError):
    """Invalid input. `fields` lists the KNOWN field names involved (never user-supplied keys)."""

    code = "validation_error"

    def __init__(self, message: str = "", fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.fields: tuple[str, ...] = tuple(fields)


class ConflictError(DomainError):
    code = "conflict"


class NotFoundError(DomainError):
    code = "not_found"


class InvalidTransition(DomainError):
    code = "invalid_transition"


class StorageError(DomainError):
    code = "storage_error"
