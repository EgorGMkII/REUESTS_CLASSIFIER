class ClassificationError(Exception):
    """Base error raised by the classification package."""


class ClassificationValidationError(ClassificationError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            "Classification result validation failed: " + "; ".join(errors)
        )
