"""Domain errors returned by the laboratory 5 service."""


class SynthesisError(RuntimeError):
    """Base class for expected synthesis failures."""


class ArtifactError(SynthesisError):
    """A required model artifact is missing or incompatible."""


class InputTextError(SynthesisError):
    """The supplied text cannot be processed by the trained pipeline."""


class UnknownWordError(InputTextError):
    """No compatible pronunciation is available for a word."""

