class EmbedPapersError(Exception):
    """Base exception for embed-papers."""


class InvalidPapersFileError(EmbedPapersError):
    """Raised when a papers file is missing required fields."""


class CacheMissRequiresApiKeyError(EmbedPapersError):
    """Raised when embeddings are missing and no OpenAI API key is available."""
