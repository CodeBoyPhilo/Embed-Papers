class EmbedPapersError(Exception):
    """Base exception for embed-papers."""


class InvalidPapersFileError(EmbedPapersError):
    """Raised when a papers file is missing required fields."""


class CacheMissRequiresApiKeyError(EmbedPapersError):
    """Raised when embeddings are missing and no OpenAI API key is available."""


class ExternalServiceError(EmbedPapersError):
    """Raised when a remote service request fails after retries."""


class OpenReviewRequestError(ExternalServiceError):
    """Raised when OpenReview API requests fail."""


class EmbeddingRequestError(ExternalServiceError):
    """Raised when embedding API requests fail."""


class NoPapersFoundError(EmbedPapersError):
    """Raised when crawl returns zero papers and empty results are not allowed."""
