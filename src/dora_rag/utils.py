"""Utility functions shared across modules."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from .models import Config


def get_qdrant_client(config: Config) -> QdrantClient:
    """Create a Qdrant client based on configuration.

    Constructs a QdrantClient by reading connection parameters from the
    provided ``Config`` object.  The ``qdrant`` dict must contain a
    ``location`` key (an HTTP/HTTPS URL, a local path, or a special value
    like ``":memory:"``).  URL-based locations are parsed into host and
    port; everything else is passed through as ``location``.

    Args:
        config: Application settings.  Relevant fields inside
            ``config.qdrant``:

            * ``location`` (str): Qdrant endpoint URL or path.
            * ``host`` (str): Overrides host extracted from location.
            * ``port`` (int): Overrides port extracted from location.
            * ``grpc`` (bool): Use gRPC protocol (default False).
            * ``timeout`` (int): Request timeout in seconds.

    Returns:
        qdrant_client.QdrantClient: Configured client instance.

    Example:
        >>> from dora_rag.models import Config
        >>> cfg = Config(qdrant={"location": "http://localhost:6333"})
        >>> client = get_qdrant_client(cfg)
        >>> isinstance(client, QdrantClient)
        True
    """
    location = config.qdrant["location"]
    if location.startswith("http://") or location.startswith("https://"):
        # Extract host and port
        host_str = location.replace("http://", "").replace("https://", "")
        if ":" in host_str:
            host, port_str = host_str.split(":", 1)
            port = int(port_str)
        else:
            host = host_str
            port = 6333
        return QdrantClient(host=host, port=port)
    else:
        # Treat as path or special location like ":memory:"
        return QdrantClient(location=location)


def get_vector_params(config: Config) -> VectorParams:
    """Get VectorParams for Qdrant collection from config."""
    vector_size = config.qdrant["vector_size"]
    distance = Distance.COSINE  # hardcoded as in ingestion.py
    return VectorParams(size=vector_size, distance=distance)
