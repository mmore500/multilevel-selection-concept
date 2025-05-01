import secrets
import uuid


def strong_uuid4_str() -> str:
    """Generate a strong UUID4 string."""
    return str(uuid.UUID(bytes=secrets.token_bytes(16)))
