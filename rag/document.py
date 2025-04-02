import uuid
import datetime
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    id: int
    collection_uuid: uuid.UUID
    content: str
    created_at: datetime.datetime
    metadata: dict | None = None
