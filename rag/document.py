import datetime
from dataclasses import dataclass


@dataclass
class Document:
    id: int
    content: str
    created_at: datetime.datetime
    metadata: dict | None = None
