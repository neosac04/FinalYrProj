from __future__ import annotations
import threading
from collections import OrderedDict
from app.schemas.response import DetectionResponse


class ResultCache:
    """Thread-safe in-memory LRU cache for detection results and image bytes (max 200 entries)."""

    _instance: ResultCache | None = None

    def __init__(self, maxsize: int = 200) -> None:
        self._store: OrderedDict[str, DetectionResponse] = OrderedDict()
        self._images: dict[str, bytes] = {}
        self._maxsize = maxsize
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ResultCache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set(self, result_id: str, result: DetectionResponse, image_bytes: bytes | None = None) -> None:
        with self._lock:
            if result_id in self._store:
                self._store.move_to_end(result_id)
            self._store[result_id] = result
            if image_bytes is not None:
                self._images[result_id] = image_bytes
            if len(self._store) > self._maxsize:
                evicted_id, _ = self._store.popitem(last=False)
                self._images.pop(evicted_id, None)

    def get(self, result_id: str) -> DetectionResponse | None:
        with self._lock:
            item = self._store.get(result_id)
            if item:
                self._store.move_to_end(result_id)
            return item
