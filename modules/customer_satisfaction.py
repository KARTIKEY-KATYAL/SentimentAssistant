import time
from typing import List, Dict, Any


class CustomerSatisfactionTracker:
    """Track and analyze customer satisfaction feedback for optimization."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_feedback(self, message_id: str, rating: int, comment: str | None = None):
        feedback = {
            "timestamp": time.time(),
            "message_id": message_id,
            "rating": int(rating),
            "comment": comment or ""
        }
        self.history.append(feedback)
        return feedback

    def average_rating(self, last_n: int | None = None) -> float | None:
        data = self.history[-last_n:] if last_n else self.history
        if not data:
            return None
        return sum(f["rating"] for f in data) / len(data)

    def rating_trend(self, window: int = 5) -> str:
        if len(self.history) < window * 2:
            return "insufficient_data"
        first = self.history[:window]
        last = self.history[-window:]
        first_avg = sum(f["rating"] for f in first) / window
        last_avg = sum(f["rating"] for f in last) / window
        if last_avg > first_avg + 0.4:
            return "improving"
        if last_avg < first_avg - 0.4:
            return "declining"
        return "stable"

    def export(self) -> List[Dict[str, Any]]:
        return self.history
