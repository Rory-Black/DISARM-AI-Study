"""Structured event plumbing between the classification engine and any observer.

The engine already logs what it is doing to stderr via `Log`, but text logs are
awkward to build a UI on. This module carries the same activity as structured
events so a consumer (the GUI) can show *what the agent is working on right
now*, what it searched for, what evidence it gathered and what it concluded.

Nothing here is required by the engine: if no observer is attached, emitting an
event is a no-op, so the CLI paths behave exactly as they did before.
"""

import queue
import threading
import time


class Cancelled(Exception):
    """Raised inside a run when the observer has asked it to stop."""


class EventBus:
    """Fan-out of run events to any number of subscribers.

    Subscribers get their own unbounded queue, so a slow consumer (a browser
    tab that has been backgrounded) can never block the classification thread.
    """

    def __init__(self, history_limit=5000):
        self._lock = threading.Lock()
        self._subscribers = []
        self._history = []
        self._history_limit = history_limit
        self._seq = 0

    def subscribe(self, replay=True):
        q = queue.Queue()
        with self._lock:
            if replay:
                for event in self._history:
                    q.put(event)
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def emit(self, event_type, **payload):
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "type": event_type, "ts": time.time()}
            event.update(payload)
            self._history.append(event)
            if len(self._history) > self._history_limit:
                # keep the tail; a reconnecting browser gets recent context rather
                # than the whole history of a long dataset run
                del self._history[: len(self._history) - self._history_limit]
            subscribers = list(self._subscribers)

        for q in subscribers:
            q.put(event)
        return event

    def history(self):
        with self._lock:
            return list(self._history)

    def clear(self):
        with self._lock:
            self._history.clear()
            self._seq = 0


def truncate(text, limit=400):
    """Shorten free text for display without dragging whole articles into the UI."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
