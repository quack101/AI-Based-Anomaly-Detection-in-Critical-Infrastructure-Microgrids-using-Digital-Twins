"""
streaming/windowing.py -- decides when a timestamp's buffered messages are final,
for a stage reading a multi-partition topic.

One timestep's messages are spread over several partitions, and a partition can
pause while its leader fails over to another broker. A plain wall-clock timeout
cannot tell that pause from a dropped meter: it would close the window early, and
the messages arriving afterwards would reopen it. Instead a window closes on what
has been read, relying on two facts: the producer publishes timestamps in order,
and each partition preserves that order.

  * Once one window is complete, every meter has moved on to that timestamp, so
    nothing more can arrive for an older window.
  * Once every partition has been read past t, nothing more can arrive for t, so
    a dropped meter closes t at once (no timeout to wait out).
  * Backstop: if no window closes for stall_timeout_s while messages keep arriving,
    the partitions holding the rest back are set aside until they deliver again.
    The default is well above a failover, which takes seconds. This only matters
    when a partition has no leader at all.

Windows close in timestamp order and each closes once; a message arriving after
its window closed is counted in `late` and dropped.
"""

import time

DEFAULT_STALL_TIMEOUT_S = 60.0


class WindowGate:
    def __init__(self, partitions, stall_timeout_s=DEFAULT_STALL_TIMEOUT_S, clock=time.monotonic):
        if not partitions:
            raise ValueError("no partitions: create the topics first (python -m streaming.demo_tools reset)")
        self.high = {p: None for p in partitions}   # partition -> highest timestamp read
        self.set_aside = set()                      # stalled partitions, until they deliver again
        self.closed_upto = None
        self.late = 0
        self.stall_timeout_s = stall_timeout_s
        self._clock = clock
        self._waiting_since = None

    def observe(self, partition, t):
        """Record that t was read from partition."""
        high = self.high.get(partition)
        self.high[partition] = t if high is None else max(high, t)
        self.set_aside.discard(partition)

    def accept(self, partition, t):
        """observe(), for a message that belongs in a window. False if its window already closed."""
        self.observe(partition, t)
        if self.closed_upto is not None and t <= self.closed_upto:
            self.late += 1
            return False
        return True

    def closable(self, open_timestamps, is_complete):
        """The open timestamps that are now final, oldest first."""
        open_ts = sorted(open_timestamps)
        if not open_ts:
            return []
        now = self._clock()
        if self._waiting_since is None:
            self._waiting_since = now
        elif now - self._waiting_since >= self.stall_timeout_s:
            self._set_aside_laggards()
            self._waiting_since = now

        final_before = self._read_past()
        ready = []
        last_complete = max((t for t in open_ts if is_complete(t)), default=None)
        for t in open_ts:
            if (last_complete is not None and t <= last_complete) or (final_before is not None and t < final_before):
                ready.append(t)
        if ready:
            self.closed_upto = ready[-1]
            self._waiting_since = now if len(ready) < len(open_ts) else None
        return ready

    def _read_past(self):
        highs = [h for p, h in self.high.items() if p not in self.set_aside]
        if not highs or None in highs:
            return None
        return min(highs)

    def _set_aside_laggards(self):
        highs = [h for p, h in self.high.items() if p not in self.set_aside and h is not None]
        lowest = min(highs) if highs else None
        for p, h in self.high.items():
            if h is None or h == lowest:
                self.set_aside.add(p)
