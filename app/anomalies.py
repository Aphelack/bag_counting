"""Anomaly monitoring for the counting pipeline.

Each anomaly type is its own small class implementing `AnomalyRule` — a
per-frame `check(ctx)` that returns zero or more `Anomaly` records.
`AnomalyMonitor` just runs the configured rules and collects the results;
adding a new anomaly type means adding a new `AnomalyRule` class and
listing it in `AnomalyMonitor.DEFAULT_RULES`, nothing else in the
pipeline needs to change.
"""
from dataclasses import dataclass
from typing import Protocol

from app.detector import Detection
from app.models import Anomaly


@dataclass
class FrameContext:
    frame_idx: int
    timestamp_sec: float
    detections: list[Detection]
    bag_count: int
    forward_crossings: int  # zone crossings *this frame*, not the running total
    reverse_crossings: int


class AnomalyRule(Protocol):
    def check(self, ctx: FrameContext) -> list[Anomaly]: ...


class ReverseDirectionRule:
    """Flags each reverse-direction zone crossing.

    The counter already nets these out of the total automatically (see
    app/tracker.py) — this just surfaces them as a visible event, since a
    bag moving backwards on the belt (jostled, jammed, someone reaching in)
    is exactly the kind of thing an operator would want flagged even though
    the count itself stays correct.
    """

    def check(self, ctx: FrameContext) -> list[Anomaly]:
        if ctx.reverse_crossings <= 0:
            return []
        return [
            Anomaly(
                frame=ctx.frame_idx,
                timestamp_sec=ctx.timestamp_sec,
                kind="reverse_direction",
                message="Bag moved backward on the belt",
            )
        ]


class DetectionGapRule:
    """Flags a long stretch with zero detections — possible belt stoppage,
    camera obstruction, or the detector losing the scene entirely.

    Fires once per gap (not every frame the gap continues), and resets as
    soon as a detection appears again. Threshold is in seconds, not
    frames, so it doesn't silently assume a particular fps.
    """

    def __init__(self, gap_seconds_threshold: float = 5.0) -> None:
        self._threshold = gap_seconds_threshold
        self._gap_start_sec: float | None = None
        self._flagged_this_streak = False

    def check(self, ctx: FrameContext) -> list[Anomaly]:
        if ctx.detections:
            self._gap_start_sec = None
            self._flagged_this_streak = False
            return []

        if self._gap_start_sec is None:
            self._gap_start_sec = ctx.timestamp_sec

        gap_duration = ctx.timestamp_sec - self._gap_start_sec
        if gap_duration < self._threshold or self._flagged_this_streak:
            return []

        self._flagged_this_streak = True
        return [
            Anomaly(
                frame=ctx.frame_idx,
                timestamp_sec=ctx.timestamp_sec,
                kind="detection_gap",
                message=f"No bags detected for {gap_duration:.0f}s",
            )
        ]


class AnomalyMonitor:
    DEFAULT_RULES: tuple[type[AnomalyRule], ...] = (ReverseDirectionRule, DetectionGapRule)

    def __init__(self, rules: list[AnomalyRule] | None = None) -> None:
        self._rules = rules if rules is not None else [rule_cls() for rule_cls in self.DEFAULT_RULES]

    def check(self, ctx: FrameContext) -> list[Anomaly]:
        anomalies = []
        for rule in self._rules:
            anomalies.extend(rule.check(ctx))
        return anomalies
