"""Compile a model spec into what one forward pass needs to know.

A ``Plan`` covers one training mode or one inference stage: which streams are live, in
what order their tokens sit, who reads whom, and which streams stay fixed for the whole
stage. Like the spec it does not import PyTorch. The attention structure comes out as
plain ``Segment`` values that map one to one onto ``AttentionSegment``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .spec import (
    ABSENT,
    CAUSAL,
    FRAME_CAUSAL,
    FULL,
    ContextRead,
    ModelSpec,
    Read,
    Stage,
    TrainMode,
    attention_groups,
)


@dataclass(frozen=True)
class Segment:
    """Query rows ``[query_start, query_end)`` read exactly the key ranges listed.

    ``no_grad`` repeats the ranges that must not pass training signal back.
    """

    query_start: int
    query_end: int
    key_ranges: tuple[tuple[int, int], ...]
    no_grad: tuple[tuple[int, int], ...] = ()


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _key_end(read: Read, start: int, end: int, query_index: int, frame_tokens: int) -> int:
    """Where this read stops for the query at ``query_index`` within its own stream."""
    if read.pattern == FULL:
        return end
    if read.pattern == CAUSAL:
        return min(end, start + query_index + 1)
    if read.pattern == FRAME_CAUSAL:
        return min(end, start + (query_index // frame_tokens + 1) * frame_tokens)
    raise ValueError(f"Unknown read pattern: {read.pattern}")


@dataclass(frozen=True)
class Plan:
    name: str
    streams: tuple[str, ...]
    rows: Mapping[str, tuple[Read, ...]]
    states: Mapping[str, str]
    towers: Mapping[str, str]
    tie: tuple[tuple[str, ...], ...] = ()
    static: frozenset = frozenset()
    use: Mapping[str, str] = field(default_factory=dict)
    steps: int = 0
    features: Mapping[str, ContextRead] = field(default_factory=dict)
    sometimes: Mapping[str, float] = field(default_factory=dict)

    def spans(self, lengths: Mapping[str, int]) -> dict[str, tuple[int, int]]:
        """Token range of every live stream. A stream of length zero has no range."""
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        for stream in self.streams:
            length = int(lengths.get(stream, 0))
            if length > 0:
                spans[stream] = (cursor, cursor + length)
                cursor += length
        return spans

    def segments(self, lengths: Mapping[str, int], frame_tokens: int = 1) -> tuple[Segment, ...]:
        """The attention structure for streams of the given token lengths.

        Rows are cut only where the keys they read change, and neighbouring rows that
        read the same keys are merged, so the result is as coarse as the structure allows.
        """
        spans = self.spans(lengths)
        segments: list[Segment] = []
        for stream, (query_start, query_end) in spans.items():
            reads = [read for read in self.rows[stream] if read.source in spans]
            if any(read.pattern == CAUSAL for read in reads):
                chunk = 1
            elif any(read.pattern == FRAME_CAUSAL for read in reads):
                chunk = frame_tokens
            else:
                chunk = query_end - query_start
            for offset in range(0, query_end - query_start, chunk):
                open_ranges, held_ranges = [], []
                for read in reads:
                    start, end = spans[read.source]
                    target = open_ranges if read.grad else held_ranges
                    target.append((start, _key_end(read, start, end, offset, frame_tokens)))
                held = _merge_ranges(held_ranges)
                keys = tuple(sorted(_merge_ranges(open_ranges) + held))
                end = min(query_start + offset + chunk, query_end)
                last = segments[-1] if segments else None
                if last and (last.key_ranges, last.no_grad) == (keys, held):
                    segments[-1] = Segment(last.query_start, end, keys, held)
                else:
                    segments.append(Segment(query_start + offset, end, keys, held))
        return tuple(segments)

    @property
    def detached(self) -> tuple[tuple[str, str], ...]:
        """Reads that must not pass training signal back, as (reader, source) pairs."""
        return tuple(
            (stream, read.source)
            for stream in self.streams
            for read in self.rows[stream]
            if not read.grad
        )

    @property
    def active_towers(self) -> tuple[str, ...]:
        """Towers with at least one live stream, in sequence order."""
        return tuple(dict.fromkeys(self.towers[stream] for stream in self.streams))

    @property
    def cached_towers(self) -> tuple[str, ...]:
        """Towers whose live streams are all fixed: they run once per stage, not per step."""
        return tuple(
            tower
            for tower in self.active_towers
            if all(stream in self.static for stream in self.streams if self.towers[stream] == tower)
        )

    @property
    def groups(self) -> tuple[tuple[str, ...], ...]:
        """Towers that share attention, in the order they run.

        A tower that reads a layer of another tower has to wait for it, so it runs later.
        """
        waiting = attention_groups(self.active_towers, self.rows, self.towers)
        ordered: list[tuple[str, ...]] = []
        finished: set[str] = set()
        while waiting:
            ready = next(
                group
                for group in waiting
                if all(self.features[tower].source in finished for tower in group if tower in self.features)
            )
            waiting.remove(ready)
            ordered.append(ready)
            finished.update(ready)
        return tuple(ordered)

    def captures(self, towers: "tuple[str, ...]") -> dict[str, int]:
        """The layers of these towers that another tower reads."""
        return {read.source: read.layer for read in self.features.values() if read.source in towers}


def _live_rows(
    spec: ModelSpec,
    overrides: Mapping[str, tuple[Read, ...]],
    live: tuple[str, ...],
    cut: tuple[tuple[str, str], ...] = (),
) -> dict[str, tuple[Read, ...]]:
    rows = spec.rows(overrides)
    return {
        stream: tuple(
            read
            for read in rows[stream]
            if read.source in live and (stream, read.source) not in cut
        )
        for stream in live
    }


def _towers(spec: ModelSpec, live: tuple[str, ...]) -> dict[str, str]:
    return {stream: spec.streams[stream].tower for stream in live}


def _features(spec: ModelSpec, live: tuple[str, ...]) -> dict[str, ContextRead]:
    towers = set(_towers(spec, live).values())
    return {tower: read for tower, read in spec.feature_reads.items() if tower in towers}


def static_streams(rows: Mapping[str, tuple[Read, ...]], changing: tuple[str, ...]) -> frozenset:
    """Streams whose attention keys and values stay fixed for a whole stage.

    A stream is fixed if it is not being denoised and everything it reads is fixed too.
    Fixed streams only need to run once per stage.
    """
    fixed = {stream for stream in rows if stream not in changing}
    while True:
        moving = {stream for stream in fixed if any(read.source not in fixed for read in rows[stream])}
        if not moving:
            return frozenset(fixed)
        fixed -= moving


def plan_train(spec: ModelSpec, mode: "str | TrainMode") -> Plan:
    if isinstance(mode, str):
        mode = next(item for item in spec.train if item.name == mode)
    live = tuple(stream for stream in spec.token_streams if mode.state(stream) != ABSENT)
    return Plan(
        name=mode.name,
        streams=live,
        rows=_live_rows(spec, mode.reads, live),
        states={stream: mode.state(stream) for stream in live},
        towers=_towers(spec, live),
        tie=mode.tie,
        features=_features(spec, live),
        sometimes=dict(mode.sometimes_noisy),
    )


def plan_stage(spec: ModelSpec, program: str, index: int) -> Plan:
    stage: Stage = spec.infer[program][index]
    live = tuple(stream for stream in spec.token_streams if stage.state(stream) != ABSENT)
    rows = _live_rows(spec, stage.reads, live, stage.cut_edges())
    return Plan(
        name=f"{program}[{index}]",
        streams=live,
        rows=rows,
        states={stream: stage.state(stream) for stream in live},
        towers=_towers(spec, live),
        tie=stage.tie,
        static=static_streams(rows, stage.denoise),
        use=dict(stage.use),
        steps=stage.steps,
        features=_features(spec, live),
    )


def plan_infer(spec: ModelSpec, program: str) -> tuple[Plan, ...]:
    return tuple(plan_stage(spec, program, index) for index in range(len(spec.infer[program])))


def sample_weights(spec: ModelSpec) -> dict[str, float]:
    """Training modes and their share of minibatches."""
    total = sum(mode.weight for mode in spec.train)
    return {mode.name: mode.weight / total for mode in spec.train}
