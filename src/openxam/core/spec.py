"""A model described as data.

An OpenXAM model is a configuration rather than a class. It names the towers, the token
streams that run through them, which stream may read which, how noise is applied in
training and at inference, and what is supervised.

This module holds that description and checks it. It does not import PyTorch, so a
configuration can be checked anywhere, before any GPU time is spent. See
``docs/design/multi_tower_core_zh.md`` for the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

FULL = "full"
FRAME_CAUSAL = "frame_causal"
CAUSAL = "causal"
READ_PATTERNS = (FULL, FRAME_CAUSAL, CAUSAL)

TRAIN_POLICIES = ("full", "lora", "frozen")

FLOW = "flow"
MASKED = "masked"
FEATURE = "feature"
LOSS_TYPES = (FLOW, MASKED, FEATURE)
NOISE_LOSSES = (FLOW, MASKED)

CLEAN = "clean"
NOISY = "noisy"
EITHER = "either"
HOLD = "hold"
ABSENT = "absent"


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a configuration. ``code`` names the kind of check."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class SpecError(ValueError):
    """A configuration that cannot be used, with everything found wrong in it."""

    def __init__(self, problems: Sequence[Problem]):
        self.problems = tuple(problems)
        super().__init__("\n".join(str(problem) for problem in self.problems))


def _only_keys(raw: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = set(raw) - set(allowed)
    if unknown:
        raise SpecError([Problem("format", f"{where}: unknown keys {sorted(unknown)}")])


def _names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _groups(value: Any) -> tuple[tuple[str, ...], ...]:
    return tuple(_names(group) for group in (value or ()))


@dataclass(frozen=True)
class Read:
    """A stream reading another stream through joint attention."""

    source: str
    grad: bool = True
    pattern: str = FULL

    @classmethod
    def parse(cls, raw: Any, where: str) -> "Read":
        if isinstance(raw, str):
            return cls(raw)
        _only_keys(raw, ("from", "grad", "pattern"), where)
        if "from" not in raw:
            raise SpecError([Problem("format", f"{where}: a read needs `from`")])
        pattern = raw.get("pattern", FULL)
        if pattern not in READ_PATTERNS:
            raise SpecError([Problem("format", f"{where}: unknown pattern {pattern!r}")])
        return cls(str(raw["from"]), bool(raw.get("grad", True)), pattern)


def _parse_rows(raw: Optional[Mapping[str, Any]], where: str) -> dict[str, tuple[Read, ...]]:
    return {
        str(reader): tuple(Read.parse(item, f"{where}.{reader}") for item in row)
        for reader, row in (raw or {}).items()
    }


@dataclass(frozen=True)
class ContextRead:
    """A fixed sequence a tower reads through cross-attention at every layer.

    The source is a stream without a tower, such as a text encoder output, or the name
    of another tower together with the layer whose features are read.
    """

    source: str
    layer: Optional[int] = None
    grad: bool = True

    @classmethod
    def parse(cls, raw: Any, where: str) -> "ContextRead":
        if isinstance(raw, str):
            return cls(raw)
        _only_keys(raw, ("from", "layer", "grad"), where)
        if "from" not in raw:
            raise SpecError([Problem("format", f"{where}: a context read needs `from`")])
        layer = raw.get("layer")
        return cls(str(raw["from"]), None if layer is None else int(layer), bool(raw.get("grad", True)))


@dataclass(frozen=True)
class TowerShape:
    layers: int
    heads: int


# Shapes of the released backbones. A tower that does not appear here, such as the action
# tower, is built to match the towers it reads.
KNOWN_SHAPES = {
    "wan22": TowerShape(30, 24),
    "cosmos25": TowerShape(28, 16),
    "qwen3_vl": TowerShape(28, 16),
}


@dataclass(frozen=True)
class TowerSpec:
    name: str
    kind: str
    train: str
    context: tuple[ContextRead, ...]
    options: Mapping[str, Any]

    @classmethod
    def parse(cls, name: str, raw: Mapping[str, Any]) -> "TowerSpec":
        where = f"towers.{name}"
        if "kind" not in raw:
            raise SpecError([Problem("format", f"{where}: a tower needs `kind`")])
        train = raw.get("train", "full")
        if train not in TRAIN_POLICIES:
            raise SpecError([Problem("format", f"{where}: unknown train policy {train!r}")])
        context = tuple(ContextRead.parse(item, f"{where}.context") for item in raw.get("context", ()))
        options = {key: value for key, value in raw.items() if key not in ("kind", "train", "context")}
        return cls(name, str(raw["kind"]), train, context, options)

    @property
    def shape(self) -> Optional[TowerShape]:
        layers, heads = self.options.get("num_layers"), self.options.get("num_heads")
        if layers is not None and heads is not None:
            return TowerShape(int(layers), int(heads))
        return KNOWN_SHAPES.get(self.kind)


@dataclass(frozen=True)
class StreamSpec:
    """One kind of token. A stream without a tower is a fixed sequence, such as text."""

    name: str
    tower: Optional[str]
    data: tuple[str, ...]
    encoder: Optional[str]

    @classmethod
    def parse(cls, name: str, raw: Mapping[str, Any]) -> "StreamSpec":
        _only_keys(raw, ("tower", "data", "encoder"), f"streams.{name}")
        tower = raw.get("tower")
        return cls(name, None if tower is None else str(tower), _names(raw.get("data")), raw.get("encoder"))


@dataclass(frozen=True)
class TrainMode:
    """One way of training. Streams listed in ``noisy`` get noise, the rest stay clean."""

    name: str
    noisy: tuple[str, ...]
    weight: float
    tie: tuple[tuple[str, ...], ...]
    absent: tuple[str, ...]
    sometimes_noisy: Mapping[str, float]
    reads: Mapping[str, tuple[Read, ...]]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "TrainMode":
        name = str(raw.get("name", ""))
        where = f"train.{name}"
        _only_keys(raw, ("name", "noisy", "weight", "tie", "absent", "sometimes_noisy", "reads"), where)
        if not name:
            raise SpecError([Problem("format", "train: every training mode needs a `name`")])
        return cls(
            name=name,
            noisy=_names(raw.get("noisy")),
            weight=float(raw.get("weight", 1.0)),
            tie=_groups(raw.get("tie")),
            absent=_names(raw.get("absent")),
            sometimes_noisy={str(key): float(value) for key, value in (raw.get("sometimes_noisy") or {}).items()},
            reads=_parse_rows(raw.get("reads"), f"{where}.reads"),
        )

    def state(self, stream: str) -> str:
        if stream in self.absent:
            return ABSENT
        if stream in self.noisy:
            return NOISY
        if stream in self.sometimes_noisy:
            return EITHER
        return CLEAN


@dataclass(frozen=True)
class Stage:
    """One step of an inference program.

    ``denoise`` runs the solver on those streams. ``hold`` keeps a stream at full noise
    without denoising it. ``use`` feeds a stream with what an earlier stage produced.
    ``cut`` removes reads, written ``"reader<-source"``, to test what a read contributes.
    """

    denoise: tuple[str, ...]
    steps: int
    absent: tuple[str, ...]
    hold: tuple[str, ...]
    tie: tuple[tuple[str, ...], ...]
    use: Mapping[str, str]
    reads: Mapping[str, tuple[Read, ...]]
    cut: tuple[str, ...]

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "Stage":
        _only_keys(raw, ("denoise", "steps", "absent", "hold", "tie", "use", "reads", "cut"), where)
        return cls(
            denoise=_names(raw.get("denoise")),
            steps=int(raw.get("steps", 0)),
            absent=_names(raw.get("absent")),
            hold=_names(raw.get("hold")),
            tie=_groups(raw.get("tie")),
            use={str(key): str(value) for key, value in (raw.get("use") or {}).items()},
            reads=_parse_rows(raw.get("reads"), f"{where}.reads"),
            cut=_names(raw.get("cut")),
        )

    def state(self, stream: str) -> str:
        if stream in self.absent:
            return ABSENT
        if stream in self.denoise:
            return NOISY
        if stream in self.hold:
            return HOLD
        return CLEAN

    def cut_edges(self) -> tuple[tuple[str, str], ...]:
        edges = []
        for item in self.cut:
            reader, separator, source = item.partition("<-")
            edges.append((reader.strip(), source.strip()) if separator else (item, ""))
        return tuple(edges)


@dataclass(frozen=True)
class LossSpec:
    stream: str
    type: str
    weight: float
    training_only: bool
    options: Mapping[str, Any]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "LossSpec":
        if "stream" not in raw or "type" not in raw:
            raise SpecError([Problem("format", "losses: a loss needs `stream` and `type`")])
        if raw["type"] not in LOSS_TYPES:
            raise SpecError([Problem("format", f"losses: unknown type {raw['type']!r}")])
        options = {
            key: value for key, value in raw.items() if key not in ("stream", "type", "weight", "training_only")
        }
        return cls(
            stream=str(raw["stream"]),
            type=raw["type"],
            weight=float(raw.get("weight", 1.0)),
            training_only=bool(raw.get("training_only", False)),
            options=options,
        )


def _agrees(inference_state: str, training_state: str) -> bool:
    """Whether a stream at inference is in a state training has shown the model."""
    if training_state == EITHER:
        return inference_state in (CLEAN, NOISY, HOLD)
    if training_state == NOISY:
        return inference_state in (NOISY, HOLD)
    return inference_state == training_state


@dataclass(frozen=True)
class ModelSpec:
    towers: Mapping[str, TowerSpec]
    streams: Mapping[str, StreamSpec]
    reads: Mapping[str, tuple[Read, ...]]
    train: tuple[TrainMode, ...]
    infer: Mapping[str, tuple[Stage, ...]]
    losses: tuple[LossSpec, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelSpec":
        """Build a spec from plain data, without checking that it makes sense."""
        _only_keys(raw, ("towers", "streams", "reads", "train", "infer", "losses"), "model")
        missing = [key for key in ("towers", "streams", "reads", "train", "infer", "losses") if key not in raw]
        if missing:
            raise SpecError([Problem("format", f"model: missing sections {missing}")])
        return cls(
            towers={str(name): TowerSpec.parse(str(name), value) for name, value in raw["towers"].items()},
            streams={str(name): StreamSpec.parse(str(name), value) for name, value in raw["streams"].items()},
            reads=_parse_rows(raw["reads"], "reads"),
            train=tuple(TrainMode.parse(item) for item in raw["train"]),
            infer={
                str(name): tuple(Stage.parse(item, f"infer.{name}") for item in stages)
                for name, stages in raw["infer"].items()
            },
            losses=tuple(LossSpec.parse(item) for item in raw["losses"]),
        )

    @classmethod
    def load(cls, path: "str | Path") -> "ModelSpec":
        """Read a YAML file and raise ``SpecError`` if anything is wrong with it."""
        import yaml

        spec = cls.from_dict(yaml.safe_load(Path(path).read_text()))
        spec.validate()
        return spec

    def validate(self) -> None:
        problems = self.check()
        if problems:
            raise SpecError(problems)

    @property
    def token_streams(self) -> tuple[str, ...]:
        """Streams that run through a tower, in sequence order: tower by tower."""
        return tuple(
            stream.name
            for tower in self.towers
            for stream in self.streams.values()
            if stream.tower == tower
        )

    def rows(self, overrides: Optional[Mapping[str, tuple[Read, ...]]] = None) -> dict[str, tuple[Read, ...]]:
        rows = dict(self.reads)
        rows.update(overrides or {})
        return rows

    def stages(self) -> Iterable[tuple[str, int, Stage]]:
        for program, stages in self.infer.items():
            for index, stage in enumerate(stages):
                yield program, index, stage

    def check(self) -> list[Problem]:
        """Everything wrong with this configuration, without building a model."""
        problems = self._check_names() or self._check_rows()
        if problems:
            return problems  # the checks below assume every name resolves and every stream has a row
        for check in (
            self._check_shapes,
            self._check_modes,
            self._check_losses,
            self._check_stages,
        ):
            problems.extend(check())
        return problems

    def _check_names(self) -> list[Problem]:
        problems = []
        token_streams = set(self.token_streams)

        def need_stream(name: str, where: str) -> None:
            if name not in token_streams:
                problems.append(Problem("name", f"{where}: {name!r} is not a stream with a tower"))

        def need_row_names(rows: Mapping[str, tuple[Read, ...]], where: str) -> None:
            for reader, row in rows.items():
                need_stream(reader, f"{where} row")
                seen = [read.source for read in row]
                if len(seen) != len(set(seen)):
                    problems.append(Problem("name", f"{where}: row {reader} reads a stream twice"))
                for source in seen:
                    need_stream(source, f"{where} row {reader}")

        for stream in self.streams.values():
            if stream.tower is not None and stream.tower not in self.towers:
                problems.append(Problem("name", f"streams: {stream.name} uses unknown tower {stream.tower!r}"))
        for tower in self.towers.values():
            for read in tower.context:
                if read.source in self.towers:
                    shape = self.towers[read.source].shape
                    if read.layer is None:
                        problems.append(Problem("name", f"towers.{tower.name}: reading tower {read.source} needs `layer`"))
                    elif shape is not None and not 0 <= read.layer < shape.layers:
                        problems.append(Problem("name", f"towers.{tower.name}: layer {read.layer} is outside {read.source}"))
                elif self.streams.get(read.source, StreamSpec("", "tower", (), None)).tower is not None:
                    problems.append(Problem("name", f"towers.{tower.name}: unknown context {read.source!r}"))
        need_row_names(self.reads, "reads")
        for mode in self.train:
            where = f"train.{mode.name}"
            need_row_names(mode.reads, where)
            for stream in (*mode.noisy, *mode.absent, *mode.sometimes_noisy, *(s for group in mode.tie for s in group)):
                need_stream(stream, where)
        for program, index, stage in self.stages():
            where = f"infer.{program}[{index}]"
            need_row_names(stage.reads, where)
            for stream in (*stage.denoise, *stage.absent, *stage.hold, *stage.use, *stage.use.values()):
                need_stream(stream, where)
            for stream in (s for group in stage.tie for s in group):
                need_stream(stream, where)
        for loss in self.losses:
            need_stream(loss.stream, "losses")
        return problems

    def _check_rows(self) -> list[Problem]:
        return [
            Problem("row", f"reads: stream {stream} has no read row")
            for stream in self.token_streams
            if stream not in self.reads
        ]

    def _tables(self) -> Iterable[tuple[str, Mapping[str, tuple[Read, ...]]]]:
        yield "reads", self.reads
        for mode in self.train:
            yield f"train.{mode.name}", self.rows(mode.reads)
        for program, index, stage in self.stages():
            yield f"infer.{program}[{index}]", self.rows(stage.reads)

    def _check_shapes(self) -> list[Problem]:
        problems = []
        for where, rows in self._tables():
            for reader, row in rows.items():
                for read in row:
                    first = self.towers[self.streams[reader].tower]
                    second = self.towers[self.streams[read.source].tower]
                    if first.name == second.name or first.shape is None or second.shape is None:
                        continue
                    if first.shape != second.shape:
                        problems.append(
                            Problem(
                                "shape",
                                f"{where}: {reader} reads {read.source} across towers of different shape: "
                                f"{first.name} {first.shape} and {second.name} {second.shape}",
                            )
                        )
        return problems

    def _check_modes(self) -> list[Problem]:
        problems = []
        if not self.train:
            problems.append(Problem("mode", "train: at least one training mode is needed"))
        names = [mode.name for mode in self.train]
        if len(names) != len(set(names)):
            problems.append(Problem("mode", "train: training mode names must be unique"))
        for mode in self.train:
            where = f"train.{mode.name}"
            if mode.weight <= 0:
                problems.append(Problem("mode", f"{where}: weight must be positive"))
            if set(mode.noisy) & set(mode.absent):
                problems.append(Problem("mode", f"{where}: a stream cannot be noisy and absent"))
            for stream, probability in mode.sometimes_noisy.items():
                if not 0.0 <= probability <= 1.0:
                    problems.append(Problem("mode", f"{where}: probability for {stream} is outside [0, 1]"))
            for group in mode.tie:
                if not set(group) <= set(mode.noisy):
                    problems.append(Problem("mode", f"{where}: tied streams must all be noisy: {list(group)}"))
            rows = self.rows(mode.reads)
            for stream in self.token_streams:
                if mode.state(stream) != ABSENT and stream in rows:
                    live = [read for read in rows[stream] if mode.state(read.source) != ABSENT]
                    if not live:
                        problems.append(Problem("mode", f"{where}: {stream} reads nothing"))
        return problems

    def _check_losses(self) -> list[Problem]:
        problems = []
        noisy_in_training = set()
        for mode in self.train:
            noisy_in_training |= set(mode.noisy) | set(mode.sometimes_noisy)
        for loss in self.losses:
            if loss.type in NOISE_LOSSES and loss.stream not in noisy_in_training:
                problems.append(
                    Problem("loss", f"losses: {loss.type} loss on {loss.stream}, which no training mode adds noise to")
                )
            if loss.type == FEATURE and "target" not in loss.options:
                problems.append(Problem("loss", f"losses: feature loss on {loss.stream} needs `target`"))
        return problems

    def _check_stages(self) -> list[Problem]:
        problems = []
        if not self.infer:
            problems.append(Problem("stage", "infer: at least one inference program is needed"))
        noisy_in_training = set()
        for mode in self.train:
            noisy_in_training |= set(mode.noisy) | set(mode.sometimes_noisy)

        denoised_before: dict[str, set[str]] = {}
        for program, index, stage in self.stages():
            where = f"infer.{program}[{index}]"
            done = denoised_before.setdefault(program, set())
            if not stage.denoise:
                problems.append(Problem("stage", f"{where}: a stage must denoise something"))
            if stage.steps <= 0:
                problems.append(Problem("stage", f"{where}: `steps` must be positive"))
            if len({*stage.denoise}) + len({*stage.hold}) + len({*stage.absent}) != len(
                {*stage.denoise, *stage.hold, *stage.absent}
            ):
                problems.append(Problem("stage", f"{where}: `denoise`, `hold` and `absent` must not overlap"))
            for group in stage.tie:
                if not set(group) <= set(stage.denoise):
                    problems.append(Problem("stage", f"{where}: tied streams must all be denoised: {list(group)}"))
            for target, source in stage.use.items():
                if source not in done:
                    problems.append(Problem("stage", f"{where}: `use` reads {source}, which no earlier stage produced"))
                if stage.state(target) != CLEAN:
                    problems.append(Problem("stage", f"{where}: `use` target {target} must be clean"))
            for stream in stage.denoise:
                if stream not in noisy_in_training:
                    problems.append(Problem("denoise", f"{where}: denoises {stream}, which no training mode made noisy"))
            problems.extend(self._check_stage_against_training(stage, where))
            done |= set(stage.denoise)
        return problems

    def _check_stage_against_training(self, stage: Stage, where: str) -> list[Problem]:
        rows = self.rows(stage.reads)
        live = [stream for stream in self.token_streams if stage.state(stream) != ABSENT]

        problems = []
        for reader, source in stage.cut_edges():
            if reader not in rows or source not in {read.source for read in rows[reader]}:
                problems.append(Problem("stage", f"{where}: cut {reader}<-{source} is not a read"))
        for stream in live:
            for read in rows[stream]:
                if stage.state(read.source) == ABSENT:
                    problems.append(Problem("unseen", f"{where}: {stream} reads {read.source}, which is absent here"))
        if problems:
            return problems

        matches = []
        for mode in self.train:
            mode_rows = self.rows(mode.reads)
            same = all(
                {read.source for read in mode_rows[stream]} == {read.source for read in rows[stream]}
                for stream in live
            )
            if same:
                matches.append(mode)
        if not matches:
            return [Problem("table", f"{where}: uses a read table no training mode used")]
        if not any(all(_agrees(stage.state(s), mode.state(s)) for s in live) for mode in matches):
            return [Problem("unseen", f"{where}: puts streams in a state no matching training mode used")]
        return []
