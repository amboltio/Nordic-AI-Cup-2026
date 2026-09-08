"""Helpers for the medical-appointment case.

Nothing here is required by the protocol. It is the plumbing you would
otherwise write yourself: getting the audio off the wire, checking that your
answers line up with the questions before they leave, and loading the supplied
conversations.

There is deliberately no audio dependency. ``audio_duration_seconds`` reads the
MP3 frame header directly, so ``pip install -r requirements.txt`` does not drag
in a decoding stack that would fight whatever ASR you end up choosing.

``gold_evidence`` and ``evidence_iou`` are the other half of the case: the
supplied questions carry the span of audio the answer was read off, and the
score is part accuracy, part how well your spans line up with those.
"""

import base64
import collections
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dtos import ASRQuestionResponseDto

DATA_DIRECTORY = Path(__file__).resolve().parent / 'data'
AUDIO_DIRECTORY = DATA_DIRECTORY / 'audio'
QUESTIONS_CSV = DATA_DIRECTORY / 'question_train.csv'

# A span of audio, in seconds from the start of the conversation.
Span = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Audio on the wire
# --------------------------------------------------------------------------- #

def decode_audio(audio_base64: str) -> bytes:
    """Turn ``request.audio_base64`` back into MP3 bytes.

    The field is plain base64 of the file, with no ``data:audio/mpeg;base64,``
    prefix, so this is the whole of it.
    """
    return base64.b64decode(audio_base64)


def encode_audio(audio_bytes: bytes) -> str:
    """The inverse. Used by ``local_evaluator.py`` to build requests."""
    return base64.b64encode(audio_bytes).decode('utf-8')


# Bitrate table for MPEG-1 Layer III, indexed by the 4-bit field in the header.
_BITRATES_KBPS = (
    None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None,
)
_SAMPLE_RATES_HZ = {0: 44100, 1: 48000, 2: 32000}


def audio_duration_seconds(audio_bytes: bytes) -> Optional[float]:
    """Length of the clip in seconds, or ``None`` if the header cannot be read.

    Assumes constant bitrate, which every supplied conversation is (MPEG-1
    Layer III, 128 kbps, 44.1 kHz, mono). Good enough to sanity-check that the
    bytes really did survive the trip; not a substitute for decoding.
    """
    offset = 0

    # An ID3v2 tag, if present, sits in front of the first frame. Its size is
    # four seven-bit bytes.
    if audio_bytes[:3] == b'ID3' and len(audio_bytes) >= 10:
        size = 0
        for byte in audio_bytes[6:10]:
            size = (size << 7) | (byte & 0x7F)
        offset = 10 + size

    for i in range(offset, min(len(audio_bytes) - 4, offset + 200_000)):
        if audio_bytes[i] != 0xFF or (audio_bytes[i + 1] & 0xE0) != 0xE0:
            continue

        version = (audio_bytes[i + 1] >> 3) & 0b11    # 3 == MPEG-1
        layer = (audio_bytes[i + 1] >> 1) & 0b11      # 1 == Layer III
        bitrate = _BITRATES_KBPS[(audio_bytes[i + 2] >> 4) & 0xF]
        sample_rate = _SAMPLE_RATES_HZ.get((audio_bytes[i + 2] >> 2) & 0b11)

        if version == 3 and layer == 1 and bitrate and sample_rate:
            return len(audio_bytes) * 8 / (bitrate * 1000)

    return None


# --------------------------------------------------------------------------- #
# Evidence and scoring
# --------------------------------------------------------------------------- #
#
# Ported from the evaluation service so a local number means the same thing as a
# real one. If you change anything here you are no longer measuring what you
# will be scored on.

def evidence_interval(start: Any, end: Any) -> Optional[Span]:
    """Read one evidence interval, from the annotations or from a prediction.

    An interval only counts if both timestamps are there, are real numbers and
    are the right way round. Everything else — a missing timestamp, a blank
    annotation cell, ``None``, a NaN, text, an end before its start — is no
    interval at all. Deliberately lenient: a prediction like that scores a
    temporal IoU of 0 rather than raising.
    """
    if start is None or end is None:
        return None

    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        return None

    return start, end


def gold_evidence(row: Dict[str, str]) -> Optional[Span]:
    """The annotated span for one CSV row, or ``None`` if it has none.

    Only the ``positive`` rows carry evidence; a ``hard_negative`` or
    ``off_topic`` row has both cells blank, because a no answer has nothing to
    point at.
    """
    return evidence_interval(row.get('evidence_start'), row.get('evidence_end'))


def temporal_iou(ground_truth: Span, prediction: Optional[Span]) -> float:
    """Overlap between the annotated and the predicted span.

    Measured as a fraction of the stretch the two of them cover together.
    Intervals that do not touch score 0, identical intervals score 1, and no
    prediction at all scores 0.
    """
    if prediction is None:
        return 0.0

    ground_truth_start, ground_truth_end = ground_truth
    prediction_start, prediction_end = prediction

    intersection = max(
        0.0,
        min(ground_truth_end, prediction_end)
        - max(ground_truth_start, prediction_start),
    )
    union = (
        max(ground_truth_end, prediction_end)
        - min(ground_truth_start, prediction_start)
    )

    if union <= 0:
        return 0.0

    return intersection / union


def mean_temporal_iou(
    labels: List[int],
    ground_truths: List[Optional[Span]],
    predictions: List[Optional[Span]],
) -> float:
    """Mean temporal IoU over the questions that have evidence to find.

    The denominator is fixed by the annotations, not by what you answered: every
    question whose annotated answer is yes counts, so one you answered no — or
    returned no usable span for — contributes 0 rather than being left out. No
    questions have nothing to point at and are excluded entirely, so a span
    volunteered alongside a no can neither help nor hurt.
    """
    scores = [
        temporal_iou(ground_truth, prediction)
        for label, ground_truth, prediction
        in zip(labels, ground_truths, predictions)
        if label == 1 and ground_truth is not None
    ]

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Checking your own response
# --------------------------------------------------------------------------- #

def validate_response(
    response: ASRQuestionResponseDto,
    expected_count: int,
) -> None:
    """Raise if the response would not survive the evaluator.

    The length checks are the ones that matter. The service matches answers and
    evidence to questions by position, so a list of the wrong length is not
    partially credited — it cannot be scored at all, and the whole conversation
    is scored wrong. That now includes the two evidence lists: getting one of
    them wrong costs you the answers as well, not just the evidence. Failing
    here, loudly, in your own logs beats losing ten marks silently.
    """
    if not isinstance(response, ASRQuestionResponseDto):
        raise ValueError(
            'predict() must return an ASRQuestionResponseDto, got '
            f'{type(response).__name__}.'
        )

    if not isinstance(response.answers, list):
        raise ValueError(
            f'answers must be a list, got {type(response.answers).__name__}.'
        )

    if len(response.answers) != expected_count:
        raise ValueError(
            f'answers must have one entry per question: expected '
            f'{expected_count}, got {len(response.answers)}.'
        )

    for position, answer in enumerate(response.answers):
        if not isinstance(answer, bool):
            raise ValueError(
                f'answers[{position}] must be a bool, got '
                f'{type(answer).__name__}.'
            )

    for name in ('evidence_start', 'evidence_end'):
        values = getattr(response, name)

        if not isinstance(values, list):
            raise ValueError(
                f'{name} must be a list, got {type(values).__name__}.'
            )

        if len(values) != expected_count:
            raise ValueError(
                f'{name} must have one entry per question: expected '
                f'{expected_count}, got {len(values)}.'
            )

        for position, value in enumerate(values):
            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f'{name}[{position}] must be a number of seconds or None, '
                    f'got {type(value).__name__}.'
                )

            if not math.isfinite(value):
                raise ValueError(
                    f'{name}[{position}] must be a finite number of seconds, '
                    f'got {value!r}.'
                )

    # A span the service cannot read scores zero rather than raising, so this is
    # the last place a reversed or half-filled interval is still cheap to spot.
    for position, (start, end) in enumerate(
        zip(response.evidence_start, response.evidence_end)
    ):
        if (start is None) != (end is None):
            raise ValueError(
                f'evidence_start[{position}] and evidence_end[{position}] must '
                'either both be set or both be None; a half-filled interval '
                'scores nothing.'
            )

        if start is not None and end < start:
            raise ValueError(
                f'evidence_end[{position}] ({end}) is before '
                f'evidence_start[{position}] ({start}); that interval scores '
                'nothing.'
            )


# --------------------------------------------------------------------------- #
# The supplied sample data
# --------------------------------------------------------------------------- #

def load_sample_questions() -> List[Dict[str, str]]:
    """The supplied questions, in the order the evaluator would send them."""
    with open(QUESTIONS_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_sample_audio(audio_filename: str) -> bytes:
    """Raw bytes of one supplied conversation."""
    return (AUDIO_DIRECTORY / audio_filename).read_bytes()


def audio_filename_for_transcript(transcript_id: str) -> str:
    """CSV rows carry `sample_4`; the file on disk is `conversation_sample_4.mp3`."""
    return f'conversation_{transcript_id}.mp3'


def group_questions_by_conversation() -> List[Tuple[str, List[Dict[str, str]]]]:
    """The supplied questions batched the way the evaluator batches them.

    Returns ``(audio_filename, rows)`` pairs, keeping both the order the
    conversations first appear in the CSV and the row order inside each one —
    which is the order the questions arrive in, and therefore the order your
    answers have to come back in.
    """
    groups: Dict[str, List[Dict[str, str]]] = collections.OrderedDict()

    for row in load_sample_questions():
        groups.setdefault(
            audio_filename_for_transcript(row['transcript_id']), []
        ).append(row)

    return list(groups.items())
