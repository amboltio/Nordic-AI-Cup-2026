"""Helpers for the medical-appointment case.

Nothing here is required by the protocol. It is the plumbing you would
otherwise write yourself: getting the audio off the wire, checking that your
answers line up with the questions before they leave, and loading the supplied
conversations.

There is deliberately no audio dependency. ``audio_duration_seconds`` reads the
MP3 frame header directly, so ``pip install -r requirements.txt`` does not drag
in a decoding stack that would fight whatever ASR you end up choosing.
"""

import base64
import collections
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dtos import ASRQuestionResponseDto

DATA_DIRECTORY = Path(__file__).resolve().parent / 'data'
AUDIO_DIRECTORY = DATA_DIRECTORY / 'audio'
QUESTIONS_CSV = DATA_DIRECTORY / 'questions_sample.csv'


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
# Checking your own response
# --------------------------------------------------------------------------- #

def validate_response(
    response: ASRQuestionResponseDto,
    expected_count: int,
) -> None:
    """Raise if the response would not survive the evaluator.

    The length check is the one that matters. The service matches answers to
    questions by position, so a list of the wrong length is not partially
    credited — the whole conversation is scored wrong. Failing here, loudly, in
    your own logs beats losing ten marks silently.
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
