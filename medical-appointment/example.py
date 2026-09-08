"""The baseline. This is the file to replace.

It answers ``True`` to everything and points at nothing, which scores the floor
and nothing more. It is here to prove the plumbing — that the audio arrives
intact and that your server speaks the protocol — not to compete.

Note how weak that floor now is. Answering yes to everything still gets half
the questions right, but it finds none of the evidence, and evidence is the
larger half of the score. The sketch under the dummy model shows where a real
system goes.
"""

import logging
from typing import Optional, Tuple

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import Span, audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    The whole conversation and all of its questions arrive together, so the
    expensive half — transcription — is paid once here and shared by every
    answer below.
    """
    audio_bytes = decode_audio(request.audio_base64)

    duration = audio_duration_seconds(audio_bytes)
    logger.info(
        '%s (%.1f s, %.1f MB): %d questions',
        request.audio_filename,
        duration if duration is not None else float('nan'),
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    # Never let this raise. An exception means no response, and no response
    # means every question about this conversation is scored wrong — ten marks,
    # not one. A guess is worth half a mark on average; an error is worth
    # nothing.
    answers = []
    evidence_start = []
    evidence_end = []

    for question in request.questions:
        try:
            answer, span = answer_question(
                audio_bytes, request.audio_filename, question
            )
        except Exception:
            logger.exception('Falling back to a guess for: %s', question)
            answer, span = True, None

        answers.append(answer)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )


### DUMMY MODEL ###

def answer_question(
    audio_bytes: bytes,
    audio_filename: str,
    question: str,
) -> Tuple[bool, Optional[Span]]:
    """Always says yes, and never says where.

    Both splits are exactly balanced between yes and no, so the answer half of
    this scores 0.500: every ``positive`` question right, every
    ``hard_negative`` and ``off_topic`` question wrong. The evidence half scores
    0.000, because ``None`` means "nothing to point at" and every annotated yes
    question is therefore missed. Run ``local_evaluator.py`` and read the
    per-type breakdown and the evidence block — that shape is the problem you
    are solving.

    Replace this. The shape of a real answer is roughly:

        def predict(request):
            # The expensive half, paid once per request rather than once per
            # question. Ten questions share this transcript.
            segments = transcribe(decode_audio(request.audio_base64))

            answers, starts, ends = [], [], []

            for question in request.questions:
                answer, span = answer_from_transcript(segments, question)
                answers.append(answer)
                starts.append(span[0] if span else None)
                ends.append(span[1] if span else None)

            return ASRQuestionResponseDto(
                answers=answers, evidence_start=starts, evidence_end=ends,
            )

    where ``transcribe`` is a local ASR model **that returns timestamps** — the
    span you send back is the start and end of the segment you read the answer
    off, so word- or segment-level timing is not an optional extra here. Both
    halves must run without calling a cloud API; see the Rules section of the
    README.

    Two things to watch while you work:

    Return the passage, not the clip. A span covering the whole conversation
    overlaps every annotation and scores a temporal IoU near zero against all
    of them.

    Watch the ``hard_negative`` questions. They are near-misses on dose, drug
    and entity — "0.15 mg" against a transcript that says "0.3 mg" — so
    anything that answers from topical overlap alone stays at the floor no
    matter how good the transcript is.
    """
    return True, None
