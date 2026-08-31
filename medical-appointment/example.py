"""The baseline. This is the file to replace.

It answers ``True`` to everything, which scores exactly the floor and nothing
more. It is here to prove the plumbing — that the audio arrives intact and that
your server speaks the protocol — not to compete.

The sketch under the dummy model shows where a real system goes.
"""

import logging

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

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

    for question in request.questions:
        try:
            answers.append(
                answer_question(audio_bytes, request.audio_filename, question)
            )
        except Exception:
            logger.exception('Falling back to a guess for: %s', question)
            answers.append(True)

    return ASRQuestionResponseDto(answers=answers)


### DUMMY MODEL ###

def answer_question(audio_bytes: bytes, audio_filename: str, question: str) -> bool:
    """Always says yes.

    Both splits are exactly balanced between yes and no, so this scores the
    floor: every ``positive`` question right, every ``hard_negative`` and
    ``off_topic`` question wrong. Run ``local_evaluator.py`` and read the
    per-type breakdown — that shape is the problem you are solving.

    Replace this. The shape of a real answer is roughly:

        def predict(request):
            # The expensive half, paid once per request rather than once per
            # question. Ten questions share this transcript.
            transcript = transcribe(decode_audio(request.audio_base64))

            return ASRQuestionResponseDto(answers=[
                answer_from_transcript(transcript, question)
                for question in request.questions
            ])

    where ``transcribe`` is a local ASR model and ``answer_from_transcript`` is
    whatever decides the question — a local LLM, an extractive QA model, or
    something you build yourself. Both halves must run without calling a cloud
    API; see the Rules section of the README.

    Watch the ``hard_negative`` questions while you work. They are near-misses
    on dose, drug and entity — "0.15 mg" against a transcript that says
    "0.3 mg" — so anything that answers from topical overlap alone stays at the
    floor no matter how good the transcript is.
    """
    return True
