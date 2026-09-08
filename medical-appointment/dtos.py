"""The request and response models for the medical-appointment case.

These mirror the models the evaluation service uses on its side. Keep them as
they are: the field names below are the wire protocol, and a response the
service cannot parse is scored as a wrong answer.

A response carries two things now: the yes/no answer, and — for every yes — the
stretch of audio the answer was read off. The three lists are read positionally
and must be the same length, so a body that gets that wrong cannot be scored at
all and loses every question about the conversation, not just its evidence.

The service is permissive about what it accepts back — it will coerce ``1``,
``"yes"`` and ``"true"`` into ``True``, and it ignores extra keys rather than
rejecting them. Do not rely on that. Send real JSON booleans, real JSON numbers
and real JSON nulls.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# What the evaluator sends you
# --------------------------------------------------------------------------- #

class ASRQuestionRequestDto(BaseModel):
    """One conversation and every question asked about it.

    You get one request per conversation, carrying all ten of its questions, so
    you transcribe the audio once and then answer ten times. There is no second
    request for the same conversation and nothing to cache between requests.
    """

    audio_base64: str = Field(
        description='The raw MP3 bytes, base64 encoded. No "data:" URI prefix — '
                    'pass it straight to base64.b64decode.',
    )
    audio_filename: str = Field(
        description='e.g. "conversation_sample_11.mp3". Identifies the '
                    'conversation; useful in your logs.',
    )
    questions: List[str] = Field(
        description='The English yes/no questions about this conversation. Ten '
                    'of them, during validation and evaluation alike.',
    )


# --------------------------------------------------------------------------- #
# What you send back
# --------------------------------------------------------------------------- #

class ASRQuestionResponseDto(BaseModel):
    """Your answers and the evidence behind them.

    One entry per question in each of the three lists, in the order the
    questions arrived. A ``yes`` answer has to point at the passage it was read
    from — the start and end timestamp, in seconds from the beginning of the
    audio. A ``no`` answer has nothing to point at, so both of its timestamps
    are ``None``.
    """

    answers: List[bool] = Field(
        description='True for yes, False for no. Exactly as many answers as '
                    'there were questions, in the same order.',
    )
    evidence_start: List[Optional[float]] = Field(
        description='Where the supporting passage starts, in seconds from the '
                    'beginning of the audio. None for a no answer.',
    )
    evidence_end: List[Optional[float]] = Field(
        description='Where the supporting passage ends, in seconds. None for a '
                    'no answer.',
    )

    @model_validator(mode='after')
    def evidence_matches_answers(self) -> 'ASRQuestionResponseDto':
        """The three lists are read positionally, so they have to line up.

        A response that does not carry one evidence slot per answer cannot be
        scored at all — the service fails the whole conversation rather than
        guessing which answer a timestamp belongs to.
        """
        if (
            len(self.evidence_start) != len(self.answers)
            or len(self.evidence_end) != len(self.answers)
        ):
            raise ValueError(
                f'evidence_start and evidence_end must both hold one value per '
                f'answer: got {len(self.answers)} answers, '
                f'{len(self.evidence_start)} evidence_start and '
                f'{len(self.evidence_end)} evidence_end.'
            )

        return self
