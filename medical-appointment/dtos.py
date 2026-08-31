"""The request and response models for the medical-appointment case.

These mirror the models the evaluation service uses on its side. Keep them as
they are: the field names below are the wire protocol, and a response the
service cannot parse is scored as a wrong answer.

The service is permissive about what it accepts back — it will coerce ``1``,
``"yes"`` and ``"true"`` into ``True``, and it ignores extra keys rather than
rejecting them. Do not rely on that. Send real JSON booleans and nothing else.
"""

from typing import List

from pydantic import BaseModel, Field


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
    """Your answers. One boolean per question, in the order you received them."""

    answers: List[bool] = Field(
        description='True for yes, False for no. Exactly as many answers as '
                    'there were questions, in the same order.',
    )
