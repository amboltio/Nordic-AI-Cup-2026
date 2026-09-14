"""Wire protocol for the drone flyby use case.

These models mirror the evaluation service exactly. Keep them in sync: the
evaluator validates your response with the same rules, and a response that
fails validation is discarded together with every detection in it.

The asymmetry to internalise first is the protocol's, not the models':

* What you receive is **crop-local**. A 960x540 image of wherever the camera is
  pointed, plus the ``source_region_xyxy`` that says which part of the source
  frame it covers.
* What you send back is **frame-global**. ``annotations`` is your best
  prediction for the entire current source frame, with every box normalized
  against ``original_width`` and ``original_height``. Objects outside the
  current crop are valid answers.

Two smaller asymmetries in the models themselves:

* The request models are permissive. If the evaluator ever gains a field, your
  server keeps running instead of rejecting the frame.
* The response models are strict, because the evaluator's are. Unknown keys,
  numeric strings and floats where integers are expected are all rejected
  there, so it is better to find out here.
"""

from typing import List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    conint,
    conlist,
    field_validator,
    model_validator,
)


# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

# Every camera center and every scored box lives in this coordinate system.
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

# The resolution level decides how much of the source frame the camera covers.
SOURCE_REGION_SIZES = {
    0: (IMAGE_WIDTH, IMAGE_HEIGHT),          # 3840x2160, the complete frame
    1: (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2),  # 1920x1080
    2: (IMAGE_WIDTH // 4, IMAGE_HEIGHT // 4),  # 960x540, sent at native size
}

# Whatever the level, the transmitted image is always this size.
TRANSMITTED_VIEW_SIZE = (IMAGE_WIDTH // 4, IMAGE_HEIGHT // 4)  # 960x540
FULL_FRAME_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)      # (1920, 1080)

# The zoom levels reachable in one response, keyed by the current level.
ALLOWED_RESOLUTION_LEVELS = {
    0: (0, 1),
    1: (0, 1, 2),
    2: (1, 2),
}

# Maximum Euclidean center movement per accepted response, by current level.
MAXIMUM_CENTER_DELTA_PIXELS = {
    0: 2203.0,
    1: 1102.0,
    2: 551.0,
}

# This order is the evaluator's COCO category order. Do not reorder it: the
# local scorer builds category IDs from it and would disagree with the server.
OBJECT_CLASSES = (
    'hangar',
    'helicopter',
    'jet_plane',
    'large_launcher',
    'large_tower',
    'medium_launcher',
    'medium_plane',
    'mine_roller',
    'small_launcher',
    'small_plane',
    'small_tower',
    'ta-ta',
    'tank',
    'car',
    'condor',
    'jammer',
    'spacecraft',
)


Number = Union[StrictInt, StrictFloat]
NormalizedBoundingBox = conlist(Number, min_length=4, max_length=4)
PixelBoundingBox = conlist(StrictInt, min_length=4, max_length=4)
FrameNumber = conint(strict=True, ge=0)
ResolutionLevel = conint(strict=True, ge=0, le=2)


# --------------------------------------------------------------------------- #
# What the evaluator sends you
# --------------------------------------------------------------------------- #

class DroneFlybyViewDto(BaseModel):
    """The transmitted image and the region of the source frame it covers."""

    resolution_level: ResolutionLevel
    center_x: StrictInt
    center_y: StrictInt
    view_id: str
    # A 960x540 PNG, Base64 encoded. There is no data: URI prefix.
    image: str
    image_media_type: str
    # Always 960 and 540. They describe the transmitted image, never the size
    # of the source region it represents.
    width: StrictInt
    height: StrictInt
    # [x1, y1, x2, y2] in source pixels: where this view was taken from. Use it
    # to lift detections you make on this image into the frame-global
    # coordinates a response uses. The evaluator never applies it to your
    # answer; see utils.view_bbox_to_global.
    source_region_xyxy: PixelBoundingBox

    model_config = ConfigDict(extra='ignore')


class CameraLevelBoundsDto(BaseModel):
    """Valid target centers for one resolution level."""

    resolution_level: ResolutionLevel
    width: StrictInt
    height: StrictInt
    minimum_center_x: StrictInt
    maximum_center_x: StrictInt
    minimum_center_y: StrictInt
    maximum_center_y: StrictInt

    model_config = ConfigDict(extra='ignore')


class CameraConstraintsDto(BaseModel):
    """The movement rules that apply to your next camera request.

    Read these from the request rather than hardcoding them. They already
    account for the camera's current level, so honouring them is enough to
    keep every command legal.
    """

    maximum_center_delta: float
    allowed_resolution_levels: List[ResolutionLevel]
    center_bounds: List[CameraLevelBoundsDto]
    full_view_reset_exempt_from_delta: StrictBool

    model_config = ConfigDict(extra='ignore')

    def bounds_for_level(self, resolution_level: int) -> Optional[CameraLevelBoundsDto]:
        """Return the center bounds for a level, or None if it is not allowed."""
        for bounds in self.center_bounds:
            if bounds.resolution_level == resolution_level:
                return bounds
        return None


class CameraCommandFeedbackDto(BaseModel):
    """Why your previous camera request was not applied.

    This repeats on every following frame until you send a usable command, so
    it still reaches you when the frame right after the rejection is skipped.
    """

    frame: FrameNumber
    requested_view: 'RequestedViewDto'
    reason: str

    model_config = ConfigDict(extra='ignore')


class DroneFlybyPredictRequestDto(BaseModel):
    """One emitted frame: its view and the camera rules that currently apply."""

    sequence_id: str
    # The source frame number. Your predictions are scored against it, and you
    # must echo it back unchanged.
    frame: FrameNumber
    # Position in the sequence. Gaps here are the frames you were too slow for.
    frame_index: FrameNumber
    # Echo this back unchanged as well.
    request_id: str
    # How often a frame is emitted, in milliseconds (333 at the configured 3 fps).
    frame_interval_ms: conint(strict=True, ge=0)
    # The full budget for this one request, measured from the POST (~3333 ms).
    response_timeout_ms: conint(strict=True, ge=0)
    original_width: StrictInt
    original_height: StrictInt
    view: DroneFlybyViewDto
    camera_constraints: CameraConstraintsDto
    camera_command_feedback: Optional[CameraCommandFeedbackDto] = None

    model_config = ConfigDict(extra='ignore')


# --------------------------------------------------------------------------- #
# What you send back
# --------------------------------------------------------------------------- #

class DroneFlybyPredictionDto(BaseModel):
    """One detection anywhere in the source frame of the matching request.

    Not "in the image you received": a detection you are still tracking from an
    earlier frame is a perfectly good annotation even when the camera is now
    pointed somewhere else entirely.
    """

    # Must be one of OBJECT_CLASSES, case-sensitive.
    object_id: str
    # [x1, y1, x2, y2] normalized against original_width and original_height,
    # so a box at source x = 1920 has x = 0.5 at every resolution level and
    # wherever the camera happens to be.
    bbox: NormalizedBoundingBox
    confidence: Number

    model_config = ConfigDict(extra='forbid')

    @field_validator('object_id')
    @classmethod
    def validate_object_id(cls, value: str) -> str:
        if value not in OBJECT_CLASSES:
            raise ValueError(
                f'object_id must be one of: {", ".join(OBJECT_CLASSES)}'
            )
        return value

    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, value):
        import math

        x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
        if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
            raise ValueError('bbox coordinates must be finite')
        # Note the strict inequalities: a zero-area box is rejected, and a
        # rejected box takes the whole response down with it.
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError(
                'bbox must use source-frame normalized coordinates in '
                '[x1, y1, x2, y2] order'
            )
        return value

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, value):
        import math

        confidence = float(value)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be between 0 and 1')
        return value


class RequestedViewDto(BaseModel):
    """The absolute camera position you want for the next frame.

    All three fields are strict integers. Sending 2500.0 instead of 2500 is a
    validation error on the evaluator, so keep your arithmetic in ints.

    An unreachable view is ignored rather than fatal: the camera stays where it
    is, the detections in the same response are still scored, and the reason
    comes back in ``camera_command_feedback``.
    """

    resolution_level: StrictInt
    center_x: StrictInt
    center_y: StrictInt

    model_config = ConfigDict(extra='forbid', frozen=True)


class DroneFlybyPredictResponseDto(BaseModel):
    """Whole-frame detections, plus where the camera should look next."""

    # Both of these must match the request exactly.
    request_id: str
    frame: FrameNumber
    # Your best prediction for the entire source frame, covering objects
    # inside and outside the crop that arrived with the request. At most 500.
    annotations: conlist(DroneFlybyPredictionDto, max_length=500)
    # Omit or set to null to leave the camera where it is.
    requested_view: Optional[RequestedViewDto] = None

    model_config = ConfigDict(extra='forbid')


CameraCommandFeedbackDto.model_rebuild()


class CameraViewSelectionDto(BaseModel):
    """A validated camera position, used by the local evaluator.

    This is the model the evaluator promotes your ``RequestedViewDto`` into
    once it has checked the bounds. It is here so the local harness can apply
    the same rules the server does.
    """

    resolution_level: ResolutionLevel
    center_x: conint(strict=True, ge=0, le=IMAGE_WIDTH)
    center_y: conint(strict=True, ge=0, le=IMAGE_HEIGHT)

    model_config = ConfigDict(extra='forbid', frozen=True)

    @model_validator(mode='after')
    def validate_center_for_level(self):
        width, height = SOURCE_REGION_SIZES[self.resolution_level]
        minimum_x, maximum_x = width // 2, IMAGE_WIDTH - width // 2
        minimum_y, maximum_y = height // 2, IMAGE_HEIGHT - height // 2
        if not minimum_x <= self.center_x <= maximum_x:
            raise ValueError(
                f'resolution level {self.resolution_level} requires center_x '
                f'in [{minimum_x}, {maximum_x}]'
            )
        if not minimum_y <= self.center_y <= maximum_y:
            raise ValueError(
                f'resolution level {self.resolution_level} requires center_y '
                f'in [{minimum_y}, {maximum_y}]'
            )
        return self
