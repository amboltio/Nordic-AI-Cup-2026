"""A baseline that answers the protocol correctly and detects almost nothing.

The point of this file is the plumbing, not the accuracy: it shows you how to
decode a view, turn pixel boxes into the normalized coordinates the evaluator
expects, and drive the camera without ever sending an illegal command. Replace
``detect`` with your model and ``choose_next_view`` with your camera policy.

Run ``python local_evaluator.py`` to see what it scores. It will be close to
zero, which is the honest starting point.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    MAXIMUM_CENTER_DELTA_PIXELS,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_view, decode_view

logger = logging.getLogger(__name__)


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: report detections and pick the next camera position."""
    # The evaluator tells you when it ignored your last camera command. Reading
    # this beats wondering why the camera never moved.
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            'Camera command from frame %s was ignored: %s',
            feedback.frame,
            feedback.reason,
        )

    image = decode_view(request.view)

    # Never let a modelling error cost you the frame. An empty list still
    # scores the frame; an exception loses it and every detection in it.
    try:
        annotations = detect(image)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        annotations = []

    return DroneFlybyPredictResponseDto(
        # These two must come straight back from the request, unchanged.
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )


### DUMMY MODEL ###

# A placeholder class for the proposals below. Anything you report has to be
# one of the names in dtos.OBJECT_CLASSES, spelled exactly.
PLACEHOLDER_CLASS = 'car'

MINIMUM_BOX_PIXELS = 8
MAXIMUM_BOX_PIXELS = 320
MAXIMUM_PROPOSALS = 20


def detect(image: np.ndarray) -> List[DroneFlybyPredictionDto]:
    """Propose boxes around whatever stands out from the ground.

    This is edge detection, not object detection: it has no idea what it is
    looking at, so it labels everything ``car`` with low confidence. It exists
    to show the coordinate conversion on real data. Swap it out.
    """
    height, width = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        longest = max(box_width, box_height)
        if longest < MINIMUM_BOX_PIXELS or longest > MAXIMUM_BOX_PIXELS:
            continue
        # Compactness stands in for "looks like a thing" here.
        area_ratio = cv2.contourArea(contour) / float(box_width * box_height or 1)
        proposals.append((area_ratio, (x, y, box_width, box_height)))

    proposals.sort(key=lambda item: item[0], reverse=True)

    annotations: List[DroneFlybyPredictionDto] = []
    for area_ratio, (x, y, box_width, box_height) in proposals[:MAXIMUM_PROPOSALS]:
        # Boxes leave your model in pixels and have to reach the evaluator
        # normalized to this exact 960x540 view.
        bbox = clip_bbox_to_view(
            (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
        )
        # clip_bbox_to_view returns None when nothing survives clipping. Drop
        # those: one degenerate box invalidates the entire response.
        if bbox is None:
            continue
        annotations.append(
            DroneFlybyPredictionDto(
                object_id=PLACEHOLDER_CLASS,
                bbox=list(bbox),
                confidence=round(min(0.30, 0.05 + 0.25 * area_ratio), 4),
            )
        )
    return annotations


### DUMMY CAMERA POLICY ###

# Where the sweep goes next, per sequence. The evaluator sends the camera's
# real position in every request, so this only needs to remember intent.
_sweep_direction: Dict[str, int] = {}


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
) -> Optional[RequestedViewDto]:
    """Sweep sideways at the deepest zoom the camera can reach right now.

    Everything here is read from ``request.camera_constraints`` rather than
    hardcoded, which is the whole trick: honour the constraints you are handed
    and your commands cannot be rejected. Return ``None`` to hold position.
    """
    constraints = request.camera_constraints
    current = request.view
    allowed = [level for level in constraints.allowed_resolution_levels if level > 0]
    if not allowed:
        return None

    # Zoom in one step at a time; L0 cannot reach L2 directly.
    target_level = min(max(allowed), current.resolution_level + 1)
    bounds = constraints.bounds_for_level(target_level)
    if bounds is None:
        return None

    # Coming from the full view there is only one legal centre to start from.
    if current.resolution_level == 0:
        centre_x = (bounds.minimum_center_x + bounds.maximum_center_x) // 2
        centre_y = (bounds.minimum_center_y + bounds.maximum_center_y) // 2
        return RequestedViewDto(
            resolution_level=target_level,
            center_x=int(centre_x),
            center_y=int(centre_y),
        )

    direction = _sweep_direction.setdefault(request.sequence_id, 1)

    # Move as far as this response is allowed to, and no further. The limit
    # belongs to the level the camera is on now, not the one we are going to.
    limit = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS[
        current.resolution_level
    ]
    step = int(limit * 0.9)

    centre_x = current.center_x + direction * step
    if centre_x > bounds.maximum_center_x or centre_x < bounds.minimum_center_x:
        # Turn around at the edge and drop down a row.
        direction = -direction
        _sweep_direction[request.sequence_id] = direction
        centre_x = current.center_x + direction * step

    centre_y = current.center_y

    # Clamp into the legal window. int() matters: these fields are strict ints
    # on the evaluator, so a float here is a validation error.
    centre_x = int(min(max(centre_x, bounds.minimum_center_x), bounds.maximum_center_x))
    centre_y = int(min(max(centre_y, bounds.minimum_center_y), bounds.maximum_center_y))

    return RequestedViewDto(
        resolution_level=target_level,
        center_x=centre_x,
        center_y=centre_y,
    )
