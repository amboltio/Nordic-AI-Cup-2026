"""Helpers for decoding views, checking responses and reading the sample data.

Nothing here is required by the protocol. It exists so that the boring parts
(Base64, coordinate conversion, response validation) are already solved and the
mistakes they cause surface on your machine instead of during an attempt.

The part worth reading twice is the coordinates section. What you observe is
crop-local; what you answer is frame-global.
"""

import base64
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    TRANSMITTED_VIEW_SIZE,
    DroneFlybyPredictResponseDto,
    DroneFlybyViewDto,
)


DATA_DIRECTORY = Path(__file__).resolve().parent / 'src'
DEFAULT_SCENE = 'helsinki'


# --------------------------------------------------------------------------- #
# Images on the wire
# --------------------------------------------------------------------------- #

def decode_view(view: DroneFlybyViewDto) -> np.ndarray:
    """Decode a transmitted view into a 960x540 BGR array.

    BGR because that is what OpenCV hands you everywhere else. Convert with
    ``cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`` if your model wants RGB.
    """
    return decode_image(view.image)


def decode_image(encoded_image: str) -> np.ndarray:
    """Decode a Base64 PNG string into a BGR array."""
    image_bytes = base64.b64decode(encoded_image)
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not decode the transmitted image')
    return image


def encode_image(image: np.ndarray) -> str:
    """Encode a BGR array as a Base64 PNG, the way the evaluator does."""
    encoded, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not encoded:
        raise ValueError('Could not encode the image')
    return base64.b64encode(buffer.tobytes()).decode('ascii')


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
#
# Three coordinate systems are in play, and the protocol is deliberately
# asymmetric about which one belongs where:
#
# * view-normalized  - [0, 1] inside the 960x540 image you just received;
# * source pixels    - [0, 3840] x [0, 2160] in the full frame;
# * frame-global     - [0, 1] across the full frame.
#
# You observe locally and you answer globally: every box in a response is
# frame-global, whatever the camera happens to be showing. The conversion you
# want after running a detector on the transmitted image is therefore
# :func:`view_bbox_to_global`.


def view_bbox_to_source(
    bbox: Sequence[float],
    source_region_xyxy: Sequence[int],
) -> Tuple[float, float, float, float]:
    """Map a box normalized to the transmitted view into source pixels.

    ``source_region_xyxy`` comes straight from ``request.view``. This is the
    first half of turning a detection into an answer; the second half is
    :func:`source_bbox_to_global`.
    """
    local_x1, local_y1, local_x2, local_y2 = (float(c) for c in bbox)
    source_x1, source_y1, source_x2, source_y2 = source_region_xyxy
    source_width = source_x2 - source_x1
    source_height = source_y2 - source_y1
    return (
        source_x1 + local_x1 * source_width,
        source_y1 + local_y1 * source_height,
        source_x1 + local_x2 * source_width,
        source_y1 + local_y2 * source_height,
    )


def source_bbox_to_view(
    bbox: Sequence[float],
    source_region_xyxy: Sequence[int],
) -> Tuple[float, float, float, float]:
    """Map a source-pixel box into coordinates normalized to the transmitted view.

    The inverse of :func:`view_bbox_to_source`. Useful for drawing on the image
    you received, or for cropping a known object out of it. A response uses
    :func:`source_bbox_to_global`.
    """
    x1, y1, x2, y2 = (float(c) for c in bbox)
    source_x1, source_y1, source_x2, source_y2 = source_region_xyxy
    source_width = source_x2 - source_x1
    source_height = source_y2 - source_y1
    return (
        (x1 - source_x1) / source_width,
        (y1 - source_y1) / source_height,
        (x2 - source_x1) / source_width,
        (y2 - source_y1) / source_height,
    )


def source_bbox_to_global(
    bbox: Sequence[float],
    original_width: int = IMAGE_WIDTH,
    original_height: int = IMAGE_HEIGHT,
) -> Tuple[float, float, float, float]:
    """Normalize a source-pixel box against the full frame.

    This is the coordinate system every ``bbox`` in a response uses. Take the
    dimensions from ``request.original_width`` and ``request.original_height``
    rather than the defaults if you want to be strict about it.
    """
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return (
        x1 / original_width,
        y1 / original_height,
        x2 / original_width,
        y2 / original_height,
    )


def global_bbox_to_source(
    bbox: Sequence[float],
    original_width: int = IMAGE_WIDTH,
    original_height: int = IMAGE_HEIGHT,
) -> Tuple[float, float, float, float]:
    """Scale a frame-global box back into source pixels.

    The inverse of :func:`source_bbox_to_global`, and exactly what the
    evaluator does to your annotations before it scores them: a plain scale by
    the frame dimensions.
    """
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return (
        x1 * original_width,
        y1 * original_height,
        x2 * original_width,
        y2 * original_height,
    )


def view_bbox_to_global(
    bbox: Sequence[float],
    source_region_xyxy: Sequence[int],
    original_width: int = IMAGE_WIDTH,
    original_height: int = IMAGE_HEIGHT,
) -> Tuple[float, float, float, float]:
    """Lift a detection made on the transmitted view into response coordinates.

    The whole local-to-global pipeline in one call, and the function most
    solutions need on every detection:

        view-normalized -> source pixels -> frame-global

    At Level 0 the crop covers the whole frame and the conversion is the
    identity. At Level 1 and Level 2 the crop is a sub-region, and the
    conversion places the box in the frame.
    """
    return source_bbox_to_global(
        view_bbox_to_source(bbox, source_region_xyxy),
        original_width,
        original_height,
    )


def source_region_for_view(resolution_level: int, center_x: int, center_y: int):
    """Return the source rectangle a camera position covers."""
    width, height = SOURCE_REGION_SIZES[resolution_level]
    half_width, half_height = width // 2, height // 2
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def center_bounds_for_level(resolution_level: int) -> Tuple[int, int, int, int]:
    """Return (min_x, max_x, min_y, max_y) centers that keep the crop in frame."""
    width, height = SOURCE_REGION_SIZES[resolution_level]
    return (
        width // 2,
        IMAGE_WIDTH - width // 2,
        height // 2,
        IMAGE_HEIGHT - height // 2,
    )


def clip_bbox_to_frame(bbox: Sequence[float], epsilon: float = 1e-6):
    """Clip a frame-global box into [0, 1], or return None if nothing is left.

    A box that has been clipped down to zero width or height is invalid, and
    an invalid box fails the whole response. Returning None here lets you drop
    it instead of losing the frame.
    """
    x1, y1, x2, y2 = (float(c) for c in bbox)
    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
    if x2 - x1 <= epsilon or y2 - y1 <= epsilon:
        return None
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #

def validate_response(response: DroneFlybyPredictResponseDto) -> None:
    """Check a response against the evaluator's rules.

    ``dtos.py`` already enforces most of this at construction time. This is the
    belt-and-braces version with error messages that name the offending
    detection, which is what you want when a whole frame is being thrown away
    and you cannot see why.
    """
    if len(response.annotations) > 500:
        raise ValueError(
            f'A response may carry at most 500 annotations, got '
            f'{len(response.annotations)}'
        )

    for index, annotation in enumerate(response.annotations):
        prefix = f'annotations[{index}]'
        if annotation.object_id not in OBJECT_CLASSES:
            raise ValueError(
                f'{prefix}: unknown object_id {annotation.object_id!r}. '
                f'Names are case-sensitive.'
            )
        if len(annotation.bbox) != 4:
            raise ValueError(f'{prefix}: bbox must have exactly 4 values')
        x1, y1, x2, y2 = (float(c) for c in annotation.bbox)
        if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
            raise ValueError(f'{prefix}: bbox coordinates must be finite')
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError(
                f'{prefix}: bbox {list(annotation.bbox)} is not a source-frame '
                f'normalized [x1, y1, x2, y2] box with x1 < x2 and y1 < y2 '
                f'inside [0, 1]'
            )
        confidence = float(annotation.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f'{prefix}: confidence must be between 0 and 1')

    requested_view = response.requested_view
    if requested_view is None:
        return

    for name in ('resolution_level', 'center_x', 'center_y'):
        value = getattr(requested_view, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f'requested_view.{name} must be an int, got {type(value).__name__}'
            )
    if requested_view.resolution_level not in SOURCE_REGION_SIZES:
        raise ValueError(
            f'requested_view.resolution_level must be 0, 1 or 2, got '
            f'{requested_view.resolution_level}'
        )


def describe_camera_rejection(
    current_level: int,
    current_center: Tuple[int, int],
    requested_level: int,
    requested_center: Tuple[int, int],
) -> Optional[str]:
    """Return why a camera move would be rejected, or None if it is legal.

    Same three checks the evaluator runs, in the same order. Call it before you
    answer if you want to guarantee your camera never idles on a bad command.
    """
    if requested_level not in ALLOWED_RESOLUTION_LEVELS.get(current_level, ()):
        return (
            f'cannot change directly from resolution level {current_level} '
            f'to {requested_level}'
        )

    minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(
        requested_level
    )
    if not minimum_x <= requested_center[0] <= maximum_x:
        return (
            f'resolution level {requested_level} requires center_x in '
            f'[{minimum_x}, {maximum_x}]'
        )
    if not minimum_y <= requested_center[1] <= maximum_y:
        return (
            f'resolution level {requested_level} requires center_y in '
            f'[{minimum_y}, {maximum_y}]'
        )

    # Level 0 always resets to the full-frame center and ignores the limit.
    if requested_level == 0:
        return None

    distance = math.hypot(
        requested_center[0] - current_center[0],
        requested_center[1] - current_center[1],
    )
    limit = MAXIMUM_CENTER_DELTA_PIXELS[current_level]
    if distance > limit:
        return (
            f'center movement {distance:.2f}px exceeds the L{current_level} '
            f'limit of {limit:.2f}px'
        )
    return None


# --------------------------------------------------------------------------- #
# The supplied sample data
# --------------------------------------------------------------------------- #

def scene_directory(scene: str = DEFAULT_SCENE) -> Path:
    """Return the directory holding one supplied scene."""
    directory = DATA_DIRECTORY / scene
    if not directory.is_dir():
        available = ', '.join(sorted(p.name for p in DATA_DIRECTORY.iterdir())) \
            if DATA_DIRECTORY.is_dir() else 'none'
        raise FileNotFoundError(
            f'No scene {scene!r} under {DATA_DIRECTORY}. Available: {available}'
        )
    return directory


def frame_numbers(scene: str = DEFAULT_SCENE) -> List[int]:
    """Return the sorted frame numbers available in a scene."""
    images = scene_directory(scene) / 'images'
    return sorted(
        int(path.stem.split('_')[-1]) for path in images.glob('frame_*.png')
    )


def load_frame(frame: int, scene: str = DEFAULT_SCENE) -> np.ndarray:
    """Load one full-resolution source frame as a BGR array."""
    path = scene_directory(scene) / 'images' / f'frame_{frame:06d}.png'
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Could not read {path}')
    return image


def load_annotations(frame: int, scene: str = DEFAULT_SCENE) -> List[Dict]:
    """Load the ground-truth detections for one frame.

    Boxes are ``[x1, y1, x2, y2]`` in source pixels, which is the coordinate
    system you are scored in. A response uses the same geometry, normalized
    against the full frame; :func:`source_bbox_to_global` converts these
    straight into answers.
    """
    path = scene_directory(scene) / 'annotations' / f'frame_{frame:06d}.json'
    with open(path) as handle:
        return json.load(handle)['annotations']


def load_sample(frame: int, scene: str = DEFAULT_SCENE):
    """Load one frame together with its ground truth."""
    return load_frame(frame, scene), load_annotations(frame, scene)


def load_run_metadata(scene: str = DEFAULT_SCENE) -> Dict:
    """Load the scene's capture metadata and object totals."""
    with open(scene_directory(scene) / 'run_metadata.json') as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

# Distinct-ish colours so neighbouring classes stay tellable apart.
_PALETTE = (
    (56, 168, 0), (0, 132, 255), (255, 128, 0), (200, 0, 200), (0, 200, 200),
    (40, 40, 220), (140, 200, 0), (255, 80, 140), (0, 96, 160), (160, 96, 0),
    (96, 0, 160), (0, 176, 96), (220, 180, 0), (120, 120, 255), (255, 200, 120),
    (0, 220, 140), (180, 0, 60),
)


def class_colour(object_id: str) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a class name."""
    try:
        index = OBJECT_CLASSES.index(object_id)
    except ValueError:
        return (128, 128, 128)
    return _PALETTE[index % len(_PALETTE)]


def draw_boxes(
    image: np.ndarray,
    annotations: Sequence[Dict],
    labels: bool = True,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Draw labelled boxes onto a copy of an image.

    Annotations are dicts with ``object_id`` and a source-pixel ``bbox``, the
    shape the supplied JSON files use. Line widths and text scale with the
    image so the result is readable at 4K and at preview sizes alike.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(width / 1920.0, 0.5)
    box_thickness = thickness if thickness is not None else max(1, round(2 * scale))
    font_scale = 0.6 * scale
    font = cv2.FONT_HERSHEY_SIMPLEX

    for annotation in annotations:
        object_id = annotation['object_id']
        x1, y1, x2, y2 = (int(round(float(c))) for c in annotation['bbox'])
        colour = class_colour(object_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_thickness)
        if not labels:
            continue

        confidence = annotation.get('confidence')
        text = object_id if confidence is None else f'{object_id} {float(confidence):.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, max(1, box_thickness - 1)
        )
        # Put the label above the box, or inside it when there is no room.
        text_y = y1 - baseline if y1 - text_height - baseline >= 0 else y2 + text_height
        cv2.rectangle(
            canvas,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y + baseline),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x1, text_y),
            font,
            font_scale,
            (255, 255, 255),
            max(1, box_thickness - 1),
            cv2.LINE_AA,
        )
    return canvas
