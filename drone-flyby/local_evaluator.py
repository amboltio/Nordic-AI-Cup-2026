"""Replay a supplied scene through your endpoint and score it locally.

This mirrors the evaluation service: the same crops, the same 960x540 PNG
payloads, the same camera rules, the same COCO mAP at IoU 0.50. Use it to find
out whether your server actually works before you spend your one evaluation
attempt on finding out.

Like the service, it reads your annotations as predictions for the whole source
frame and scores each frame against every object in it, not only the ones the
camera was showing.

    python local_evaluator.py                     # score every frame, no clock
    python local_evaluator.py --realtime           # add the 3 fps frame clock
    python local_evaluator.py --oracle             # score the ground truth (= 1.0)
    python local_evaluator.py --simulate-latency-ms 400 --realtime

Two modes, because they answer different questions:

* The default sends every frame and waits as long as it takes. It measures how
  good your detector is.
* ``--realtime`` runs the real clock. Frames are emitted every 333 ms whether
  you are ready or not, and only the newest one is ever sent, so a slow server
  loses frames. It measures what you would actually score.

The supplied helsinki scene holds one instance of each class, so treat the
number as a sanity check rather than a leaderboard prediction.
"""

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    TRANSMITTED_VIEW_SIZE,
    DroneFlybyPredictResponseDto,
)
from utils import (
    DEFAULT_SCENE,
    center_bounds_for_level,
    frame_numbers,
    global_bbox_to_source,
    load_annotations,
    load_frame,
    source_region_for_view,
)


DEFAULT_URL = 'http://localhost:9053/predict'
# Derived the same way the evaluation service derives them, so the numbers you
# see in the request here are the numbers you will see in an attempt.
FRAMES_PER_SECOND = 3.0
RESPONSE_TIMEOUT_FRAMES = 10.0
FRAME_INTERVAL_SECONDS = 1.0 / FRAMES_PER_SECOND
RESPONSE_TIMEOUT_SECONDS = RESPONSE_TIMEOUT_FRAMES * FRAME_INTERVAL_SECONDS
FRAME_INTERVAL_MS = int(FRAME_INTERVAL_SECONDS * 1000)       # 333
RESPONSE_TIMEOUT_MS = int(RESPONSE_TIMEOUT_SECONDS * 1000)   # 3333
SEQUENCE_ID = 'local'


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

class CameraRejection(ValueError):
    """Raised when a requested view cannot follow the current one."""


@dataclass
class Camera:
    """The evaluator's camera: an absolute position that persists across frames."""

    resolution_level: int = 0
    center_x: int = FULL_FRAME_CENTER[0]
    center_y: int = FULL_FRAME_CENTER[1]

    def apply(self, resolution_level: int, center_x: int, center_y: int) -> None:
        """Validate a requested view and move to it, or raise CameraRejection."""
        if resolution_level not in ALLOWED_RESOLUTION_LEVELS[self.resolution_level]:
            raise CameraRejection(
                f'cannot change directly from resolution level '
                f'{self.resolution_level} to {resolution_level}'
            )

        minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(
            resolution_level
        )
        if not minimum_x <= center_x <= maximum_x:
            raise CameraRejection(
                f'resolution level {resolution_level} requires center_x in '
                f'[{minimum_x}, {maximum_x}]'
            )
        if not minimum_y <= center_y <= maximum_y:
            raise CameraRejection(
                f'resolution level {resolution_level} requires center_y in '
                f'[{minimum_y}, {maximum_y}]'
            )

        distance = math.hypot(center_x - self.center_x, center_y - self.center_y)
        if resolution_level == 0:
            # Returning to the full view is one move and ignores the limit,
            # but it has to be the full-frame centre.
            if (center_x, center_y) != FULL_FRAME_CENTER:
                raise CameraRejection(f'full view must use center {FULL_FRAME_CENTER}')
        else:
            # The limit belongs to the level the camera is on now.
            limit = MAXIMUM_CENTER_DELTA_PIXELS[self.resolution_level]
            if distance > limit:
                raise CameraRejection(
                    f'center movement {distance:.2f}px exceeds the '
                    f'L{self.resolution_level} limit of {limit:.2f}px'
                )

        self.resolution_level = resolution_level
        self.center_x = center_x
        self.center_y = center_y

    @property
    def source_region(self) -> Tuple[int, int, int, int]:
        return source_region_for_view(
            self.resolution_level, self.center_x, self.center_y
        )

    def constraints(self) -> dict:
        allowed = ALLOWED_RESOLUTION_LEVELS[self.resolution_level]
        bounds = []
        for level in allowed:
            minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(level)
            bounds.append(
                {
                    'resolution_level': level,
                    'width': TRANSMITTED_VIEW_SIZE[0],
                    'height': TRANSMITTED_VIEW_SIZE[1],
                    'minimum_center_x': minimum_x,
                    'maximum_center_x': maximum_x,
                    'minimum_center_y': minimum_y,
                    'maximum_center_y': maximum_y,
                }
            )
        return {
            'maximum_center_delta': MAXIMUM_CENTER_DELTA_PIXELS[self.resolution_level],
            'allowed_resolution_levels': list(allowed),
            'center_bounds': bounds,
            'full_view_reset_exempt_from_delta': True,
        }


def render_view(frame_image: np.ndarray, camera: Camera) -> str:
    """Crop, downsample and Base64-encode a view exactly as the evaluator does."""
    x1, y1, x2, y2 = camera.source_region
    source_view = frame_image[y1:y2, x1:x2]

    if (source_view.shape[1], source_view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        # INTER_AREA is what the evaluator uses. Both downscales here are exact
        # integer factors, so this is plain box averaging.
        source_view = cv2.resize(
            source_view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA
        )

    encoded, buffer = cv2.imencode('.png', source_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not encoded:
        raise RuntimeError('Could not encode the transmitted view')
    return base64.b64encode(buffer.tobytes()).decode('ascii')


def build_request(
    frame: int,
    frame_index: int,
    camera: Camera,
    encoded_image: str,
    feedback: Optional[dict],
) -> dict:
    """Build the JSON body the evaluator would POST."""
    request_id = (
        f'{SEQUENCE_ID}:{frame_index}:{camera.resolution_level}:'
        f'{camera.center_x}:{camera.center_y}'
    )
    return {
        'sequence_id': SEQUENCE_ID,
        'frame': frame,
        'frame_index': frame_index,
        'request_id': request_id,
        'frame_interval_ms': FRAME_INTERVAL_MS,
        'response_timeout_ms': RESPONSE_TIMEOUT_MS,
        'original_width': IMAGE_WIDTH,
        'original_height': IMAGE_HEIGHT,
        'view': {
            'resolution_level': camera.resolution_level,
            'center_x': camera.center_x,
            'center_y': camera.center_y,
            'view_id': request_id,
            'image': encoded_image,
            'image_media_type': 'image/png',
            'width': TRANSMITTED_VIEW_SIZE[0],
            'height': TRANSMITTED_VIEW_SIZE[1],
            'source_region_xyxy': list(camera.source_region),
        },
        'camera_constraints': camera.constraints(),
        'camera_command_feedback': feedback,
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

@dataclass
class Statistics:
    """The same counters the evaluation service logs for your attempt."""

    frames_total: int = 0
    frames_sent: int = 0
    frames_skipped: int = 0
    frames_unanswered: int = 0
    responses_accepted: int = 0
    timeouts: int = 0
    http_errors: int = 0
    invalid_responses: int = 0
    commands_applied: int = 0
    invalid_commands: int = 0
    round_trip_ms: List[float] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f'  frames in scene      {self.frames_total}',
            f'  frames sent          {self.frames_sent}',
            f'  frames skipped       {self.frames_skipped}',
            f'  frames unanswered    {self.frames_unanswered}',
            f'  responses accepted   {self.responses_accepted}',
            f'  timeouts             {self.timeouts}',
            f'  http errors          {self.http_errors}',
            f'  invalid responses    {self.invalid_responses}',
            f'  camera moves applied {self.commands_applied}',
            f'  camera moves refused {self.invalid_commands}',
        ]
        if self.round_trip_ms:
            ordered = sorted(self.round_trip_ms)
            median = ordered[len(ordered) // 2]
            worst = ordered[-1]
            mean = sum(ordered) / len(ordered)
            lines.append(
                f'  round trip ms        mean {mean:.0f} / median {median:.0f} '
                f'/ max {worst:.0f}'
            )
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# The replay loop
# --------------------------------------------------------------------------- #

def wait_for_endpoint(url: str, attempts: int = 30) -> bool:
    """Poll the server root until it answers, the way the real harness does."""
    root = url.rsplit('/', 1)[0] if url.count('/') > 2 else url
    for attempt in range(attempts):
        try:
            requests.get(root, timeout=2)
            return True
        except requests.RequestException:
            if attempt == 0:
                print(f'Waiting for {root} ...')
            time.sleep(1)
    return False


def replay(
    url: str,
    scene: str,
    realtime: bool,
    simulate_latency_ms: float,
    verbose: bool,
) -> Tuple[Dict[int, List[dict]], Statistics]:
    """Send frames to the endpoint and collect its detections per frame."""
    frames = frame_numbers(scene)
    statistics = Statistics(frames_total=len(frames))
    predictions: Dict[int, List[dict]] = {}
    camera = Camera()
    feedback: Optional[dict] = None
    session = requests.Session()

    interval = FRAME_INTERVAL_SECONDS
    timeout = RESPONSE_TIMEOUT_SECONDS
    started = time.monotonic()
    frame_index = 0

    while frame_index < len(frames):
        frame = frames[frame_index]
        image = load_frame(frame, scene)
        encoded_image = render_view(image, camera)
        payload = build_request(frame, frame_index, camera, encoded_image, feedback)

        statistics.frames_sent += 1
        sent_at = time.monotonic()
        try:
            http_response = session.post(url, json=payload, timeout=timeout)
            http_response.raise_for_status()
            body = http_response.json()
        except requests.Timeout:
            statistics.timeouts += 1
            print(f'frame {frame}: no answer within {RESPONSE_TIMEOUT_MS} ms')
            body = None
        except requests.RequestException as exc:
            statistics.http_errors += 1
            print(f'frame {frame}: HTTP error: {exc}')
            body = None
        except json.JSONDecodeError as exc:
            statistics.invalid_responses += 1
            print(f'frame {frame}: response was not JSON: {exc}')
            body = None

        if simulate_latency_ms:
            time.sleep(simulate_latency_ms / 1000.0)
        statistics.round_trip_ms.append((time.monotonic() - sent_at) * 1000.0)

        if body is None:
            statistics.frames_unanswered += 1
        else:
            # Validate with the same models the evaluator uses. An invalid
            # response costs you every detection it carried.
            try:
                response = DroneFlybyPredictResponseDto.model_validate(body)
                if response.request_id != payload['request_id']:
                    raise ValueError(
                        f'request_id mismatch: expected '
                        f'{payload["request_id"]!r}, got {response.request_id!r}'
                    )
                if response.frame != frame:
                    raise ValueError(
                        f'frame mismatch: expected {frame}, got {response.frame}'
                    )
            except Exception as exc:
                statistics.invalid_responses += 1
                statistics.frames_unanswered += 1
                print(f'frame {frame}: invalid response: {exc}')
                response = None

            if response is not None:
                statistics.responses_accepted += 1
                # Responses are global, so this is a plain scale by the
                # frame size, and it accepts boxes anywhere in the frame.
                predictions[frame] = [
                    {
                        'object_id': annotation.object_id,
                        'bbox': global_bbox_to_source(
                            annotation.bbox,
                            payload['original_width'],
                            payload['original_height'],
                        ),
                        'confidence': float(annotation.confidence),
                    }
                    for annotation in response.annotations
                ]
                if verbose:
                    print(
                        f'frame {frame:3d} idx {frame_index:3d} '
                        f'L{camera.resolution_level} '
                        f'({camera.center_x:4d},{camera.center_y:4d}) '
                        f'-> {len(response.annotations):3d} detections'
                    )

                # Apply the camera command. A refused move keeps the old view
                # and reports back on every following frame until it is fixed.
                if response.requested_view is not None:
                    requested = response.requested_view
                    try:
                        camera.apply(
                            requested.resolution_level,
                            requested.center_x,
                            requested.center_y,
                        )
                        statistics.commands_applied += 1
                        feedback = None
                    except CameraRejection as exc:
                        statistics.invalid_commands += 1
                        feedback = {
                            'frame': frame,
                            'requested_view': {
                                'resolution_level': requested.resolution_level,
                                'center_x': requested.center_x,
                                'center_y': requested.center_y,
                            },
                            'reason': str(exc),
                        }
                        print(f'frame {frame}: camera command refused: {exc}')

        if not realtime:
            frame_index += 1
            continue

        # The clock owns the sequence: frame i exists at start + i * interval,
        # and only the newest emitted frame is ever sent.
        elapsed = time.monotonic() - started
        next_index = max(frame_index + 1, int(elapsed / interval))
        # Only count frames that actually exist; the last jump can overshoot.
        counted = min(next_index, len(frames))
        statistics.frames_skipped += max(0, counted - (frame_index + 1))
        frame_index = next_index

    return predictions, statistics


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(
    scene: str,
    predictions: Dict[int, List[dict]],
) -> Tuple[float, Dict[str, float]]:
    """Calculate COCO mAP at IoU 0.50, the way the evaluation service does."""
    from faster_coco_eval import COCO, COCOeval_faster

    frames = frame_numbers(scene)
    ground_truth = {frame: load_annotations(frame, scene) for frame in frames}

    present_classes = {
        annotation['object_id']
        for annotations in ground_truth.values()
        for annotation in annotations
    }
    evaluated_classes = tuple(
        object_id for object_id in OBJECT_CLASSES if object_id in present_classes
    )
    if not evaluated_classes:
        raise ValueError('The scene has no ground-truth detections')

    frame_to_image_id = {frame: index for index, frame in enumerate(frames, start=1)}
    category_ids = {name: index for index, name in enumerate(OBJECT_CLASSES, start=1)}

    annotations = []
    annotation_id = 1
    for frame in frames:
        for annotation in ground_truth[frame]:
            x, y, width, height = _xyxy_to_xywh(annotation['bbox'])
            annotations.append(
                {
                    'id': annotation_id,
                    'image_id': frame_to_image_id[frame],
                    'category_id': category_ids[annotation['object_id']],
                    'bbox': [x, y, width, height],
                    'area': width * height,
                    'iscrowd': 0,
                }
            )
            annotation_id += 1

    coco_ground_truth = {
        'info': {'description': f'Drone flyby - {scene}'},
        'licenses': [],
        'images': [
            {
                'id': frame_to_image_id[frame],
                'file_name': f'frame_{frame:06d}.png',
                'width': IMAGE_WIDTH,
                'height': IMAGE_HEIGHT,
            }
            for frame in frames
        ],
        'categories': [
            {'id': category_ids[name], 'name': name, 'supercategory': 'object'}
            for name in OBJECT_CLASSES
        ],
        'annotations': annotations,
    }

    # A frame with no predictions still has ground truth, so it costs recall.
    coco_predictions = []
    for frame in frames:
        for detection in predictions.get(frame, []):
            x, y, width, height = _xyxy_to_xywh(detection['bbox'])
            if width <= 0 or height <= 0:
                continue
            coco_predictions.append(
                {
                    'image_id': frame_to_image_id[frame],
                    'category_id': category_ids[detection['object_id']],
                    'bbox': [x, y, width, height],
                    'score': detection['confidence'],
                }
            )

    if not coco_predictions:
        return 0.0, {name: 0.0 for name in evaluated_classes}

    coco_gt = COCO(coco_ground_truth)
    coco_dt = coco_gt.loadRes(coco_predictions)
    evaluator = COCOeval_faster(coco_gt, coco_dt, 'bbox')
    evaluator.params.imgIds = list(frame_to_image_id.values())
    evaluator.params.catIds = [category_ids[name] for name in evaluated_classes]
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval['precision']
    ap_by_class = {}
    for category_index, name in enumerate(evaluated_classes):
        class_precision = precision[0, :, category_index, 0, -1]
        valid = class_precision[class_precision > -1]
        ap_by_class[name] = _clamp(float(np.mean(valid)) if valid.size else 0.0)
    return _clamp(sum(ap_by_class.values()) / len(ap_by_class)), ap_by_class


def oracle_predictions(scene: str) -> Dict[int, List[dict]]:
    """Return the ground truth as perfect predictions, to check the scorer."""
    return {
        frame: [
            {
                'object_id': annotation['object_id'],
                'bbox': tuple(float(c) for c in annotation['bbox']),
                'confidence': 1.0,
            }
            for annotation in load_annotations(frame, scene)
        ]
        for frame in frame_numbers(scene)
    }


def _xyxy_to_xywh(bbox: Sequence[float]):
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return x1, y1, x2 - x1, y2 - y1


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Replay a scene through your endpoint and score it.'
    )
    parser.add_argument('--url', default=DEFAULT_URL, help='Your predict endpoint.')
    parser.add_argument('--scene', default=DEFAULT_SCENE, help='Scene under src/.')
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='Run the 3 fps frame clock, so a slow server loses frames.',
    )
    parser.add_argument(
        '--simulate-latency-ms',
        type=float,
        default=0.0,
        help='Pretend your server is this much slower. Useful with --realtime.',
    )
    parser.add_argument(
        '--oracle',
        action='store_true',
        help='Score the ground truth instead of calling your server. '
        'Should print 1.000 and proves the scorer agrees with the data.',
    )
    parser.add_argument('--verbose', action='store_true', help='Log every frame.')
    arguments = parser.parse_args()

    try:
        frame_numbers(arguments.scene)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if arguments.oracle:
        print(f'Scoring the ground truth for {arguments.scene} ...')
        predictions = oracle_predictions(arguments.scene)
        statistics = None
    else:
        if not wait_for_endpoint(arguments.url):
            print(
                f'Could not reach {arguments.url}. Start your server with '
                f'`python api.py` first.',
                file=sys.stderr,
            )
            return 1
        mode = 'realtime, 3 fps' if arguments.realtime else 'offline, every frame'
        print(f'Replaying {arguments.scene} through {arguments.url} ({mode})')
        predictions, statistics = replay(
            arguments.url,
            arguments.scene,
            arguments.realtime,
            arguments.simulate_latency_ms,
            arguments.verbose,
        )

    coco_map_50, ap_by_class = score(arguments.scene, predictions)

    print()
    if statistics is not None:
        print('Attempt statistics')
        print(statistics.report())
        print()
    print('AP@0.50 by class')
    for name, value in sorted(ap_by_class.items(), key=lambda item: -item[1]):
        print(f'  {name:16s} {value:.3f}')
    print()
    print(f'COCO mAP@0.50: {coco_map_50:.3f}')
    if statistics is not None and not arguments.realtime:
        print(
            'This is the offline number. Run with --realtime to see what the '
            'frame clock costs you.'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
