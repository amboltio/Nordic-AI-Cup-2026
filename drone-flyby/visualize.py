"""Render the supplied frames with their ground-truth boxes drawn on.

The kit ships the raw frames and the annotations; this draws one on the other
so you can see what you are being asked to find.

    python visualize.py --frame 0                    # one preview next to the kit
    python visualize.py --all --out annotated/       # the whole scene
    python visualize.py --frame 0 --show             # in a window
    python visualize.py --frame 0 --full-res         # at the source 3840x2160

Previews are written at 1920x1080 by default, which is plenty for looking at
and about a twentieth of the file size.
"""

import argparse
import sys
from pathlib import Path

import cv2

from utils import (
    DEFAULT_SCENE,
    draw_boxes,
    frame_numbers,
    load_annotations,
    load_frame,
    load_run_metadata,
)


PREVIEW_WIDTH = 1920
DEFAULT_OUTPUT_DIRECTORY = Path('annotated')


def render_frame(
    frame: int,
    scene: str,
    full_res: bool,
    labels: bool = True,
    width: int = PREVIEW_WIDTH,
):
    """Load one frame, draw its ground truth, and scale it for viewing."""
    image = load_frame(frame, scene)
    annotations = load_annotations(frame, scene)
    canvas = draw_boxes(image, annotations, labels=labels)

    if not full_res and canvas.shape[1] > width:
        height = round(canvas.shape[0] * width / canvas.shape[1])
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return canvas, annotations


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Draw the supplied ground-truth boxes onto the frames.'
    )
    parser.add_argument('--scene', default=DEFAULT_SCENE, help='Scene under src/.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--frame', type=int, help='Render a single frame number.')
    group.add_argument('--all', action='store_true', help='Render every frame.')
    parser.add_argument(
        '--out',
        type=Path,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIRECTORY}/).',
    )
    parser.add_argument(
        '--full-res',
        action='store_true',
        help='Write at the source 3840x2160 instead of a 1920-wide preview.',
    )
    parser.add_argument(
        '--no-labels', action='store_true', help='Draw boxes without class names.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=PREVIEW_WIDTH,
        help=f'Preview width in pixels (default: {PREVIEW_WIDTH}).',
    )
    parser.add_argument(
        '--out-file',
        type=Path,
        help='Write a single frame to this exact path instead of a directory.',
    )
    parser.add_argument(
        '--show', action='store_true', help='Open a window instead of writing a file.'
    )
    arguments = parser.parse_args()

    try:
        available = frame_numbers(arguments.scene)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not available:
        print(f'No frames found for scene {arguments.scene!r}.', file=sys.stderr)
        return 1

    if arguments.all:
        frames = available
    elif arguments.frame is not None:
        if arguments.frame not in available:
            print(
                f'Frame {arguments.frame} is not in scene {arguments.scene!r}. '
                f'Available: {available[0]}-{available[-1]}.',
                file=sys.stderr,
            )
            return 1
        frames = [arguments.frame]
    else:
        frames = [available[0]]

    metadata = load_run_metadata(arguments.scene)
    print(
        f'{arguments.scene}: {len(available)} frames, '
        f'{metadata.get("total_objects", "?")} object instances, '
        f'{metadata.get("capture", {}).get("altitude_m", "?")} m altitude'
    )

    if arguments.out_file and len(frames) > 1:
        print('--out-file takes a single --frame.', file=sys.stderr)
        return 1

    output_directory = arguments.out or DEFAULT_OUTPUT_DIRECTORY
    if not arguments.show and not arguments.out_file:
        output_directory.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        canvas, annotations = render_frame(
            frame,
            arguments.scene,
            arguments.full_res,
            labels=not arguments.no_labels,
            width=arguments.width,
        )
        if arguments.show:
            cv2.imshow(f'{arguments.scene} frame {frame}', canvas)
            print(f'frame {frame}: {len(annotations)} objects. Press any key.')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

        destination = arguments.out_file or (
            output_directory / f'frame_{frame:06d}.png'
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), canvas)
        print(
            f'frame {frame}: {len(annotations):2d} objects -> {destination} '
            f'({canvas.shape[1]}x{canvas.shape[0]})'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
