"""Score your endpoint against the supplied conversations.

    python local_evaluator.py                 # score the 390 supplied questions
    python local_evaluator.py --oracle        # feed ground truth in; prints 1.000
    python local_evaluator.py --verbose       # one line per question
    python local_evaluator.py --url ...       # point at a remote endpoint

This replays the questions the way the evaluation service does: one POST per
conversation, carrying all ten of its questions, strictly sequential, in file
order. A request that fails, times out, returns a non-2xx, answers the wrong
number of questions or sends a body that will not parse scores every question
about that conversation wrong, and the run carries on — same as the real thing.

Two things end an attempt early on the service and are mirrored below: five
timeouts in a row, and a whole-attempt budget of 60 seconds per conversation.
With 39 conversations in this folder either can fire locally — a server that is
consistently over budget will end the run early here just as it would in the
real attempt.

Scoring has two halves. Answering the question is one; pointing at the passage
you answered from is the other, measured as temporal IoU against the annotated
span. Read the per-type breakdown and the evidence block, not the headline
number.
"""

import argparse
import collections
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from dtos import ASRQuestionResponseDto
from utils import (
    Span,
    encode_audio,
    evidence_interval,
    gold_evidence,
    group_questions_by_conversation,
    load_sample_audio,
    temporal_iou,
)

DEFAULT_URL = 'http://localhost:9054/predict'

# Mirrors the timeout the evaluation service uses per request. One request now
# covers a whole conversation, so this budget has to fit one transcription plus
# every answer.
REQUEST_TIMEOUT_SECONDS = 60

# Mirrors the service again: the whole attempt gets one request's budget per
# conversation. Spending it all on the early conversations means the late ones
# are never sent, and their questions are scored wrong.
ATTEMPT_BUDGET_SECONDS_PER_CONVERSATION = REQUEST_TIMEOUT_SECONDS

# Mirrors the service: five consecutive timeouts and the rest of the attempt is
# cancelled. With 39 supplied conversations this can fire locally, so a run that
# stops early is telling you something the accuracy number alone would not.
MAX_CONSECUTIVE_TIMEOUTS = 5

# Fills an answer slot that never arrived. Matches neither label, so an
# unanswered question is counted wrong without any special casing.
UNANSWERED = -1

# The label a question carries when the answer is yes, which is also the only
# case that has an evidence span to score.
YES = 1

# The final score splits between answering the question and pointing at the
# passage the answer came from. Both halves carry real weight.
ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

@dataclass
class Statistics:
    """Everything worth knowing about a run."""

    total: int = 0
    correct: int = 0
    errors: int = 0

    conversations: int = 0
    failed_conversations: int = 0
    timeouts: int = 0
    aborted: bool = False

    by_type: Dict[str, List[int]] = field(
        default_factory=lambda: collections.defaultdict(lambda: [0, 0])
    )
    latencies_ms: List[float] = field(default_factory=list)
    questions_per_request: List[int] = field(default_factory=list)

    # One entry per question that has an annotated span to find, whether or not
    # anything usable came back for it.
    tious: List[float] = field(default_factory=list)
    missing_spans: int = 0
    # Diagnostic only: the same IoUs, restricted to the ones actually answered
    # yes. Not part of the score.
    tious_answered_yes: List[float] = field(default_factory=list)

    def record(
        self,
        question_type: str,
        label: int,
        prediction: int,
        gold: Optional[Span] = None,
        predicted: Optional[Span] = None,
    ) -> float:
        """Score one question. Returns its temporal IoU, for the verbose line."""
        self.total += 1

        if prediction == UNANSWERED:
            self.errors += 1

        # An error is a wrong answer, not an excused one.
        is_correct = int(prediction == label)
        self.correct += is_correct

        bucket = self.by_type[question_type]
        bucket[0] += is_correct
        bucket[1] += 1

        # Only the annotated yes questions have a passage to point at. The
        # denominator is theirs, not ours: answering no to one of them scores a
        # zero here rather than dropping out of the average.
        if label != YES or gold is None:
            return 0.0

        iou = temporal_iou(gold, predicted)
        self.tious.append(iou)

        if predicted is None:
            self.missing_spans += 1

        if prediction == YES:
            self.tious_answered_yes.append(iou)

        return iou

    def record_request(
        self,
        question_count: int,
        latency_ms: Optional[float],
        failed: bool,
        timed_out: bool = False,
    ) -> None:
        self.conversations += 1

        if failed:
            self.failed_conversations += 1

        if timed_out:
            self.timeouts += 1

        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)
            self.questions_per_request.append(question_count)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mean_tiou(self) -> float:
        return sum(self.tious) / len(self.tious) if self.tious else 0.0

    @property
    def mean_tiou_answered_yes(self) -> float:
        if not self.tious_answered_yes:
            return 0.0
        return sum(self.tious_answered_yes) / len(self.tious_answered_yes)

    @property
    def final_score(self) -> float:
        return ACCURACY_WEIGHT * self.accuracy + TIOU_WEIGHT * self.mean_tiou

    def report(self) -> str:
        lines = ['', 'Attempt statistics']
        lines.append(f'  questions            {self.total}')
        lines.append(f'  correct              {self.correct}')
        lines.append(f'  unanswered           {self.errors}')
        lines.append(f'  conversations        {self.conversations}')
        lines.append(f'  failed conversations {self.failed_conversations}')
        lines.append(f'  timeouts             {self.timeouts}')

        if self.aborted:
            lines.append('  ABORTED: the service would have stopped sending here')

        lines.append('')
        lines.append('Accuracy by question type')
        for question_type in ('positive', 'hard_negative', 'off_topic'):
            correct, total = self.by_type.get(question_type, [0, 0])
            if total:
                lines.append(
                    f'  {question_type:<20} {correct / total:.3f}  ({correct}/{total})'
                )

        lines.append('')
        lines.append('Evidence localization')
        lines.append(
            f'  {"mean tIoU":<24} {self.mean_tiou:.3f}  '
            f'(over {len(self.tious)} annotated yes questions)'
        )
        lines.append(f'  {"no span returned":<24} {self.missing_spans}')
        lines.append(
            f'  {"tIoU when answered yes":<24} '
            f'{self.mean_tiou_answered_yes:.3f}  '
            f'(diagnostic, n={len(self.tious_answered_yes)}, not scored)'
        )

        if self.latencies_ms:
            lines.append('')
            lines.append('Round trip')
            lines.append(self._latency_lines())

        lines.append('')
        lines.append(f'Accuracy:  {self.accuracy:.3f}')
        lines.append(f'Mean tIoU: {self.mean_tiou:.3f}')
        lines.append(f'Score:     {self.final_score:.3f}')
        return '\n'.join(lines)

    def _latency_lines(self) -> str:
        mean = sum(self.latencies_ms) / len(self.latencies_ms)
        worst = max(self.latencies_ms)
        answered = sum(self.questions_per_request)
        per_question = sum(self.latencies_ms) / answered if answered else 0.0

        budget_ms = REQUEST_TIMEOUT_SECONDS * 1000

        return '\n'.join([
            f'  {"per conversation":<24} {mean:8.0f} ms mean, '
            f'{worst:8.0f} ms worst  (n={len(self.latencies_ms)})',
            f'  {"per question":<24} {per_question:8.0f} ms',
            f'  {"budget":<24} {budget_ms:8.0f} ms  '
            f'({worst / budget_ms:.0%} of it used at worst)',
        ])


# --------------------------------------------------------------------------- #
# The replay loop
# --------------------------------------------------------------------------- #

def wait_for_endpoint(url: str, attempts: int = 30) -> bool:
    """Poll the server root until it answers, so a slow import is not an error."""
    root = url.rsplit('/', 1)[0] + '/'
    for _ in range(attempts):
        try:
            requests.get(root, timeout=2)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


def replay(url: str, verbose: bool) -> Statistics:
    """Send every supplied conversation to the endpoint and score the answers."""
    statistics = Statistics()
    session = requests.Session()
    consecutive_timeouts = 0

    conversations = group_questions_by_conversation()
    attempt_budget = len(conversations) * ATTEMPT_BUDGET_SECONDS_PER_CONVERSATION
    started_attempt = time.time()

    for audio_filename, rows in conversations:
        # The service stops sending once the whole-attempt budget is gone. The
        # questions it never sends are scored wrong there; here they are simply
        # left out, which the note below flags.
        if time.time() - started_attempt > attempt_budget:
            statistics.aborted = True
            print(
                f'  attempt budget of {attempt_budget} seconds is gone before '
                f'{audio_filename}: the service would stop sending here. The '
                'questions it never sent are scored wrong there, but left out '
                'of the numbers below, so this run is not comparable to a '
                'competition score.',
                file=sys.stderr,
            )
            break

        questions = [row['question'] for row in rows]

        payload = {
            'audio_base64': encode_audio(load_sample_audio(audio_filename)),
            'audio_filename': audio_filename,
            'questions': questions,
        }

        answers, spans, latency_ms, error, timed_out = _ask(
            session, url, payload, len(questions)
        )

        consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0

        statistics.record_request(
            question_count=len(questions),
            latency_ms=latency_ms,
            failed=error is not None,
            timed_out=timed_out,
        )

        if error is not None:
            print(
                f'  {audio_filename}: {error} '
                f'All {len(questions)} questions scored wrong.',
                file=sys.stderr,
            )

        for row, prediction, predicted_span in zip(rows, answers, spans):
            label = int(row['label'])
            gold = gold_evidence(row)

            iou = statistics.record(
                row['question_type'], label, prediction, gold, predicted_span
            )

            if verbose:
                mark = 'ok  ' if prediction == label else 'WRONG'
                said = {1: 'yes', 0: 'no'}.get(prediction, '-')
                evidence = f' tIoU {iou:.3f}' if gold is not None else ''
                print(
                    f'  {mark} {row["question_id"]:<24} {row["question_type"]:<14}'
                    f' said {said:<3} wanted {row["answer"]:<3}{evidence}'
                )

        if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
            statistics.aborted = True
            print(
                f'  {MAX_CONSECUTIVE_TIMEOUTS} timeouts in a row: the service would '
                'stop sending here. The questions it never sent are scored wrong '
                'there, but simply left out of the numbers below, so this run '
                'is not comparable to a competition score.',
                file=sys.stderr,
            )
            break

    return statistics


def _ask(
    session: requests.Session,
    url: str,
    payload: dict,
    expected_count: int,
) -> Tuple[
    List[int],
    List[Optional[Span]],
    Optional[float],
    Optional[str],
    bool,
]:
    """One request, one conversation.

    Returns ``(predictions, spans, latency, error, timed_out)``. ``predictions``
    and ``spans`` always have one entry per question: ``UNANSWERED`` and ``None``
    wherever no usable answer arrived, so the caller never has to special-case a
    failure. ``timed_out`` is the only failure the service counts towards its
    five-in-a-row abort, so it is reported separately rather than folded into
    ``error``.

    Note what is *not* handled softly here: a body whose evidence lists are the
    wrong length fails to parse at all and lands in the catch-all below, losing
    every question about the conversation. That is what the service does too.
    """
    unanswered = [UNANSWERED] * expected_count
    no_spans: List[Optional[Span]] = [None] * expected_count

    started = time.time()
    try:
        response = session.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        latency_ms = (time.time() - started) * 1000
        response.raise_for_status()

        prediction = ASRQuestionResponseDto.model_validate(response.json())
        answers = prediction.answers

        if len(answers) != expected_count:
            return (
                unanswered,
                no_spans,
                latency_ms,
                f'expected {expected_count} answers, got {len(answers)}.',
                False,
            )

        spans = [
            evidence_interval(start, end)
            for start, end in zip(prediction.evidence_start, prediction.evidence_end)
        ]

        return (
            [int(answer) for answer in answers],
            spans,
            latency_ms,
            None,
            False,
        )

    # A timeout is the one failure that accumulates, so it cannot stay hidden in
    # the catch-all below.
    except requests.Timeout:
        return (
            unanswered,
            no_spans,
            None,
            f'no answer within {REQUEST_TIMEOUT_SECONDS} seconds.',
            True,
        )

    except Exception as exc:
        return unanswered, no_spans, None, f'{type(exc).__name__}: {exc}', False


def oracle() -> Statistics:
    """Score the ground truth against itself. Proves harness and data agree."""
    statistics = Statistics()

    for _, rows in group_questions_by_conversation():
        statistics.record_request(len(rows), None, failed=False)

        for row in rows:
            label = int(row['label'])
            gold = gold_evidence(row)
            statistics.record(row['question_type'], label, label, gold, gold)

    return statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL, help='Your /predict endpoint.')
    parser.add_argument('--oracle', action='store_true',
                        help='Score the ground truth instead of your endpoint.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print a line per question.')
    args = parser.parse_args()

    if args.oracle:
        print(oracle().report())
        print('\nThis is the harness scoring the ground truth. Anything below '
              '1.000 means the\nsetup is broken, not the model.')
        return 0

    if not wait_for_endpoint(args.url):
        print(f'Nothing answering at {args.url}. Start it with "python api.py".',
              file=sys.stderr)
        return 1

    statistics = replay(args.url, args.verbose)
    print(statistics.report())
    print(f'\n{statistics.conversations} conversations is a correctness check, '
          'not a benchmark. The breakdown by\nquestion type and the evidence '
          'block are the numbers to act on.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
