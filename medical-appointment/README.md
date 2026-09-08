# Medical appointment

![Medical Appointment AI image](../images/Medical_Appointment_Image.png)

A patient sees their doctor. The consultation is recorded, and afterwards
somebody wants to know what was actually agreed — which vaccine was given, what
dose was prescribed, whether the results were good. Your job is to answer that
question from the audio alone.

Each request carries one conversation and the ten yes/no questions asked about
it. You return ten booleans — and, for every one you answer yes, the stretch of
audio you answered from. No transcript is provided; the speech recognition is
yours to do.

So there are two things to get right. The catch in the first is that the wrong
answers are not nonsense. Most of them are near-misses — the right drug at the
wrong dose, the right course at the wrong length — so a system that hears
roughly what a conversation was about, without hearing it precisely, scores no
better than a coin toss.

The second is that saying yes is not enough: you have to point at the passage
that makes it true, as a start and end timestamp. "It is in here somewhere" is
worth nothing, and pointing at the whole conversation is worth almost nothing.
That half of the score is the larger one.

## Quickstart

```cmd
git clone https://github.com/amboltio/Nordic-AI-Cup-2026
cd Nordic-AI-Cup-2026/medical-appointment
pip install -r requirements.txt
```

Serve the baseline:

```cmd
python api.py
```

Then, in a second terminal, score it against the supplied conversations:

```cmd
python local_evaluator.py
```

You now have a working endpoint and a number to improve. The baseline answers
`true` to everything and points at nothing, which scores `0.200` — half the
answers right, none of the evidence found. That is the floor, not a start. It is
there to prove the plumbing works, not to compete.

Check that the harness and the data agree with each other at any time:

```cmd
python local_evaluator.py --oracle
```

That feeds the ground truth in — answers and spans alike — and should print
`1.000` for accuracy, tIoU and score. If it does, a low score is your model, not
your setup.

### What is in this folder

| File | What it is |
| --- | --- |
| `api.py` | The FastAPI server the evaluator calls. You probably will not change it. |
| `example.py` | The baseline. **This is the file to replace.** |
| `dtos.py` | The request and response models. |
| `utils.py` | Base64 decoding, MP3 duration, response validation, sample loading, the tIoU scoring helpers. |
| `local_evaluator.py` | Replays the supplied questions through your endpoint and scores it. |
| `requirements.txt` | Dependencies. Loose pins, so they will not fight your ASR stack. |
| `Dockerfile` | If you would rather containerise the server. |
| `data/` | 39 training samples: audio, questions, and the annotated evidence spans. |

## About the challenge

There are two halves to this and you can lose the points in either. The first
is hearing the conversation: names, numbers and units, spoken at conversational
speed. The second is deciding what the question is really asking, which is
where most of the difficulty lives — see below.

Both feed the same output. Because a yes has to carry the timestamps of the
passage it came from, the timings your transcription produces are part of your
answer, not just a means to an end. Pick an ASR setup that gives you segment or
word timings, not just text.

Nothing carries over between requests. Each conversation arrives complete, with
every question asked about it, and the evaluator keeps no state.

## The conversations

Simulated consultations between a doctor and a patient, in **English**. They
run from about one minute to three and a half, with the median around two.
Subject matter is ordinary general practice: vaccination scheduling,
prescriptions and doses, lab values, examination findings, side effects,
follow-up plans.

Each is a single-channel MP3, 128 kbps at 44.1 kHz. Both speakers share the one
channel, so if you want to know who said what, you will have to work it out.

## The questions

Every question is answerable with yes or no, and every conversation gets ten of
them — and all ten arrive in the same request. They come in three flavours, and
the difference between them is the challenge:

| Type | Answer | What it is |
| --- | --- | --- |
| `positive` | yes | Something the conversation actually establishes. |
| `hard_negative` | no | A near-miss on something the conversation establishes — the same drug at a different dose, the same symptom in a different place, a plausible-sounding detail that was never agreed. |
| `off_topic` | no | A subject that never comes up at all. |

Here are seven of the ten questions asked about one consultation, on a course
of antibiotics started after a swab result. Read the pairs:

```text
positive       Should the daily dose be 100 mg?                          yes
hard_negative  Was the prescribed dose 200 mg daily?                     no

positive       Will the treatment last two weeks?                        yes
hard_negative  The treatment is planned to run for six weeks, right?     no

positive       Is the medicine to be taken after a meal?                 yes
hard_negative  Should the tablets be taken on an empty stomach?          no

off_topic      Is there any mention of attending a concert?              no
```

The off-topic questions are free: if a subject never appears, the answer is no.
The hard negatives are not. They are lexically almost identical to the true
statement, so a model that answers from topical overlap gets every one of them
wrong. 

Two more things worth knowing. Yes and no answers are **exactly balanced** in
both the validation and the evaluation set, so a constant answer earns the
floor and no more. And the questions are not uniformly interrogative — some are
tag questions, like *"The lipid profile came back normal, didn't it?"* — so do
not key off sentence shape.

## Supplied data

39 training samples:

```text
data/
├── audio/
│   ├── conversation_sample_4.mp3
│   ├── conversation_sample_5.mp3
│   └── ...                            (39 conversations in all)
└── question_train.csv                 (390 questions, ten per conversation)
```

`data/question_train.csv` is the file `local_evaluator.py` reads. Its columns:

| Column | What it is |
| --- | --- |
| `question_id` | Identifies the question. Useful in your logs. |
| `transcript_id` | Names the conversation — `sample_17` is `data/audio/conversation_sample_17.mp3`. |
| `question` | The question, exactly as it will be sent to you. |
| `answer` | `yes` or `no`, in words. |
| `label` | The same thing as a number: `1` = yes, `0` = no. |
| `question_type` | `positive`, `hard_negative` or `off_topic`. |
| `evidence_start` | Where the supporting passage starts, in **seconds from the beginning of the audio**. |
| `evidence_end` | Where it ends, in seconds. |

The two evidence columns are only filled in where there is something to point
at — the 195 `positive` rows. For every `hard_negative` and `off_topic` row they
are **blank**, because a no answer has no supporting passage by definition. That
is exactly the shape your own response has to take.


## Your goal

Given one conversation and the ten questions about it, return the right ten
booleans, and for each yes the start and end second of the passage that supports
it. Every question is worth the same.

## What the evaluator sends you

One POST per conversation, to the URL you submitted, carrying every question
asked about it.

| Field | What it is |
| --- | --- |
| `audio_base64` | The MP3 file, base64 encoded. |
| `audio_filename` | e.g. `conversation_sample_17.mp3`. Identifies the conversation. |
| `questions` | The English yes/no questions about it. Ten of them. |

```json
{
  "audio_base64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYyLjMuMTAw...",
  "audio_filename": "conversation_sample_17.mp3",
  "questions": [
    "Should the daily dose be 100 mg?",
    "Was the prescribed dose 200 mg daily?",
    "Will the treatment last two weeks?"
  ]
}
```

The `questions` list is trimmed to three above; a real body carries all ten.

`audio_base64` is plain base64 of the file's bytes. There is **no
`data:audio/mpeg;base64,` prefix** — hand it straight to `base64.b64decode`.
Base64 costs a third on top of the file, so expect bodies of roughly 1.5 to
4.5 MB — one conversation per body, however many questions come with it.

That works out to 19 requests for a validation attempt and 38 for an evaluation
attempt, sent strictly one at a time, in file order.

## What you send back

| Field | What it is |
| --- | --- |
| `answers` | One boolean per question. `true` for yes, `false` for no. |
| `evidence_start` | One value per question: the second the supporting passage starts, or `null`. |
| `evidence_end` | One value per question: the second it ends, or `null`. |

```json
{
  "answers": [true, false, true],
  "evidence_start": [21.62, null, 43.5],
  "evidence_end": [26.24, null, 44.98]
}
```

All three lists are required, and that is the whole response — nothing else is
read.

**All three are matched to the questions by position.** Each must hold exactly
as many entries as you were sent, in the same order, so `evidence_start[3]` and
`evidence_end[3]` belong to `answers[3]`. A `true` should carry the timestamps of
the passage it was read from; a `false` has nothing to point at, so both of its
timestamps are `null`.

This is the part of the protocol that is easy to get wrong and expensive when
you do. A list of the wrong length — any of the three — cannot be matched up at
all, so the body does not parse and **every question about that conversation is
scored wrong**. Getting `evidence_start` wrong costs you the answers too, not
just the evidence: ten marks, not one.

`utils.validate_response` checks all of it for you before the response leaves
your server, including the cases the service will silently score zero rather
than complain about — a half-filled interval, or an end before its start.

Returning `null` for a question you answered `true` is legal. It scores nothing
for evidence, but it is far better than a malformed body, so a model that cannot
localize should still answer.

The service is lenient about what it will accept — it coerces `1`, `"yes"` and
`"true"` into `true`, and ignores keys it does not recognise rather than
rejecting them. Do not build on that. Send real JSON booleans, numbers and
nulls.

## Timing

Each request has a budget of **60 seconds** from the POST, and the requests are
sent strictly one at a time, in file order.

There is a second budget on top of that: the **whole attempt** gets 60 seconds
per conversation. Run over it and the conversations still queued are never sent,
and their questions are scored wrong. So 60 seconds is an average to stay under,
not a per-request allowance to spend in full — being slow on the early
conversations takes marks off the late ones.

One request is one whole conversation, so that budget has to cover the expensive
half once and the cheap half ten times: transcribe the audio, then answer ten
questions against the transcript you just made. There is nothing to cache
between requests — each conversation is sent once and never comes back.

Budget it deliberately. If transcription takes 40 seconds you have 20 left, or
about 2 seconds a question, and an answering model that wants longer than that
will time out the whole conversation rather than one question of it. Sizing the
ASR model against the answering model is a real trade-off here, and it is worth
measuring before the attempt rather than during it.

Asking your ASR for timestamps is not an extra pass over the audio — the
segment timings come out of the same transcription — so the evidence half of the
score costs you almost nothing here. What it does cost is the work of deciding
*which* segment answered the question.

**Five timeouts in a row ends the attempt.** An endpoint that has missed five
60-second budgets back to back is treated as unavailable, and the conversations
that were still queued are never sent — their questions are scored wrong, exactly
as if they had timed out too. Any reply at all clears the count, so it takes five
consecutive silences, not five slow requests scattered through the set.

There is no separate warm-up period. The first inference is usually the
slowest, so load and exercise your model at import time, before the attempt
starts.

## Scoring

Two things are measured: whether you answered the question, and whether you
found the passage that answers it. Your score is a weighted combination of the
two:

$$
Score = 0.4 \times Accuracy + 0.6 \times \overline{tIoU}
$$

So finding the evidence is the larger half. Perfect answers with no spans is not
a good score, and neither is tight localization on badly answered questions —
the shipped baseline answers `true` to everything and points at nothing, which
works out to `0.4 × 0.5 + 0.6 × 0 = 0.200`.

### Accuracy

$$
Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

One point per question, no partial credit, and no confidence to calibrate. Both
sets are exactly balanced between yes and no, so answering the same thing every
time earns `0.500` on this half and no more. An off-topic question is worth
exactly as much as a hard negative.

### Temporal IoU

Each span you return is compared with the annotated one by **temporal
intersection over union** — how much of the two of them overlap, as a fraction
of the stretch they cover together:

$$
tIoU = \frac{\text{overlap of the two spans}}{\text{total stretch they cover}}
$$

An exact match scores `1`. Spans that do not touch score `0`. So, against an
annotated passage running from `21.62` to `26.24`:

```text
predicted 21.62 - 26.24   overlap 4.62 / total 4.62   tIoU 1.000
predicted 22.00 - 27.00   overlap 4.24 / total 5.38   tIoU 0.788
predicted 24.00 - 34.00   overlap 2.24 / total 12.38  tIoU 0.181
predicted  0.00 - 120.0   overlap 4.62 / total 120.0  tIoU 0.038
predicted 40.00 - 45.00   overlap 0.00 / total 23.38  tIoU 0.000
```

Read the fourth line twice. Returning the whole conversation is not a hedge — it
scores about as badly as pointing at the wrong place entirely. Return the
passage, not the region it sits in.

Those per-question numbers are averaged into one, and the set they are averaged
over is **fixed by the annotations, not by your answers**: every question whose
true answer is yes counts, all 195 of them in the supplied data.

- Answer `no` to one of them and it scores `0` there. It does not drop out of
  the average — a missed positive costs you twice, once on each half.
- Return `null`, a malformed interval, or an end before its start, and it scores
  `0`. Nothing raises; it just earns nothing.
- Questions whose true answer is `no` are left out entirely, so a span you
  volunteer alongside a `no` can neither help you nor hurt you. There is no
  penalty for guessing, and no credit either.

A request that fails scores its questions wrong on both halves and is not
excused. A timeout, a non-2xx status, an unparseable body, a list of the wrong
length — `answers`, `evidence_start` or `evidence_end` — and an exception inside
your model all count the same as ten confident wrong answers with no evidence.

One failure mode does more than that. **Five consecutive timeouts and we stop
sending** — the remaining conversations are never delivered and their questions are
scored wrong. Only timeouts accumulate towards the five, and a single reply resets
the count: a 500, a body we cannot parse and a list of the wrong length are all
scored wrong, but they prove you are alive and put the counter back to zero. What
ends an attempt early is silence. The whole-attempt budget in the Timing section
above ends it the other way, without any silence at all.

That is the thing worth taking seriously about one request per conversation:
**a single dead request costs ten marks, not one** — over 5% of a validation
attempt. Since a coin toss is worth half a mark on average and an error is worth
nothing, catch everything and return a guess for every question.

## Validation and evaluation

Everything happens through [cases.nordicaicup.com](https://cases.nordicaicup.com) with the API
key your team was given.

**Verify** sends one conversation with its ten questions and checks the shape of
your reply. It is a format check, not a speed check. Passing it does not mean
you are fast enough.

**Validation** runs 190 questions over 19 conversations, so 19 requests. You can
only have one attempt going at a time, but you can validate as often as you
like.

**Evaluation** runs a different, larger set — 380 questions over 38
conversations — and you get **one completed attempt only**. That score is the
one you are judged on.


## Rules

**Your endpoint must answer without calling a cloud API.** Build your solution
with whatever helps — hosted models, paid APIs, anything at all — while you are
developing it. But when we call `/predict`, everything has to run on your own
machine. No hosted transcription service and no hosted LLM in the request path.

That makes local ASR the first thing to get working. `faster-whisper` is the
usual starting point and reads MP3 without a separate ffmpeg install;
`whisper.cpp` and `WhisperX` are the other common choices, the last of these if
you want speaker labels or tighter word-level timings. For the answering half, a
local instruction-tuned LLM over the transcript is the obvious baseline, and an
extractive QA model is the cheaper one.

**Keep the timings.** The obvious version of this throws them away by joining
the segments into one string, and then you have nothing to put in
`evidence_start`. Keep the segments:

```python
import tempfile
from faster_whisper import WhisperModel

MODEL = WhisperModel('large-v3', device='cuda', compute_type='float16')

def transcribe(audio_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(audio_bytes)
        f.flush()
        segments, _ = MODEL.transcribe(f.name, language='en')

    return [
        {'start': segment.start, 'end': segment.end, 'text': segment.text}
        for segment in segments
    ]
```

Call that once at the top of `predict`, then answer every question against the
one transcript, and you have the expensive half solved. What remains is the half
that actually separates the field: deciding whether a question is true of the
transcript, and which segments it was made true by. Those two questions have the
same answer, which is the useful thing about this case — the passage you would
cite to justify a yes is exactly the span you are asked to return, so a system
built to find its evidence answers better than one built to guess.

Joining neighbouring segments is often the right move; how far to take that is
worth measuring against the annotations in `question_train.csv` rather than
guessing.

## Test locally

`local_evaluator.py` replays the supplied conversations through your endpoint
using the same payloads, the same ordering and the same failure rules as the
competition.

```cmd
python local_evaluator.py                              # score the 390 questions
python local_evaluator.py --oracle                     # score the ground truth
python local_evaluator.py --verbose                    # a line per question
python local_evaluator.py --url http://host:9054/predict
```

Ignore the headline number and read the breakdowns underneath it.

**Accuracy by question type** tells you what kind of wrong you are. The shipped
baseline prints this, and it is the shape you are trying to get away from:

```text
Accuracy by question type
  positive             1.000  (195/195)
  hard_negative        0.000  (0/142)
  off_topic            0.000  (0/53)
```

Perfect on positives and zero on everything else means a model biased towards
yes. The reverse means one that cannot find anything. A real system has to move
`hard_negative` without giving up `positive`, and that trade is the whole
challenge.

**Evidence localization** is the other half, and the baseline has none of it:

```text
Evidence localization
  mean tIoU                0.000  (over 195 annotated yes questions)
  no span returned         195
  tIoU when answered yes   0.000  (diagnostic, n=195, not scored)
```

`no span returned` counts the annotated yes questions you sent nothing back
for — the first number to get off the floor. The third line is a diagnostic
only and is **not** part of your score: it is the mean tIoU over just the ones
you actually answered yes, which separates two different failures. A high
diagnostic with a low mean tIoU means you localize well when you notice, and
mostly do not notice. Both low means the localization itself needs work.

The weights are in `local_evaluator.py` as `ACCURACY_WEIGHT` and `TIOU_WEIGHT`
if you want to read the arithmetic.

**Round trip** is your per-conversation latency against the 60-second budget,
with the per-question cost derived from it. Read the worst case, not the mean:
it takes one conversation over budget to lose ten marks.

The five-timeout abort is implemented here too, and with 39 conversations it can
fire locally: a server that is consistently over budget will end the run early
here just as it would in a real attempt. The `timeouts` line in the report is
worth a look either way — it separates "answered too late" from "answered
wrongly", which the accuracy number alone cannot.

## Serve your endpoint

Serve your endpoint locally and test that everything starts without errors:

```cmd
cd medical-appointment
python api.py
```

Open a browser and navigate to http://localhost:9054. You should see a message
stating that the endpoint is running. Feel free to change the `HOST` and `PORT`
settings in `api.py`.

There is also a `Dockerfile` if you would rather containerise it:

```cmd
docker build -t medical-appointment .
docker run -p 9054:9054 medical-appointment
```

### Make your endpoint reachable

The evaluation service has to be able to call you from the internet.

- **Cloud instance** — run the same steps on a VM from UCloud, Azure, GCP or
  AWS and open the port. This is the path we would recommend.
- **Local machine** — you need the port forwarded to your machine, which
  depends on your router and is often not possible on a university network.

**Get a server up and reachable early in the competition.** If something is
wrong with your deployment, you want to find out on day one and not an hour
before the deadline.

**The URL you submit is used exactly as given, path included.** With the
`api.py` in this folder that means `http://<your-host>:9054/predict`, not just
the host.

## OBS

Things that quietly cost people points:

- **One failure costs ten questions.** A request is a whole conversation, so a
  single timeout or exception loses all ten of its marks — over 5% of a
  validation attempt.
- **Five timeouts in a row costs the whole tail.** Once your endpoint has gone
  silent five times running we stop sending, so a server that dies mid-attempt
  loses every conversation after it — not one. Return a guess quickly rather than
  the right answer eventually.
- **Answer in the order you were asked.** All three lists are matched to the
  questions by position, and a list of the wrong length scores the whole
  conversation wrong rather than the part you got wrong.
- **`evidence_start` and `evidence_end` are not optional.** They are required
  fields, one value per question, `null` where the answer is no. Omitting one,
  or sending nine values for ten questions, makes the body unparseable and
  loses all ten marks — the answers included.
- **Point at the passage, not the conversation.** A span covering the whole
  clip scores about as badly as pointing at the wrong place.
- **A missed positive costs you twice.** It is a wrong answer, and it is also a
  zero on the evidence half, which averages over every annotated yes whether
  you found it or not.
- **Don't let your model raise.** An exception means no response, which means
  every question about that conversation is scored wrong. Catch, log, and return
  a guess — it is worth half a mark on average, and an error is worth none.
- **`audio_base64` has no `data:` prefix.** It is plain base64 of the MP3
  bytes.
- **Hard negatives are most of the no's.** They differ from a true statement by
  a dose, a drug or a single word. Anything that scores by topical similarity
  answers yes to all of them and lands on the floor.
- **Don't key off question syntax.** Some questions are tag questions rather
  than plain interrogatives, and the phrasing tells you nothing about the
  answer.
- **Warm your model up at import time.** The first inference is the slowest and
  there is no grace period for it.
- **Bodies are megabytes, not kilobytes.** If you put a proxy in front of your
  server, check its request size limit before the attempt rather than after.
- **You get one evaluation attempt.** Validate first, as often as you like.
