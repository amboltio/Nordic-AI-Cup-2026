# Medical appointment

![Medical Appointment AI image](../images/Medical_Appointment_Image.png)

A patient sees their doctor. The consultation is recorded, and afterwards
somebody wants to know what was actually agreed — which vaccine was given, what
dose was prescribed, whether the results were good. Your job is to answer that
question from the audio alone.

Each request carries one conversation and the ten yes/no questions asked about
it. You return ten booleans. No transcript is provided; the speech recognition
is yours to do.

The catch is that the wrong answers are not nonsense. Most of them are
near-misses — the right drug at the wrong dose, the right course at the wrong
length — so a system that hears roughly what a conversation was about, without
hearing it precisely, scores no better than a coin toss.

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
`true` to everything and scores exactly `0.500` — that is the floor, not a
start. It is there to prove the plumbing works, not to compete.

Check that the harness and the data agree with each other at any time:

```cmd
python local_evaluator.py --oracle
```

That feeds the ground truth in as predictions and should print `1.000`. If it
does, a low score is your model, not your setup.

### What is in this folder

| File | What it is |
| --- | --- |
| `api.py` | The FastAPI server the evaluator calls. You probably will not change it. |
| `example.py` | The baseline. **This is the file to replace.** |
| `dtos.py` | The request and response models. |
| `utils.py` | Base64 decoding, MP3 duration, response validation, sample loading. |
| `local_evaluator.py` | Replays the supplied questions through your endpoint and scores it. |
| `requirements.txt` | Dependencies. Loose pins, so they will not fight your ASR stack. |
| `Dockerfile` | If you would rather containerise the server. |
| `data/` | 39 training samples, including audio files and questions. |

## About the challenge

There are two halves to this and you can lose the points in either. The first
is hearing the conversation: names, numbers and units, spoken at conversational
speed. The second is deciding what the question is really asking, which is
where most of the difficulty lives — see below.

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
└── questions_sample.csv               (390 questions, ten per conversation)
```

The CSV carries `question_id`, `transcript_id`, `question`, `answer`, `label`
(`1` = yes, `0` = no), `question_type`. The `transcript_id` names the
conversation — `sample_17` is `data/audio/conversation_sample_17.mp3`.


## Your goal

Given one conversation and the ten questions about it, return the right ten
booleans. Every question is worth the same.

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

```json
{
  "answers": [true, false, true]
}
```

That is the whole response. Nothing else is read.

**The answers are matched to the questions by position.** The list must hold
exactly as many entries as you were sent, in the same order. This is the one
thing in the protocol that is easy to get wrong and expensive when you do: a
list of the wrong length is not partially credited, and every question about
that conversation is scored wrong. `utils.validate_response` checks it for you
before the response leaves your server.

The service is lenient about what it will accept — it coerces `1`, `"yes"` and
`"true"` into `true`, and ignores keys it does not recognise rather than
rejecting them. Do not build on that. Send JSON booleans under `answers`.

## Timing

Each request has a budget of **90 seconds** from the POST, and the requests are
sent strictly one at a time, in file order.

One request is one whole conversation, so that budget has to cover the expensive
half once and the cheap half ten times: transcribe the audio, then answer ten
questions against the transcript you just made. There is nothing to cache
between requests — each conversation is sent once and never comes back.

Budget it deliberately. If transcription takes 60 seconds you have 30 left, or
about 3 seconds a question, and an answering model that wants longer than that
will time out the whole conversation rather than one question of it. Sizing the
ASR model against the answering model is a real trade-off here, and it is worth
measuring before the attempt rather than during it.

**Five timeouts in a row ends the attempt.** An endpoint that has missed five
90-second budgets back to back is treated as unavailable, and the conversations
that were still queued are never sent — their questions are scored wrong, exactly
as if they had timed out too. Any reply at all clears the count, so it takes five
consecutive silences, not five slow requests scattered through the set.

There is no separate warm-up period. The first inference is usually the
slowest, so load and exercise your model at import time, before the attempt
starts.

## Scoring

Your score is **accuracy** over every question in the set.

$$
Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

One point per question, no partial credit, and no confidence to calibrate. Both
sets are exactly balanced between yes and no, so answering the same thing every
time earns `0.500` and no more. An off-topic question is worth exactly as much
as a hard negative.

A request that fails scores its questions wrong and is not excused. A timeout, a
non-2xx status, an unparseable body, an `answers` list of the wrong length and an
exception inside your model all count the same as ten confident wrong answers.

One failure mode does more than that. **Five consecutive timeouts and we stop
sending** — the remaining conversations are never delivered and their questions are
scored wrong. Only timeouts accumulate towards the five, and a single reply resets
the count: a 500, a body we cannot parse and a wrong-length `answers` list are all
scored wrong, but they prove you are alive and put the counter back to zero. What
ends an attempt early is silence.

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
you want speaker labels. For the answering half, a local instruction-tuned LLM
over the transcript is the obvious baseline, and an extractive QA model is the
cheaper one.

Dropping ASR into `example.py` looks about like this:

```python
import tempfile
from faster_whisper import WhisperModel

MODEL = WhisperModel('large-v3', device='cuda', compute_type='float16')

def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(audio_bytes)
        f.flush()
        segments, _ = MODEL.transcribe(f.name, language='en')
    return ' '.join(segment.text for segment in segments)
```

Call that once at the top of `predict`, then answer every question against the
one transcript, and you have the expensive half solved. What remains — deciding
whether a question is true of the transcript — is the half that actually
separates the field.

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

Ignore the headline number and read the two breakdowns underneath it.

**Accuracy by question type** tells you what kind of wrong you are. The shipped
baseline prints this, and it is the shape you are trying to get away from:

```text
Accuracy by question type
  positive             1.000  (10/10)
  hard_negative        0.000  (0/7)
  off_topic            0.000  (0/3)
```

Perfect on positives and zero on everything else means a model biased towards
yes. The reverse means one that cannot find anything. A real system has to move
`hard_negative` without giving up `positive`, and that trade is the whole
challenge.

**Round trip** is your per-conversation latency against the 90-second budget,
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
- **Answer in the order you were asked.** The answers are matched to the
  questions by position, and a list of the wrong length scores the whole
  conversation wrong rather than the part you got wrong.
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
