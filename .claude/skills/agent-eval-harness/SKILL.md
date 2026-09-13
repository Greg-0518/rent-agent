---
name: agent-eval-harness
description: Use when building, debugging, or trusting an evaluation harness for an LLM agent — writing pass/fail assertions, deciding whether a failure is the model's fault or the product's, or investigating a case that passes when it should not.
metadata:
  provenance: self-improving-skills
  origin: distilled
---

# Building an eval harness for an LLM agent

## When this applies

Any project where an agent's output is scored by a test suite: text2sql / NL2SQL,
tool-using agents, multi-node graphs with intent routing. Triggers: designing the
assertion layer; a case fails and you must decide who to blame; a suite reports a
green or near-green run that you do not believe.

The techniques below come from building one such harness (a 89-case text2sql
suite over a LangGraph agent). Every gotcha in the last section cost real time.

## The technique

### 1. Assert on results, not on the agent's text

Re-execute the query the agent produced and compare **result sets**, never query
strings. The same question has many equivalent phrasings; text comparison fails
them systematically and the score becomes meaningless.

Compare on the **intersection of column names**, sorted, as a multiset — the agent
will not project exactly the columns you expected, and demanding that is an
unreasonable bar. Count occurrences, not rows-in-order, unless ordering is the
thing under test.

But *which* query you re-execute is a separate decision, and getting it wrong
fails correct agents — see §10.

### 2. Four outcomes, not two

The single highest-value decision. `pass` / `fail` is not enough.

| Outcome | Means | Counts toward model score? |
|---|---|---|
| `pass` | result set / value matches | yes |
| `fail` | definitely wrong; attach an attribution | yes |
| `unverifiable` | reached the code under test, but the assertion cannot judge | yes — deliberately |
| `unreachable` | never reached the code under test | **no** |

- `unreachable` must not be `fail`: if routing sends a case down a path that never
  touches the component under test, the case cannot go green no matter what the
  model does. Twenty permanently-red cases turn the gate into a red-always gate,
  and a red-always gate is no gate.
- `unreachable` must not be `skip` either: then "routing swallowed everything"
  disguises itself as a green, empty run. Count it, display it, gate it (see 6).
- `unverifiable` stays **in** the denominator. If you drop everything you cannot
  judge, the score ratchets upward on its own — you are paying yourself a bonus.

### 3. Attribution is the expensive part — budget for it

Splitting "it failed" into *why* is where the real work is, and getting it wrong
sends you to fix the wrong module. Separate **model-capability** codes from
**non-model** codes, and never mix them in one statistic:

```
model:     SQL_SYNTAX_ERROR · SCHEMA_MISUNDERSTOOD · CONDITION_MAPPED_WRONG
           BOUNDARY_OFF_BY_ONE · RESULT_TRANSFORM_WRONG · NO_QUERY_ATTEMPT
not model: ROUTED_AWAY (product gap) · GUARD_BYPASS (safety layer)
```

**Split a code whenever the two halves are fixed in different places.** A single
`CONDITION_MAPPED_WRONG` was hiding two: "mapped the condition to the wrong
value/column" (fix the schema description) and "mapped it right but got the
boundary inclusivity wrong" — `>=` written as `>` (fix the prompt's semantics for
the language's interval words). Pooled, the attribution table sent you to the
schema docs for a bug that lived in the prompt.

Mechanise the detector so it cannot drift: flag it when *flipping one comparison
operator in the golden query reproduces the agent's result set exactly*. That is
evidence the agent wrote the golden's conditions with the boundary inverted, not
guessed. Zero false positives across a 70-case suite.

The four-way split of "the model produced no result set" is the one that matters
most, because all four look identical in a trace (no rows) and have four different
fixes: routed away (change routing) · never called a tool (change the prompt) ·
called tools but wrote no executable query (change the schema description) ·
query errored (change the schema or the dialect hint).

### 4. Read ground truth from graph/agent state, not from the trace

Do not infer "the agent was routed away" from "it called zero tools" — a model
that declines to query also calls zero tools. Read the routing decision out of
the agent's own state (e.g. the classified intent written by the first node) and
branch on that value.

This exact conflation mis-scored an entire tier (12 cases) as a model defect when
it was a product gap. The headline number moved 60% → 65% with **no change to the
agent** — only to the attribution.

### 5. Isolate per case, not per session

Give every case a fresh store/checkpointer/session. That is correct — but then
ask what state it makes **unreachable**. Fresh-per-case isolation means any
"the store already has this user's data" branch is never entered, silently.

Isolation grain should be **the case**, not **the session**: allow a single case
to contain several sessions against one shared store. When you find this gap,
expect a crash in it — untested second-session paths rot. (In the suite above,
the second session died with `KeyError` on a preference key the write path had
omitted via `exclude_none=True`; a `.get()` on the read side was the fix.)

### 6. Gate the false green

Ratio caps, enforced as a **session failure**, not a warning in a report:

- unverifiable ≤ ~20% — above that the score is not trustworthy enough to quote;
- unreachable ≤ ~40% — this is the anti-"empty green run" gate;
- hard fail if **zero** cases reached the component under test.

Record the measured baseline next to each cap. And keep the two "unreachable"
numbers distinct if you have both: a **reachability probe** (can this case ever
reach text2sql?) and the **scoring** count differ whenever a mode short-circuits
before the routing check. Quoting one as the other miscounts the headroom by ~2x.

### 7. Empty-result cases need twins

"Returns nothing" is not a signal — an over-constrained query and a correct query
both return nothing. Pair every empty-result case with a **twin differing by
exactly one condition** that yields a non-empty result. Then "I forgot that
condition" is detectable instead of invisible.

Twin invariants, machine-checked (see `references/suite-audit-checklist.md`):
original returns 0 rows · twin returns > 0 · WHERE conjunct sets differ by
exactly one · same dev/holdout split · **twin row count ≤ the agent's LIMIT**.

### 8. Group failures by cause before reading them

A report that says "7 failures" is not actionable. Collapse each failure to a
short **reason signature** (prefer the attribution code; otherwise the reason text
cut before the first `(`/`[`/`:` so per-case numbers do not split the group) and
print the case ids grouped by signature. Patterns only become visible in the
grouped view — e.g. five failures across two tiers sharing one shape ("the model
widens a word into a chain of synonyms, recall over precision").

Then make the drill-down one command: `--show <outcome|code|id-prefix> --limit N`.

### 9. Verify your own diagnoses against the data

When a case fails, resist the first plausible story. Two cheap checks:

- **Is the suite internally consistent?** Before calling a question ambiguous,
  find every other case testing the same convention and check they agree. Three
  boundary cases all read "inclusive" proved the question was fine and the model
  was genuinely wrong — the opposite of the initial call.
- **Is the harness itself supplying the input?** Scripted answers to interrupts
  and scripted context are built from case metadata. If that metadata disagrees
  with the question text, the agent receives a condition nobody asked for, and
  the case blames the model for it. This stayed hidden for ten rounds because an
  empty result intersects anything and looks identical.

### 10. Score what the question asked for, not the agent's last query

A tool-using agent in a loop sees each result and can re-query. `sql_executed[-1]`
is the obvious thing to assert on and it is often wrong.

Observed shape (12 cases, ~20% of a suite): the agent **queries strictly first**,
gets the correct—sometimes exactly golden—result set, then, seeing it is short of
the requested count, widens the conditions and **labels the widened rows as a
separate reference section in its answer text** ("matching your criteria: 1 ·
⚠ other listings — pet policy not stated, confirm with the landlord: 9"). Scoring
only the last query fails it for adding a clearly-labelled appendix.

Fix: run the assertion against **every** executed query and pass if any matches.

- Iterate newest-first so the last query is still the primary verdict.
- Record `matched_sql_index` and `n_sql_executed` in the detail. Matching at
  index 0 means "it queried correctly straight away"; a late index with
  non-zero `sql_errors`/retry means "it flailed until something stuck". Those are
  different agents, and you must be able to tell them apart later — this is the
  price of the leniency, so pay it in instrumentation, not in prose.
- Structural check before you believe the widening is benign: every first query
  `executed`, zero SQL errors, zero retries. If any first query errored, you are
  looking at retry-after-failure and the rule is not the same rule.

Also handled: "returned an empty set, then widened". If the question wanted empty,
the strict query *is* the answer, so this becomes a `pass`, not `unverifiable` —
and the `unverifiable` branch you wrote for it becomes unreachable. Delete it;
a branch that can never fire misleads the next reader.

### 11. Before calling a failure a product gap, force the component and re-run

`unreachable` tells you a case never reached the component under test. It does
**not** tell you whether the product *could* answer it — and those two have
opposite fixes: change the router, or build the missing path. The baseline looks
identical either way, because in both the agent answers wrong.

Cheap decisive experiment: build a graph with **only the classification node
replaced** by one that returns the intent under test, reusing the production
nodes and edges for everything else. Then the difference between this run and the
real one is attributable to **routing alone**.

Outcome reading: 12/12 correct → purely a routing/prompt problem, no code needed;
some still wrong → a genuine capability or product gap.

Run this before proposing any "the product doesn't support this" conclusion. It
converts a design argument into a measurement, and it took one script.

### 12. Re-judge saved traces offline to isolate the asserter's effect

Once the assertion layer changes, "did the score move because of the asserter or
because of the agent?" is unanswerable from a fresh run — you changed both, and
the agent is stochastic on top. Two variables, one number.

So: persist enough per case to re-run the asserter later (final output, executed
queries, SQL errors, guard rejections, the routing decision, tool calls), then
re-judge from disk. Seconds instead of minutes, no API cost, and perfectly
reproducible — same trace in, same verdict out.

Use it to see the full blast radius of an asserted-layer change *before* spending
a real run, and re-run only to confirm. Caveat to state plainly: offline re-judging
can only show asserter effects. It can never validate a prompt or routing change.
Compare old and new `(outcome, attribution)` per case and print the diff — a
change that moves 10 cases should be inspected case by case, not trusted.

When you also change **ground truth** (a golden set), diff the regenerated
artifact against the old one and show that only the intended entries moved.
"Regenerated the goldens" is not a safe sentence on its own.

### 13. Judge the holdout set by coverage, not by proportion

A held-out set exists so one number can be quoted uncontaminated. Its adequacy
criterion is **every case type represented**, not "distributed like the full
suite".

So when you review the split, measure it as a coverage matrix over the axes that
define a type (`tier` × `assertion mode` × special shape: twin pair, refusal
case, boundary wording, aggregate) and ask *"is anything missing?"* — not *"is the
mix right?"* A tier that is over-weighted (e.g. the twin-heavy tier at 30% vs the
suite's 22%) is a *proportion* complaint and dissolves under the coverage reading.
Do not raise proportion objections as blockers.

Two things to still enforce mechanically:

- **Linked cases must land on the same side.** A twin pair or any other
  one-condition-apart control must share its holdout flag, or the closing run has
  only half the pair and the control structure is gone. Assert this, don't trust it.
- **Say how dirty the holdout already is.** Auditing tools read whole-suite
  question text and goldens; verifying a router fix may run the holdout too. The
  honest axis split is: clean for *"does the agent generalise"* (nothing was tuned
  against it), dirty for *"is my case design good"* (you have read the questions).
  Note which one your closing number is allowed to answer.

## Gotchas

- **The scorecard does not contain the harness's own test failures.** A report
  generated from saved traces only counts *case* verdicts. A red assertion-layer
  self-test is invisible in it — so a run can print a beautiful 97.1% while
  `pytest` exits non-zero. **Always read pytest's own summary line, not just the
  report.** One such red test survived a full round undetected this way.
- **Deleting a branch orphans its self-test, and nothing links the two.** When you
  remove an assertion branch, the test asserting that branch's behaviour lives in
  a different file and stays green-looking-but-failing. Change both in the same
  commit. When you do, **flip the assertion to the new behaviour and add a control
  case** — deleting the test instead quietly removes the guard on whatever leniency
  you just introduced.
- **A ±1 membership swap between runs is noise, not news.** If run A fails
  {X, Y} and run B fails {X, Z} with the same count, do not report "Z is fixed" or
  "Y is a regression" — especially when the change you shipped could not touch the
  failing step. Under sampling, single-run failure *lists* are symptoms; only a
  distribution over several runs is a trend. State the denominator you are
  comparing and, if it moved, say which cases changed sides.
- **A LIMIT silently destroys discriminating power.** If the agent appends
  `LIMIT k` and the assertion compares row counts, a golden set of k+n rows can
  never match — and worse, a wrong query truncated to k can match a correct one.
  Keep golden row counts within the limit.
- **Never judge by a character in the output text.** Using `"无" in answer` to
  detect "no results" fails on 没有 / 暂无 / 未找到, which are all more common.
  Use structure (row counts of the queries actually executed).
- **Check both sides of a comparison operator before "fixing" a boundary.** A
  regex for `<` will also match `<>` (SQL not-equals); flip it and you get `<=>`
  (null-safe equals), which is valid SQL and produces silent false positives.
  Match whole operator tokens: `<=>|<>|!=|>=|<=|>|<`.
- **A "where clause" helper that returns the offset *after* the `WHERE` keyword
  will produce `WHERE WHERE`** if you prepend it again. Return a distinct "SQL
  failed to run" channel from your tools so tool bugs do not masquerade as data
  facts — a broken rewrite that errors out looks exactly like "still empty".
- **A plausible story about the agent's motive is not evidence.** The shape "it
  queried correctly, then widened" reads as *fudging the numbers to hit a quota*
  and produces a confident, wrong diagnosis ("fix the prompt, forbid widening").
  Reading the agent's actual answer text killed it in one pass: the answer named
  the strict count first and put the widened rows in a labelled appendix. Before
  writing a root cause about *why* the agent did something, read what it said.
  The correct story was "my assertion scored the wrong query" — a harness defect
  wearing the costume of a model defect.
- **Don't ship a widened golden until you've checked it still discriminates.**
  A loosened expected set can quietly become a free pass. Write down which
  *wrong* behaviours it must still fail (returning the wider-but-incoherent set,
  ignoring the condition entirely) and confirm each one does.
- **The score is not the deliverable; the attribution table is.** A run reporting
  "62.9% pass" is a number. "7 failures, all one attribution, 5 in one tier,
  one shared shape" is a work item.
- On Windows, `Get-Content` reads UTF-8 JSONL as ANSI and mangles non-ASCII.
  Read with an explicit `encoding="utf-8"` or you will not be able to tell
  corruption from a real data problem.
