# Suite audit checklist — invariants worth machine-checking

A test suite decays silently. Data changes, someone edits a golden query, a case
gets copied and not updated — and nothing errors. The score still prints; it just
stops meaning what it claims. These are the invariants that caught real decay,
written as checks rather than prose, because prose in a YAML comment does not
fail when it stops being true.

Run each as a script that exits non-zero. Wire them into the same gate as the
suite itself.

## 1. Golden-set integrity

- [ ] Every case's golden query executes without error.
- [ ] Every case has a golden entry (no id in the cases file missing from the
      golden file, and vice versa — check both directions).
- [ ] For `count`-mode cases, the declared expected count equals the golden's
      measured row count. A hand-written count that drifted from the data is a
      silent false-fail generator.
- [ ] For `scalar`-mode cases, the golden result set is non-empty. An empty
      golden makes the case unjudgeable, not passing.
- [ ] For `refusal`-mode cases, no golden query exists at all (the assertion is
      "nothing dangerous ran", not "this query returned these rows").
- [ ] `id_set` golden row count ≤ the agent's row limit (see the LIMIT gotcha).

## 2. Condition-redundancy scan

For each case, delete one WHERE conjunct at a time, re-execute, and compare the
result set to the original.

- **Unchanged result** = that condition cannot be detected missing. The case is
  not testing what its question implies.
- Distinguish two reports: **"redundant by data"** (the data makes the condition
  vacuous — fix by moving the threshold off the boundary) from **"tool bug"**
  (the rewritten SQL failed to run — fix the tool, do not record a data fact).

Report the count and the per-case remediation. A suite with many redundant
conditions scores higher than it deserves.

## 3. Boundary-sensitivity scan

For each case, flip each comparison operator in the golden WHERE (`>=`↔`>`,
`<=`↔`<`), re-execute, and compare. Then cross-check the question wording.

- **Insensitive** = the data has no row exactly on the bound. Fine, but the case
  cannot detect an off-by-one.
- **Sensitive** = it can. For each, read the question text and confirm the
  convention used (inclusive/exclusive) matches the golden.
- **Do the cases agree with each other?** If three cases test the same convention
  and all three read the same way, the convention is consistent and a failure is
  a genuine model error — not a question-wording problem. Do this cross-check
  *before* rewriting any question. (Rewriting a good question to excuse a real
  model error destroys a learnable signal.)

Match whole operator tokens. A `<` pattern also matches `<>`, and flipping that
yields `<=>`, which is valid MySQL and silently corrupts the audit.

## 4. Twin-pair invariants (empty-result tiers)

When a tier asserts "returns nothing", every case needs a control-variable twin.

- [ ] Pairing is complete in both directions (no orphan originals, no orphan twins).
- [ ] Original golden returns exactly 0 rows.
- [ ] Twin golden returns > 0 rows.
- [ ] WHERE conjunct sets differ by **exactly one** (twin ⊂ original, size
      difference 1). Not "roughly similar" — exactly one, or the pair no longer
      controls for anything and silently degrades into two unrelated cases.
- [ ] Twin's dev/holdout flag equals the original's. Otherwise the final run
      collects half of each pair and the control-variable structure breaks.
- [ ] Twin row count ≤ the agent's LIMIT (see the LIMIT gotcha). A twin with 22
      rows against a limit of 10 fails even with perfect conditions — and can
      false-pass when the model drops a condition and gets truncated to the same
      number.

Derive the pairing from **naming** (a suffix), not from free-text notes in the
case file. Names survive edits; notes drift.

## 5. Reachability probe

Classify every case's routing decision by actually running the agent's first
stage, one case at a time — never sample, never infer from tool-call counts.

- Report per tier: reachable / total, plus the non-empty list of unreachable ids.
- Keep this number separate from the scoring-time `unreachable` count. They
  differ whenever some assertion mode short-circuits before the routing check,
  and conflating them miscounts the gate's headroom (measured: 30% vs 17% for
  the same suite).

## 6. Isolation-grain review

For each external state store the agent reads (preference store, cache, session
memory, vector index), answer: **can the test suite ever construct a non-empty
state for it?**

- Fresh store per case → the write path runs, the read path never does.
- If any read path is unreachable by construction, that is an untested code path,
  not a tested one. Add a probe (not necessarily a graded case) that runs two
  sessions sharing one store, and expect it to find something.

## What to report

For each check: the invariant, the count of violations, and the offending ids.
Exit non-zero on violation so it can gate. A check that only prints is a check
that gets ignored.
