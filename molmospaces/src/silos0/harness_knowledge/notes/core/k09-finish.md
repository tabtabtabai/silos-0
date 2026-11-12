---
id: K09
title: Check every part of the instruction before finishing
layer: core
status: active
excludes: [knowledge_v03]
priority: 9
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "False done stays near zero without this rule for every model in use."
---
Keep a short checklist of the instruction's parts (each object, each destination, each condition) and mark a part done only on evidence. Declare the task finished only when a fresh view confirms every part; if a part is uncertain and time remains, measure or look again first. When time is short, finish the part closest to completion first.
Why: partial completion scores little, and an early finish ends the attempt.
