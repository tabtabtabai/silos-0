---
id: K02
title: Pace calls and robot time against what is left
layer: core
status: active
priority: 2
requires: [budget_meter]
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "With the meter shown, cap-limited episodes do not fall, or success falls on any task family of the validation split."
---
Each result reports the calls and robot seconds you have left. Before each sub-goal, check that what remains covers the rest of the task plus one retry per remaining step. When it does not, use the tool that does the most per call, shorten motions and skip optional looks, but keep the checks on contact steps.
Why: running out mid-task scores nothing, while a slightly less careful finish often succeeds.
