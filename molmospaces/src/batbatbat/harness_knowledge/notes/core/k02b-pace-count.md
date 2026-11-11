---
id: K02b
title: Pace calls when no budget meter is shown
layer: core
status: active
priority: 2
excludes: [budget_meter]
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "Same as K02."
---
You have about {max_calls} tool calls for the whole episode and the clock keeps running while the robot moves. Keep a running count, plan the remaining steps to fit with one retry each, and prefer tools that do more per call when the count runs low.
Why: running out mid-task scores nothing, while a slightly less careful finish often succeeds.
