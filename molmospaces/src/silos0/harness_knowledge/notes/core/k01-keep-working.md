---
id: K01
title: Keep working until the environment ends the episode
layer: core
status: active
priority: 1
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "Episodes that end with time left stop being a top loss in the behaviour audit, or the rule causes measurable thrashing (more calls, lower success) on the validation split."
---
The environment decides success and when the episode ends. Until then, keep making progress, with a tool call in every reply. When unsure what to do, take the cheapest action that gives new information (measure, look from another view, a small probing move).
Why: unused robot time is a lost attempt; stopping early has cost more episodes than wrong actions.
