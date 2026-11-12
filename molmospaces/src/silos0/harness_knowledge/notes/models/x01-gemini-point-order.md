---
id: X01
title: Image points are [x, y]
layer: model
excludes: [locator]
models: ["gemini-*"]
status: active
priority: 1
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "Rejected locates for Gemini models stop being dominated by swapped coordinates."
---
Image points in this harness are [x, y]: x across the image from the left, y down from the top. This differs from the [y, x] order you may use elsewhere.
Why: a swapped point lands on a different object or gets rejected.
