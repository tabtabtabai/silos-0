---
id: K04
title: A finished command is not a finished effect
layer: core
status: active
priority: 4
evidence:
  - "Manipulation guidance used by the submitted policy."
confidence: measured
falsify: "Models already verify contact steps from measured state without the rule (check in traces)."
---
Judge every contact step (grasping, placing, inserting, pushing) by its effect: measured state first (gripper closure against fully closed, load, contact events, how far the object moved), then a fresh image from the view that best shows the contact. Positions measured before a contact are stale after it; measure again whatever may have moved.
Why: tools report what was commanded; only measurements and fresh views show what happened.
