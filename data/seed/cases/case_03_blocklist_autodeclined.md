# Case 03 — Blocklisted card, auto-declined
Date: 2026-01-19
Signals: Card present on the confirmed-fraud blocklist (fraud-ring attribution); foreign geo; crypto merchant.
Rule score: 100 (high, forced by blocklist).
Action taken: auto_decline by the deterministic engine — no human review required.
Outcome: DECLINED, SAR filed automatically.
Lesson: A confirmed blocklist hit is deterministic and never routed to investigation.
