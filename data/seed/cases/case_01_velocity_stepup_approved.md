# Case 01 — Velocity burst, step-up passed
Date: 2026-03-04
Signals: 6 transactions on one card within 40 minutes at a gift-card merchant; domestic geo; known non-proxy device.
Rule score: 42 (grey). Analyst view: burst pattern but device and geo consistent with the cardholder.
Action taken: step_up (OTP to registered email). Cardholder entered the correct OTP within 2 minutes.
Outcome: APPROVED. No chargeback filed in the following 90 days.
Lesson: A velocity burst with an otherwise clean profile is a step-up candidate, not an auto-decline.
