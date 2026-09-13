# Fraud Decision Policy

## Governing principle
The decision is set by the deterministic engine or a human — never by a model.
The ML score and the LLM analysis are inputs and explanations only.

## Risk bands (0-100 rule score)
- low (<= 30): eligible for auto-approve when below the high-value threshold.
- grey (31-70): route to investigation; a human decides after LLM analysis.
- high (> 70): route to investigation unless a deterministic auto-decline applies.

## Deterministic auto-decline
- Any confirmed blocklist hit (card, device, or IP).
- Extreme velocity (at or above three times the velocity limit).

## Investigation outcomes (analyst chooses)
- approve: profile consistent with the cardholder; no unresolved risk.
- step_up: identity in doubt but resolvable with an OTP challenge.
- decline: risk outweighs value; block the transaction.
- escalate: needs senior review or out-of-band verification.

## Step-up guidance
Prefer step_up over decline when the only issue is identity confidence and the
OTP channel is trustworthy. Do not rely on OTP when SIM-swap or OTP interception
is suspected — escalate instead.
