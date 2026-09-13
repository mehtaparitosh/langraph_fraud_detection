# FraudGraph — one-page talk track

A ~5-minute live narration. The through-line: **models advise, the engine and
the human decide.**

---

### 1. The problem (30s)
Fraud decisions have to be **fast, explainable, and defensible**. A payments
team can't tell a regulator "the model declined it" — they need a reason they
can audit, and a human in the loop for the hard calls. So the design question
isn't "how good is the model?" — it's **"who is allowed to decide, and can you
prove it?"**

### 2. The governing rule (20s)
> The LLM never authorizes a decision.

The deterministic engine or a human sets `decision`. The ML score and the LLM
are **inputs and explanations only**. This isn't a convention — a guard in the
graph **raises** if any LLM node tries to write `decision`. Point at the "who
has authority" diagram in the README.

### 3. Walk the four demos (2–3 min)
Run each and narrate the path lighting up:

- **clean** → scores 0 → `auto_approve`. *"Deterministic, no LLM, sub-millisecond.
  Most traffic never touches a model."*
- **blocklist** → blocklist hit forces 100 → `auto_decline` → SAR. *"Certain-bad is
  a deterministic decline. The LLM writes the SAR narrative, grounded strictly in
  evidence — but it doesn't decide."*
- **velocity** → grey band → `investigate`. Watch the sub-steps: the LLM calls the
  evidence tools, retrieves similar past cases, and **recommends** step-up. Then it
  **pauses for consent**. *"The model proposes; I decide."* Click Approve — the
  result is badged **MANUAL** and notes I overrode the LLM's recommendation.
- **high-value** → clean signals but the amount alone routes it to investigate →
  choose **Step-up** → real OTP email → enter the code → approved. *"Value forces a
  human check even when the score is low."*

Key beat: on high-value the **ML model screams fraud (score ~1.0)**, yet the
engine keeps it low and sends it to a human. That's the principle made concrete.

### 4. Where the LLM earns its place (20s)
Not in the decision — in the **toil**: reading the evidence, finding precedent in
past cases and policy, drafting a readable analyst summary and a five-W's SAR
narrative. It compresses a 10-minute manual write-up into seconds, and a human
still signs off.

### 5. The human gate (20s)
`interrupt()` suspends the run to disk and resumes it when the analyst responds —
consent, or a step-up OTP. The OTP send is idempotent, so resuming never
re-sends. Escalation fires a formatted email to the team. Every decision is
badged automatic vs. manual, with the reason codes and the SAR attached.

### 6. The 60-second AWS bridge (45s)
Same graph, managed services, governance preserved:
- Rule engine + router → **Step Functions** (the auditable decision path)
- `model.pkl` → **SageMaker** endpoint; velocity/features → **DynamoDB** + Feature Store
- Investigate + SAR LLM → **Bedrock**; RAG → **Bedrock Knowledge Bases / OpenSearch**
- OTP + escalation email → **SES**; tracing → **CloudWatch / X-Ray**
- Data residency (RBI localization) → **PrivateLink** + in-region deploy

The one line that matters: **the decision is still set by Step Functions or a
human — Bedrock only narrates.**

---

### If asked "why not a deep agent / let the LLM decide?"
Because a governed decision path must be auditable and deterministic. An agent
that decides is a black box you can't defend to a regulator. Here the LLM is
sandboxed to advisory work, and the boundary is enforced, not hoped for.
