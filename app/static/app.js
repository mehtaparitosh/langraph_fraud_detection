// FraudGraph — minimal, dependency-free frontend.
// Flow: POST /run -> open WS /stream/{tid} -> render node events; on interrupt
// show the pause panel and POST /resume/{tid}; on result show the result panel.

const $ = (sel) => document.querySelector(sel);
const nodeEl = (name) => document.querySelector(`.node[data-node="${name}"]`);

let threadId = null;
let ws = null;
let runStart = 0;
let stepBuffer = []; // fine-grained steps for the current node, flushed on completion
const state = {}; // accumulates state deltas for the result view

// Minimal, safe markdown -> HTML (headings, bold/italic/code, lists, paragraphs).
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function md(src) {
  const inline = (t) =>
    t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
     .replace(/(^|[^*])\*(?!\*)(.+?)\*/g, "$1<em>$2</em>")
     .replace(/`(.+?)`/g, "<code>$1</code>");
  let html = "", inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of escapeHtml(src).split("\n")) {
    const line = raw.trim();
    if (/^#{2,4}\s+/.test(line)) { closeList(); html += `<h4>${inline(line.replace(/^#+\s+/, ""))}</h4>`; }
    else if (/^([-*]|\d+\.)\s+/.test(line)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(line.replace(/^([-*]|\d+\.)\s+/, ""))}</li>`; }
    else if (line === "") { closeList(); }
    else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  closeList();
  return html;
}

function log(line) {
  const el = $("#log");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function resetGraph() {
  document.querySelectorAll(".node").forEach((n) => n.classList.remove("active", "visited"));
  $("#log").textContent = "";
  $("#result").hidden = true;
  $("#pause").hidden = true;
  // Clear result sub-sections so nothing from a previous run lingers.
  ["#r-rec-wrap", "#r-summary-wrap", "#r-sar-wrap"].forEach((s) => { $(s).hidden = true; });
  ["#r-rec", "#r-summary", "#r-sar", "#r-how", "#r-badge"].forEach((s) => { $(s).textContent = ""; });
  $("#r-badge").className = "badge";
  stepBuffer = [];
  for (const k of Object.keys(state)) delete state[k];
}

function highlight(name) {
  const el = nodeEl(name);
  if (!el) return;
  document.querySelectorAll(".node.active").forEach((n) => n.classList.replace("active", "visited"));
  el.classList.remove("visited");
  el.classList.add("active");
}

function fmtDelta(node, delta) {
  const bits = [];
  if (delta.risk_band) bits.push(`band=${delta.risk_band}`);
  if (delta.rule_score !== undefined) bits.push(`score=${delta.rule_score}`);
  if (delta.ml_score !== undefined) bits.push(`ml=${Number(delta.ml_score).toFixed(3)}`);
  if (delta.decision) bits.push(`decision=${delta.decision}`);
  if (delta.reasons) bits.push(`reasons=[${delta.reasons.join(", ")}]`);
  if (delta.evidence) bits.push(`blocklist=${delta.evidence.blocklist.hit} velocity=${delta.evidence.velocity}`);
  if (delta.recommendation) bits.push(`recommend=${delta.recommendation.recommend}`);
  if (delta.analyst_summary) bits.push("analyst_summary set");
  if (delta.sar_draft) bits.push("sar_draft set");
  return `→ ${node}  ${bits.join("  ")}`;
}

function showConsent(payload) {
  $("#pause").hidden = false;
  $("#consent").hidden = false;
  $("#otp").hidden = true;
  const rec = payload.recommendation || {};
  $("#consent-rec").textContent = rec.recommend
    ? `LLM recommends: ${rec.recommend} (confidence ${rec.confidence}) — ${rec.rationale}`
    : "Analyst decision required.";
}

function showOtp(payload) {
  $("#pause").hidden = false;
  $("#consent").hidden = true;
  $("#otp").hidden = false;
  $("#otp-note").textContent = `A one-time code was emailed to ${payload.email}. Enter it below.`;
  $("#otp-code").value = "";
  $("#otp-code").focus();
}

async function resumeWith(value) {
  $("#pause").hidden = true;
  await fetch(`/resume/${threadId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  log(`⏵ resumed with "${value}"`);
}

// Explain HOW the decision was reached: automatic engine vs. a human analyst.
function decisionProvenance(s) {
  const hd = s.human_decision;
  if (!hd) return { text: "Automatic — deterministic engine (no human review)", manual: false };
  if (hd === "step_up") {
    if (s.otp_ok === true) return { text: "Analyst requested step-up → OTP verified → approved", manual: true };
    if (s.otp_ok === false) return { text: "Analyst requested step-up → OTP failed → declined", manual: true };
    return { text: "Analyst requested step-up (OTP pending)", manual: true };
  }
  const verb = { approve: "approved", decline: "declined", escalate: "escalated" }[hd] || hd;
  let text = `Manually ${verb} by analyst`;
  if (hd === "escalate") text += " — escalation email sent to the team";
  const rec = s.recommendation && s.recommendation.recommend;
  if (rec && rec !== hd) text += ` (LLM had recommended “${rec}”)`;
  return { text, manual: true };
}

function showResult(s) {
  $("#result").hidden = false;
  $("#r-decision").textContent = s.decision || "—";
  const prov = decisionProvenance(s);
  $("#r-how").textContent = prov.text;
  $("#r-badge").textContent = prov.manual ? "manual" : "automatic";
  $("#r-badge").className = "badge " + (prov.manual ? "badge-manual" : "badge-auto");
  $("#r-band").textContent = s.risk_band || "—";
  $("#r-score").textContent = s.rule_score ?? "—";
  $("#r-ml").textContent = s.ml_score !== undefined ? Number(s.ml_score).toFixed(3) : "—";
  $("#r-reasons").textContent = (s.reasons || []).join(", ") || "none";
  if (s.recommendation) {
    $("#r-rec-wrap").hidden = false;
    const r = s.recommendation;
    $("#r-rec").textContent = `${r.recommend} (confidence ${r.confidence}) — ${r.rationale}`;
  }
  if (s.analyst_summary) { $("#r-summary-wrap").hidden = false; $("#r-summary").innerHTML = md(s.analyst_summary); }
  if (s.sar_draft) { $("#r-sar-wrap").hidden = false; $("#r-sar").innerHTML = md(s.sar_draft); }
}

function openStream() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/stream/${threadId}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "ping") {
      // Show elapsed time + which node is currently running (LLM nodes emit
      // nothing until they finish, so this is how you see they're working).
      const secs = Math.round((Date.now() - runStart) / 1000);
      const active = document.querySelector(".node.active");
      const where = active ? `${active.dataset.node} running` : "working";
      $("#status").textContent = `${where}… (${secs}s)`;
    } else if (msg.type === "node_start") {
      // Highlight the node WHILE it runs, and start a fresh step buffer for it.
      highlight(msg.node);
      stepBuffer = [];
    } else if (msg.type === "step") {
      // A fine-grained action taken inside the current node.
      stepBuffer.push(msg.message);
    } else if (msg.type === "node") {
      highlight(msg.node);
      Object.assign(state, msg.delta);
      log(fmtDelta(msg.node, msg.delta));
      stepBuffer.forEach((s) => log(`      ↳ ${s}`)); // indented sub-steps
      stepBuffer = [];
    } else if (msg.type === "interrupt") {
      log(`⏸ interrupt: ${msg.kind}`);
      if (msg.kind === "consent") showConsent(msg.payload);
      else if (msg.kind === "otp") showOtp(msg.payload);
    } else if (msg.type === "result") {
      showResult(msg.state);
      log("✔ done");
      $("#status").textContent = "Done.";
    } else if (msg.type === "error") {
      $("#status").textContent = "Error: " + msg.message;
    }
  };
  ws.onclose = () => log("(stream closed)");
}

async function readFileTxn() {
  const f = $("#file").files[0];
  if (!f) return null;
  return JSON.parse(await f.text());
}

let defaultsData = {};
async function loadDefaults() {
  try {
    defaultsData = await (await fetch("/defaults")).json();
  } catch (e) { /* leave empty */ }
  updateDescription();
}
async function updateDescription() {
  const uploaded = await readFileTxn();
  if (uploaded) {
    $("#txn-desc").textContent = "Uploaded transaction.";
    $("#payload-json").textContent = JSON.stringify(uploaded, null, 2);
    return;
  }
  const d = defaultsData[$("#default").value];
  if (d) {
    $("#txn-desc").textContent = d.description;
    $("#payload-json").textContent = JSON.stringify(d.transaction, null, 2);
  }
}

async function run() {
  resetGraph();
  runStart = Date.now();
  $("#status").textContent = "Running…";
  const uploaded = await readFileTxn();
  const body = uploaded ? { transaction: uploaded } : { default: $("#default").value };
  const res = await fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (data.error) { $("#status").textContent = "Error: " + data.error; return; }
  threadId = data.thread_id;
  log(`thread ${threadId}`);
  openStream();
}

$("#run").addEventListener("click", run);
$("#default").addEventListener("change", updateDescription);
$("#file").addEventListener("change", updateDescription);
loadDefaults();
document.querySelectorAll("#consent .buttons button").forEach((b) =>
  b.addEventListener("click", () => resumeWith(b.dataset.choice))
);
$("#otp-submit").addEventListener("click", () => resumeWith($("#otp-code").value.trim()));
