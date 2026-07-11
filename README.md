# Enterprise AI Support Platform

A production-grade AI support agent built session-by-session across 12 sessions using **LangGraph**, **Google Gemini 2.5 Flash**, and **FastAPI**. Each session adds a self-contained capability layer — nothing is ever removed, only extended.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd phase3-session6

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy NLP model (required for PII detection)
python -m spacy download en_core_web_lg

# 5. Set your Google Gemini API key
echo "GOOGLE_API_KEY=your-key-here" > .env
# Get a key at: https://aistudio.google.com/app/apikey

# 6. Run the CLI test harness
python support_agent.py

# 7. Start the web server
python api.py
# Open: http://localhost:8000
```

---

## Architecture Overview

```
User Input
    │
    ▼
┌─────────────┐
│ ingress_node │  ← PII detection + masking (Presidio)
│  (Session 6) │    Injection pattern detection (regex)
└──────┬───────┘
       │ is_safe?
   ┌───┴───┐
   │       │
   ▼       ▼
blocked  classify_node  ← LLM classifies: technical | billing | fraud | general
_node    (Session 1)
(S6)         │
         ┌───┼───────────┐
         ▼   ▼           ▼
      fraud  general   agent_node  ← ReAct loop (Session 3)
      handler handler  (Session 2)   reads sanitized_input (Session 6)
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
                tool_node    respond_node
                (Session 2)  (Session 2)
                    │             │
                    └──────┬──────┘
                           ▼
                       egress_node  ← Output PII scan + uncertainty check
                       (Session 6)
                           │
                          END
```

**Summarization branch** (Session 5): when message count > 8, `classify_node` routes to `summarization_node` → `agent_node` instead of going directly.

---

## Session-by-Session Breakdown

### Session 1 — The Blueprint
**Goal:** Establish the graph skeleton, state schema, and routing logic.

**What was built:**
- `SupportState` TypedDict with all fields pre-declared for all 12 sessions
- `classify_node` — LLM-based classifier (technical / billing / fraud / general)
- `route_by_category` — pure Python router, zero LLM calls
- Handler stubs: `technical_handler`, `billing_handler`, `fraud_handler`, `general_handler`
- `build_graph()` — LangGraph `StateGraph` compiled with `SqliteSaver` checkpointer
- `build_initial_state()` — safe defaults for all 17 fields
- `run_ticket()` — entry point called by CLI and server
- FastAPI `/api/run` endpoint + `index.html` skeleton

**Key design decisions:**
- All state fields declared upfront so later sessions can populate them without schema changes
- Entry point is `classify_node` (changes to `ingress_node` in Session 6)
- Handler stubs return static strings — replaced by the ReAct loop in Session 2

---

### Session 2 — Tool Binding & Execution
**Goal:** Give the agent real tools and a proper response loop.

**What was built:**
- `get_customer_details` — CRM lookup (mock), returns billing/subscription data
- `search_knowledge_base` — KB search (mock), returns troubleshooting articles
- `check_fraud_signals` — Fraud DB lookup (mock), returns risk score + patterns
- `agent_node` — single-pass LLM node with `llm.bind_tools()`
- `tool_node` — LangGraph `ToolNode` wrapping all three tools
- `respond_node` — extracts final `AIMessage` content → `final_response`
- `route_after_agent` — reads `tool_calls` on last message to decide next node
- Tool call panel in `index.html` showing args and results

**Key design decisions:**
- Tools use layered validation: argument format → database lookup → data filtering
- `route_after_agent` is pure Python — no LLM involved in routing
- `billing` and `technical` categories skip handler stubs and go directly to `agent_node`

---

### Session 3 — The ReAct Architecture
**Goal:** Multi-turn tool use with safety bounds.

**What was built:**
- Circuit breaker — `MAX_ITERATIONS = 5` hard stop in `agent_node`
- Duplicate tool call detection — fingerprint each call, escalate if same call repeats
- `build_escalation_response()` — graceful user-facing message when circuit fires
- `fraud_handler` upgraded from stub — uses `check_fraud_signals` tool directly
- `AGENT_SYSTEM_PROMPT` — explicit tool usage rules for the LLM
- Iteration tracker panel in `index.html` with progress bar and FIRED badge

**Key design decisions:**
- Circuit breaker counts iterations, not tool calls — simpler and more predictable
- Duplicate detection uses `tool_name::sorted_args` fingerprint — catches infinite loops
- `fraud_handler` uses a single-pass tool call, not the full ReAct loop

---

### Session 4 — Persistence & Threading
**Goal:** Conversations survive process restarts; multiple users are isolated.

**What was built:**
- `SqliteSaver` checkpointer — SQLite-backed state persistence
- `thread_id` parameter on `run_ticket()` and `stream_ticket()`
- `get_conversation_history()` — reads checkpoint history for a thread
- `get_active_threads()` — lists all threads from SQLite
- `/api/history/{thread_id}` and `/api/threads` endpoints
- Thread selector dropdown + conversation history panel in `index.html`
- `return_existing` flag on `/api/run` — prevents double-execution when streaming

**Key design decisions:**
- Each conversation has a unique `thread_id` (UUID); state is checkpointed at every node
- `return_existing=True` is used by the UI to read back what the stream already executed
- SQLite chosen over in-memory for durability without infrastructure overhead

---

### Session 5 — Context Management & Summarization
**Goal:** Prevent context window explosion in long conversations.

**What was built:**
- `SUMMARY_THRESHOLD = 8` — message count trigger
- `SUMMARIZATION_PROMPT` — dense, fact-preserving compression instructions
- `summarization_node` — compresses old messages into `system_summary`, trims to last 4
- `deduplicate_messages` — custom reducer replacing `add_messages`, handles `RemoveMessage` deletions
- `route_after_classify` — replaces `route_by_category`; checks threshold before routing to `agent_node`
- `agent_node` updated — reads `system_summary` and prepends to prompt instead of trimming
- Context Summary Panel in `index.html` with progress bar and compressed text

**Key design decisions:**
- `RemoveMessage` + custom reducer chosen over list truncation — cleaner state management
- `system_summary` is prepended to the system prompt, not injected as a message — avoids confusing the LLM
- `trim_context()` retired — summarization replaces it entirely
- Summarization only fires for `billing` and `technical` categories, not `fraud`/`general`

---

### Session 6 — Guardrails & Execution Bounding ← **Current**
**Goal:** Security sandwich — validate all inputs before the LLM sees them, scan all outputs before delivery.

**What was built:**
- `AnalyzerEngine` + `AnonymizerEngine` — Presidio PII detection/masking at module level
- `PII_ENTITIES` — 7 entity types: CREDIT_CARD, EMAIL_ADDRESS, PHONE_NUMBER, PERSON, US_SSN, IBAN_CODE, IP_ADDRESS
- `INJECTION_PATTERNS` — 14 regex patterns covering jailbreak, role override, prompt extraction attempts
- `UNCERTAINTY_MARKERS` — 9 patterns for output quality flagging
- `BLOCKED_RESPONSE_TEMPLATE` — pre-written refusal with audit reference number
- `ingress_node` — PII scan → mask → injection check → set `is_safe`. Pure CPU, zero LLM tokens
- `route_after_ingress` — `is_safe=False` → `blocked_response_node`, `True` → `classify_node`
- `blocked_response_node` — returns template refusal. Zero LLM tokens consumed
- `egress_node` — Presidio scan on `final_response` + uncertainty marker check. Logs only (no blocking)
- Graph entry point changed: `classify_node` → `ingress_node`
- `classify_node` and `agent_node` both read `sanitized_input` (PII-masked)
- Security Panel in `index.html` with PII, Injection, Safety Gate status rows
- `/health` exposes `injection_patterns` count and `security: active`

**Key design decisions:**
- PII alone does NOT block — it is masked and the ticket is processed normally
- Injection blocks regardless of PII — `is_safe` is driven only by `injection_detected`
- `egress_node` logs but does not block in this session — active remediation added in Session 9
- Blocked responses are template-only: zero LLM API calls, zero token cost
- `sanitized_input` replaces `raw_input` in the LLM call chain but `raw_input` is never modified

**Verification (5/5):**
```
✅ PII detected and masked before LLM
✅ PII ticket processed normally (is_safe=True)
✅ Injection detected and blocked
✅ Blocked response is template not LLM-generated
✅ Red team payload — both PII and injection caught
```

---

## Upcoming Sessions

### Session 7 — Multi-Agent Topologies
**Goal:** Decompose the monolithic graph into communicating subgraphs.

**What gets added:**
- `triage_subgraph` — compiled subgraph: `ingress_node` → `classify_node` → handoff
- `tech_subgraph` — compiled subgraph: `agent_node` → tool loop → `respond_node`
- `SharedState` TypedDict — shared across both subgraphs
- Master graph wires subgraphs together: `triage_subgraph` → `is_safe?` → `tech_subgraph` → END
- `append_tool_results` — new custom reducer preventing silent overwrite when two subgraphs write the same field
- Live demonstration of the silent overwrite bug and its fix

**Why:** Large graphs become unmanageable. Subgraphs allow independent testing, versioning, and parallel execution of specialist agents.

---

### Session 8 — Supervisor & Delegation
**Goal:** A supervisor agent that delegates to specialist workers.

**What gets added:**
- `supervisor_node` — LLM-based orchestrator that decides which worker to invoke
- `delegation_count` — already in state, now enforced as a limit
- `next_worker` — supervisor writes this; master router reads it
- Specialist workers promoted from stubs: billing specialist, technical specialist
- Inter-agent message passing via `internal_notes`

**Why:** Complex tickets that span multiple domains (billing + technical) need a coordinator, not a single generalist agent.

---

### Session 9 — Parallel Agent Swarm
**Goal:** Multiple agents run simultaneously on the same ticket.

**What gets added:**
- `fraud_swarm` — 3 parallel fraud analysis agents: transaction analyst, pattern detector, risk scorer
- LangGraph `Send` API for fan-out to parallel branches
- `operator.add` reducer on `internal_notes` — safe parallel writes
- `egress_node` upgraded — now actively redacts PII in output instead of logging only
- Swarm result aggregation node — merges parallel findings into a single response

**Why:** Fraud detection benefits from running multiple independent analyses simultaneously and synthesizing the results.

---

### Session 10 — Human-in-the-Loop
**Goal:** High-stakes actions require human approval before execution.

**What gets added:**
- `github_draft` — proposed GitHub issue before approval
- `github_issue_url` — populated after approval
- `interrupt()` — LangGraph interrupt primitive, pauses graph at approval gate
- Approval UI in `index.html` — approve/reject with reason
- `time_travel` — UI lets agents resume from any prior checkpoint

**Why:** Automated agents should not take irreversible external actions (creating issues, sending emails, processing refunds) without a human sign-off.

---

### Session 11 — External Write Actions
**Goal:** Agent executes approved actions against real external systems.

**What gets added:**
- GitHub Issues integration — creates real issues via GitHub API
- Slack notification node — posts to a channel after issue creation
- Action audit log — every external write is recorded with actor, timestamp, payload
- Retry logic with exponential backoff on external API failures
- Rollback node — reverses actions if downstream steps fail

**Why:** After human approval, the agent needs to actually do the work. External integrations require careful error handling and audit trails.

---

### Session 12 — The Auditor & Time Travel
**Goal:** Full observability, audit timeline, and state replay.

**What gets added:**
- Complete audit timeline UI — every checkpoint visualized as a scrubable timeline
- Time travel — load any prior checkpoint and re-run from that point
- Diff viewer — compare state between any two checkpoints
- Audit export — download full conversation + tool calls + state snapshots as JSON
- `/api/replay/{thread_id}/{checkpoint_id}` endpoint
- Performance dashboard — token usage, iteration counts, latency per node

**Why:** Production systems need full observability. Time travel enables debugging, compliance audits, and customer dispute resolution by replaying exactly what the agent did.

---

## File Structure

```
phase3-session6/
├── support_agent.py   # Core agent — LangGraph graph, nodes, tools, state
├── api.py             # FastAPI server — REST + SSE streaming endpoints
├── index.html         # Single-file frontend — all CSS, HTML, JavaScript
├── requirements.txt   # Python dependencies
├── .env               # GOOGLE_API_KEY (not committed)
├── .gitignore
└── README.md
```

## State Schema

All 17 fields declared in Session 1. Each session activates the fields it owns.

| Field | Session | Description |
|---|---|---|
| `raw_input` | S1 | Original user message, never modified |
| `sanitized_input` | S6 | PII-masked version for LLM consumption |
| `category` | S1 | technical / billing / fraud / general |
| `messages` | S2 | Conversation history with dedup reducer |
| `customer_data` | S2 | CRM tool results |
| `tool_results` | S2 | Fingerprints + tool outputs (append-safe) |
| `pii_detected` | S6 | True if Presidio found PII in input |
| `injection_detected` | S6 | True if injection pattern matched |
| `is_safe` | S6 | False blocks the request before LLM |
| `system_summary` | S5 | Compressed conversation history |
| `iteration_count` | S3 | ReAct loop counter for circuit breaker |
| `internal_notes` | S8 | Inter-agent scratchpad (append-safe) |
| `delegation_count` | S8 | Supervisor delegation limit counter |
| `next_worker` | S8 | Supervisor's routing decision |
| `github_draft` | S10 | Proposed issue pending human approval |
| `github_issue_url` | S11 | URL of created GitHub issue |
| `final_response` | S1 | Delivered to the user |

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent framework | LangGraph 1.x |
| LLM client | LangChain Google GenAI |
| PII detection | Microsoft Presidio + spaCy en_core_web_lg |
| API server | FastAPI + Uvicorn |
| Persistence | SQLite via LangGraph SqliteSaver |
| Frontend | Vanilla JS + CSS (single file, no build step) |
