"""
Enterprise AI Support Platform — FastAPI Backend
Session 6 of 12 — Guardrails & Execution Bounding
"""

import json
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from support_agent import (
    run_ticket, stream_ticket,
    run_session_verification,
    get_conversation_history,
    get_active_threads,
    TOOLS, MAX_ITERATIONS,
    SUMMARY_THRESHOLD,
    INJECTION_PATTERNS,
)

app = FastAPI(title="Enterprise AI Support Platform", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    ticket:          str
    thread_id:       Optional[str] = None
    return_existing: bool = False   # True when stream already ran this thread


class StreamRequest(BaseModel):
    ticket:    str
    thread_id: Optional[str] = None


@app.post("/api/run")
def run(req: RunRequest):
    result = run_ticket(req.ticket, thread_id=req.thread_id,
                        return_existing=req.return_existing)

    # Extract tool call log
    tool_calls_log = []
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    'tool_name': tc['name'],
                    'args':      tc['args'],
                    'call_id':   tc['id'],
                })

    # Match each call_id to a ToolMessage result
    tool_results_map = {}
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_call_id'):
            try:
                tool_results_map[msg.tool_call_id] = json.loads(msg.content)
            except Exception:
                tool_results_map[msg.tool_call_id] = msg.content

    for entry in tool_calls_log:
        entry['result'] = tool_results_map.get(entry['call_id'], {})

    iterations_used = result.get('iteration_count', 0)

    return {
        "category":              result.get("category", ""),
        "final_response":        result.get("final_response", ""),
        "is_safe":               result.get("is_safe", True),
        "pii_detected":          result.get("pii_detected", False),
        "injection_detected":    result.get("injection_detected", False),
        "sanitized_input":       result.get("sanitized_input", "")[:100],
        "blocked":               not result.get("is_safe", True),
        "iteration_count":       iterations_used,
        "raw_input":             result.get("raw_input", ""),
        "tool_calls_log":        tool_calls_log,
        "circuit_breaker_fired": iterations_used > MAX_ITERATIONS - 1,
        "iterations_used":       iterations_used,
        "max_iterations":        MAX_ITERATIONS,
        "thread_id":             result.get("thread_id", ""),
        "message_count":         len(result.get("messages", [])),
        "system_summary":        result.get("system_summary", ""),
        "summary_active":        bool(result.get("system_summary", "").strip()),
        "summary_threshold":     SUMMARY_THRESHOLD,
    }


@app.post("/api/stream")
async def stream(req: StreamRequest):
    # Generate the UUID here so the start event exposes it and /api/run can reuse it
    thread_id_used = req.thread_id if req.thread_id else str(uuid.uuid4())

    def generate():
        start_payload = {"type": "start", "thread_id": thread_id_used}
        yield f"data: {json.dumps(start_payload)}\n\n"

        for node_name, snapshot in stream_ticket(req.ticket, thread_id=thread_id_used):
            payload = {
                "node":     node_name,
                "category": snapshot.get("category", ""),
                "response": snapshot.get("final_response", ""),
            }

            # Enrich ingress_node events with security data
            if node_name == 'ingress_node':
                payload['pii_detected']       = snapshot.get('pii_detected', False)
                payload['injection_detected']  = snapshot.get('injection_detected', False)
                payload['is_safe']             = snapshot.get('is_safe', True)
                payload['sanitized_input']     = snapshot.get('sanitized_input', '')[:80]

            # Enrich blocked_response_node events
            if node_name == 'blocked_response_node':
                payload['blocked'] = True
                payload['reason']  = (
                    "injection_detected"
                    if snapshot.get('injection_detected')
                    else "unsafe_content"
                )

            # Enrich summarization_node events
            if node_name == 'summarization_node':
                payload['summary_fired'] = True
                payload['summary']       = snapshot.get('system_summary', '')[:120]
                payload['msgs_before']   = len(snapshot.get('messages', []))

            # Enrich agent_node events with tool call info and iteration data
            if node_name == 'agent_node':
                msgs = snapshot.get('messages', [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        payload['tool_calls'] = [
                            {'name': tc['name'], 'args': tc['args']}
                            for tc in last_msg.tool_calls
                        ]
                payload['iteration']      = snapshot.get('iteration_count', 0)
                payload['max_iterations'] = MAX_ITERATIONS
                payload['summary_active'] = bool(
                    snapshot.get('system_summary', '').strip()
                )

            # Enrich tool_node events with tool results
            if node_name == 'tool_node':
                msgs = snapshot.get('messages', [])
                tool_results = []
                for msg in msgs:
                    if hasattr(msg, 'tool_call_id'):
                        tool_results.append({
                            'tool_name': getattr(msg, 'name', ''),
                            'content':   msg.content,
                        })
                if tool_results:
                    payload['tool_results'] = tool_results

            yield f"data: {json.dumps(payload)}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/verify")
def verify():
    result = run_session_verification()
    return result


@app.get("/api/threads")
def list_threads():
    """
    Returns all active thread_ids from the SQLite checkpointer.
    Used by the UI thread selector dropdown.
    """
    threads = get_active_threads()
    return {"threads": threads, "count": len(threads)}


@app.get("/api/history/{thread_id}")
def get_history(thread_id: str):
    """
    Returns the full checkpoint history for a thread_id.
    Each entry: step, node, category, iteration,
    message_count, final_response, is_end, checkpoint_id.
    """
    history = get_conversation_history(thread_id)
    return {
        "thread_id": thread_id,
        "count":     len(history),
        "history":   history,
    }


@app.get("/health")
def health():
    return {
        "status":             "ok",
        "session":            6,
        "tools":              len(TOOLS),
        "max_iterations":     MAX_ITERATIONS,
        "summary_threshold":  SUMMARY_THRESHOLD,
        "injection_patterns": len(INJECTION_PATTERNS),
        "persistence":        "sqlite",
        "security":           "active",
    }


# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
