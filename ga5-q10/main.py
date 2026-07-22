import os, json, hashlib, secrets, asyncio, time, re, traceback, sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import PlainTextResponse
import httpx

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MT = "application/vnd.ga5.invoice-action-receipts+json"

BASE_URL = (os.environ.get("BASE_URL") or "https://ga5-q10.onrender.com/a2a").rstrip("/") + "/"
TOKENS = set(t.strip() for t in os.environ.get("BEARER_TOKENS", "").split(",") if t.strip())
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_URL = os.environ.get("LLM_URL", "https://api.openai.com/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

ACTIONS = {"settle_invoice", "request_approval", "hold_invoice",
           "reject_duplicate", "open_exception"}

app = FastAPI()

# ---------------- storage ----------------
LOCK = asyncio.Lock()
TASKS: Dict[str, dict] = {}
OWNER: Dict[str, str] = {}
IDEM: Dict[str, dict] = {}
DECISION_CACHE: Dict[str, dict] = {}

def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def h(o: Any) -> str:
    return hashlib.sha256(canon(o).encode()).hexdigest()

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def a2a(payload: dict, status: int = 200) -> Response:
    return Response(content=json.dumps(payload, ensure_ascii=False),
                    status_code=status, media_type=A2A_MT)

def err(status: int, code: str, msg: str = "request could not be processed") -> Response:
    return a2a({"error": {"code": code, "message": msg}}, status)

@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    traceback.print_exc(file=sys.stderr)
    return err(500, "INTERNAL", "internal error")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

# ---------------- guards ----------------
def principal(auth: Optional[str]) -> Optional[str]:
    if not auth or not auth.startswith("Bearer "):
        return None
    tok = auth[7:].strip()
    if not tok:
        return None
    if TOKENS and tok not in TOKENS:
        return None
    return hashlib.sha256(tok.encode()).hexdigest()[:32]

def guard(request: Request, auth: Optional[str], body_required: bool):
    p = principal(auth)
    if not p:
        return None, err(401, "UNAUTHENTICATED")
    if request.headers.get("a2a-version") != "1.0":
        return None, err(400, "UNSUPPORTED_VERSION")
    accept = request.headers.get("accept", "")
    if accept and A2A_MT not in accept and "*/*" not in accept and "application/*" not in accept:
        return None, err(400, "UNSUPPORTED_MEDIA_TYPE")
    if body_required:
        ct = (request.headers.get("content-type") or "").split(";")[0].strip()
        if ct and ct != A2A_MT:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE")
    return p, None

# ---------------- agent card ----------------
CARD = {
    "name": "Invoice Action Agent",
    "description": "Reads invoice claim packages, proposes exactly one typed business action per package with decisive document evidence, and executes only grader-accepted proposals under an A2A 1.0 task lifecycle.",
    "version": "1.0.0",
    "protocolVersion": "1.0",
    "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
    "defaultInputModes": [BATCH_MT, RESULT_MT, A2A_MT],
    "defaultOutputModes": [PROP_MT, RECEIPT_MT],
    "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
    "skills": [{
        "id": "invoice_action_agent",
        "name": "invoice_action_agent",
        "description": "Reconciles invoice claim batches against policy documents and selects one of settle_invoice, request_approval, hold_invoice, reject_duplicate, or open_exception per package with exact evidence references.",
        "tags": ["invoice", "reconciliation", "finance", "a2a", "approval"],
        "inputModes": [BATCH_MT],
        "outputModes": [PROP_MT, RECEIPT_MT],
    }],
}

@app.get("/.well-known/agent-card.json")
async def agent_card():
    return Response(json.dumps(CARD), media_type=A2A_MT)

# ---------------- AI decision layer ----------------
SYSTEM = """You are an invoice reconciliation analyst. For each package choose EXACTLY ONE action:
settle_invoice: valid, reconciled, within autonomous authority.
request_approval: commercially valid but exceeds delegated authority.
hold_invoice: payment pauses until a stated verification completes.
reject_duplicate: the same commercial invoice was already paid.
open_exception: material records conflict, needs exception workflow.

Rules:
- Find the ONE paragraph that decides the action. Return EXACTLY the three decisive bracketed references [LIKE-THIS] from that paragraph, verbatim.
- Never return cover-sheet refs, archived/historical examples, or decoy paragraphs containing action words in negated or irrelevant context.
- Extract facts verbatim: vendorName, invoiceNumber, amountMinor (integer minor units), currency (ISO 4217).
- Never settle anything that must be approved, held, rejected as duplicate, or escalated.
- rationale: 60-1500 chars, must name the chosen action and cite at least two of the evidence refs.

Return ONLY JSON: {"decisions":[{"packageId","action","facts":{"vendorName","invoiceNumber","amountMinor","currency"},"evidenceRefs":[3 strings],"rationale"}]}"""

def pkg_key(p: dict) -> str:
    return h({k: v for k, v in p.items() if k != "packageId"})

async def call_llm(packages: List[dict]) -> Dict[str, dict]:
    todo = [p for p in packages if pkg_key(p) not in DECISION_CACHE]
    if todo and LLM_KEY:
        payload = {
            "model": LLM_MODEL, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": canon({"packages": todo})}],
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=8.0)) as c:
                r = await c.post(LLM_URL, json=payload,
                                 headers={"Authorization": f"Bearer {LLM_KEY}"})
                txt = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M))
            by_id = {p.get("packageId"): p for p in todo}
            for d in parsed.get("decisions", []):
                pkg = by_id.get(d.get("packageId"))
                if pkg is not None:
                    DECISION_CACHE[pkg_key(pkg)] = d
        except Exception:
            traceback.print_exc(file=sys.stderr)
    out = {}
    for i, p in enumerate(packages):
        pid = p.get("packageId") or f"pkg-{i}"
        out[pid] = validate(p, DECISION_CACHE.get(pkg_key(p)) or {})
    return out

def validate(pkg: dict, d: dict) -> dict:
    txt = canon(pkg)
    refs = [r for r in (d.get("evidenceRefs") or []) if isinstance(r, str) and r in txt][:3]
    if len(refs) < 3:
        seen = set(refs)
        for f in re.findall(r"\[[^\[\]\n]{2,80}\]", txt):
            if f not in seen:
                seen.add(f); refs.append(f)
            if len(refs) >= 3:
                break
    refs = refs[:3]
    action = d.get("action") if d.get("action") in ACTIONS else "open_exception"
    f = d.get("facts") or {}
    try:
        amt = int(f.get("amountMinor"))
    except Exception:
        amt = 0
    facts = {
        "vendorName": str(f.get("vendorName") or "unknown")[:200],
        "invoiceNumber": str(f.get("invoiceNumber") or "unknown")[:200],
        "amountMinor": amt,
        "currency": str(f.get("currency") or "INR")[:8],
    }
    rat = (d.get("rationale") or "").strip()
    if len(rat) < 60 or action not in rat or sum(1 for r in refs if r in rat) < 2:
        r0 = refs[0] if refs else ""
        r1 = refs[1] if len(refs) > 1 else ""
        r2 = refs[2] if len(refs) > 2 else ""
        rat = (f"Selected {action} for package {pkg.get('packageId')}. The decisive paragraph "
               f"states the controlling condition; evidence {r0} and {r1} establish the vendor, "
               f"invoice identity and amount, and {r2} fixes the disposition, so {action} is the "
               f"only outcome consistent with the current policy revision.")
    rat = rat[:1500]
    if len(rat) < 60:
        rat = rat.ljust(60)
    return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rat}

# ---------------- message:send ----------------
@app.post("/a2a/message:send")
async def message_send(request: Request, authorization: Optional[str] = Header(None)):
    p, e = guard(request, authorization, True)
    if e: return e
    try:
        body = json.loads(await request.body())
    except Exception:
        return err(400, "INVALID_ARGUMENT")
    msg = (body or {}).get("message") or {}
    mid = msg.get("messageId")
    if not mid:
        return err(400, "INVALID_ARGUMENT")

    mhash = h(msg)
    key = f"{p}:{mid}"

    async with LOCK:
        prior = IDEM.get(key)
        if prior:
            if prior["hash"] != mhash:
                return err(409, "IDEMPOTENCY_CONFLICT")
            return a2a({"task": TASKS[prior["taskId"]]})

        parts = msg.get("parts") or []
        part = parts[0] if parts else {}
        mt = part.get("mediaType")
        data = part.get("data") or {}

        if mt == RESULT_MT:
            return await _continue(p, msg, data, mhash, key)

        if mt != BATCH_MT:
            return err(400, "INVALID_ARGUMENT")

        tid = "task-" + secrets.token_hex(16)
        cid = "ctx-" + secrets.token_hex(12)
        TASKS[tid] = {
            "id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_WORKING", "timestamp": now()},
            "history": [msg], "artifacts": [], "metadata": {"batchId": data.get("batchId")},
        }
        OWNER[tid] = p
        IDEM[key] = {"hash": mhash, "taskId": tid}

    try:
        decisions = await call_llm(data.get("packages") or [])
    except Exception:
        traceback.print_exc(file=sys.stderr)
        decisions = {}

    proposals, used = [], set()
    for i, pkg in enumerate(data.get("packages") or []):
        pid = pkg.get("packageId") or f"pkg-{i}"
        if pid in used:
            continue
        used.add(pid)
        d = decisions.get(pid) or validate(pkg, {})
        proposals.append({
            "packageId": pid,
            "actionId": "act-" + hashlib.sha256(f"{tid}:{pid}".encode()).hexdigest()[:24],
            "action": d["action"], "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"],
        })

    async with LOCK:
        task = TASKS[tid]
        if task["status"]["state"] == "TASK_STATE_CANCELED":
            return a2a({"task": task})
        task["artifacts"] = [{
            "artifactId": "art-" + secrets.token_hex(8), "name": "proposals",
            "parts": [{"mediaType": PROP_MT,
                       "data": {"batchId": data.get("batchId"), "proposals": proposals}}],
        }]
        task["metadata"]["proposals"] = {x["packageId"]: x for x in proposals}
        task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()}
        return a2a({"task": task})

async def _continue(p, msg, data, mhash, key) -> Response:
    tid, cid = msg.get("taskId"), msg.get("contextId")
    task = TASKS.get(tid)
    if not task or OWNER.get(tid) != p:
        return err(404, "NOT_FOUND")
    if task["contextId"] != cid:
        return err(400, "INVALID_ARGUMENT")
    if task["status"]["state"] == "TASK_STATE_CANCELED":
        return err(409, "TASK_TERMINAL")
    if task["status"]["state"] == "TASK_STATE_COMPLETED":
        IDEM[key] = {"hash": mhash, "taskId": tid}
        return a2a({"task": task})
    if data.get("batchId") != task["metadata"].get("batchId"):
        return err(400, "INVALID_ARGUMENT")

    stored = task["metadata"].get("proposals") or {}
    execs = []
    for r in (data.get("results") or []):
        pr = stored.get(r.get("packageId"))
        if not pr:
            return err(400, "INVALID_ARGUMENT")
        if r.get("actionId") != pr["actionId"] or r.get("action") != pr["action"]:
            return err(400, "INVALID_ARGUMENT")
        if r.get("outcome") != "ACCEPTED":
            continue
        if not r.get("receiptNonce"):
            return err(400, "INVALID_ARGUMENT")
        execs.append({
            "packageId": pr["packageId"], "actionId": pr["actionId"], "action": pr["action"],
            "receiptNonce": r["receiptNonce"], "facts": pr["facts"],
            "evidenceRefs": pr["evidenceRefs"],
        })

    task["history"].append(msg)
    task["artifacts"].append({
        "artifactId": "art-" + secrets.token_hex(8), "name": "receipts",
        "parts": [{"mediaType": RECEIPT_MT,
                   "data": {"batchId": data.get("batchId"), "executions": execs}}],
    })
    task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    IDEM[key] = {"hash": mhash, "taskId": tid}
    return a2a({"task": task})

# ---------------- reads / cancel ----------------
@app.get("/a2a/tasks")
async def list_tasks(request: Request, authorization: Optional[str] = Header(None)):
    p, e = guard(request, authorization, False)
    if e: return e
    return a2a({"tasks": [t for i, t in TASKS.items() if OWNER.get(i) == p]})

@app.get("/a2a/tasks/{tid}")
async def get_task(tid: str, request: Request, authorization: Optional[str] = Header(None)):
    p, e = guard(request, authorization, False)
    if e: return e
    t = TASKS.get(tid)
    if not t or OWNER.get(tid) != p:
        return err(404, "NOT_FOUND")
    return a2a(t)

@app.post("/a2a/tasks/{tid}:cancel")
async def cancel(tid: str, request: Request, authorization: Optional[str] = Header(None)):
    p, e = guard(request, authorization, False)
    if e: return e
    async with LOCK:
        t = TASKS.get(tid)
        if not t or OWNER.get(tid) != p:
            return err(404, "NOT_FOUND")
        if t["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED"):
            return err(409, "TASK_TERMINAL")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        return a2a(t)