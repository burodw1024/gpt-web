#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# SINGLE FULL SETUP SCRIPT: DJANGO UI + RAG + JSONL UPLOADER
# ==========================================================
# What you get (end-to-end, from scratch):
#   - Python venv
#   - Django project + app
#   - Web UI:
#       /         => Chat UI (LLM-only + RAG toggle)
#       /upload   => Upload JSONL and Upsert to Qdrant
#   - API endpoints:
#       POST /api/chat          => LLM-only (Ollama /api/generate)
#       POST /api/ask           => RAG (embed -> Qdrant search -> generate)
#       POST /api/upload-jsonl  => Upload JSONL -> embed each line -> upsert Qdrant
#
# Defaults for your cluster:
#   - Ollama URL:  http://10.109.199.247:11434
#   - Qdrant URL:  http://10.105.26.13:6333
#   - Collection:  dw_text
#   - Embed model: nomic-embed-text:latest
#   - Gen model:   qwen2.5-1.5b:latest
#
# Usage:
#   mkdir -p /home/mamun/app/buro_llm_chat_new
#   cd /home/mamun/app/buro_llm_chat_new
#   nano setup_all.sh  # paste this file
#   chmod +x setup_all.sh
#   ./setup_all.sh
#
# Override examples:
#   PORT=8002 ./setup_all.sh
#   OLLAMA_URL_DEFAULT=http://<NODE_IP>:31134 QDRANT_URL_DEFAULT=http://<NODE_IP>:31633 ./setup_all.sh

# -------------------------
# CONFIG (override via env)
# -------------------------
PROJECT_DIR="${PROJECT_DIR:-.}"            # current directory by default
DJANGO_PROJECT="${DJANGO_PROJECT:-buro_first}"
APP_NAME="${APP_NAME:-chat}"

BIND_ADDR="${BIND_ADDR:-0.0.0.0}"
PORT="${PORT:-8001}"

OLLAMA_URL_DEFAULT="${OLLAMA_URL_DEFAULT:-http://10.109.199.247:11434}"
EMBED_MODEL_DEFAULT="${EMBED_MODEL_DEFAULT:-nomic-embed-text:latest}"
GEN_MODEL_DEFAULT="${GEN_MODEL_DEFAULT:-qwen2.5-1.5b:latest}"

QDRANT_URL_DEFAULT="${QDRANT_URL_DEFAULT:-http://10.105.26.13:6333}"
QDRANT_COLLECTION_DEFAULT="${QDRANT_COLLECTION_DEFAULT:-dw_text}"

# -------------------------
# PRECHECKS
# -------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3."
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ -f "manage.py" ]]; then
  echo "ERROR: manage.py already exists in $(pwd)."
  echo "Run this script in an EMPTY directory (new project folder)."
  exit 1
fi

echo "==> Creating project in: $(pwd)"
echo "    DJANGO_PROJECT=${DJANGO_PROJECT}"
echo "    APP_NAME=${APP_NAME}"
echo "    BIND_ADDR=${BIND_ADDR} PORT=${PORT}"
echo

# -------------------------
# Create venv + install deps
# -------------------------
echo "==> Creating venv"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip >/dev/null
pip install "django==4.2.*" requests >/dev/null

# -------------------------
# Create Django project + app
# -------------------------
echo "==> Creating Django project"
django-admin startproject "${DJANGO_PROJECT}" .

echo "==> Creating app: ${APP_NAME}"
python manage.py startapp "${APP_NAME}"

# -------------------------
# settings.py (correct INSTALLED_APPS, ALLOWED_HOSTS)
# -------------------------
echo "==> Updating settings.py"
SETTINGS_FILE="${DJANGO_PROJECT}/settings.py"

python3 - <<PY
import pathlib, re

app = "${APP_NAME}"
p = pathlib.Path("${SETTINGS_FILE}")
s = p.read_text(encoding="utf-8")

app_config = f"{app}.apps.{app.capitalize()}Config"

# Insert app_config as a literal string into INSTALLED_APPS
if app_config not in s:
    s = re.sub(
        r"INSTALLED_APPS\\s*=\\s*\\[\\n",
        "INSTALLED_APPS = [\\n    " + repr(app_config) + ",\\n",
        s,
        count=1
    )

# Allow all hosts for LAN testing
if "ALLOWED_HOSTS" in s:
    s = re.sub(r"ALLOWED_HOSTS\\s*=\\s*\\[[^\\]]*\\]", 'ALLOWED_HOSTS = ["*"]', s, count=1)
else:
    s += '\\nALLOWED_HOSTS = ["*"]\\n'

p.write_text(s, encoding="utf-8")
PY

# -------------------------
# apps.py
# -------------------------
echo "==> Writing apps.py"
cat > "${APP_NAME}/apps.py" <<PY
from django.apps import AppConfig

class ${APP_NAME^}Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "${APP_NAME}"
PY

# -------------------------
# Project urls.py
# -------------------------
echo "==> Writing project urls.py"
cat > "${DJANGO_PROJECT}/urls.py" <<PY
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("${APP_NAME}.urls")),
]
PY

# -------------------------
# App urls.py
# -------------------------
echo "==> Writing app urls.py"
cat > "${APP_NAME}/urls.py" <<'PY'
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload", views.upload_page, name="upload_page"),
    path("api/chat", views.api_chat, name="api_chat"),
    path("api/ask", views.api_ask, name="api_ask"),
    path("api/upload-jsonl", views.api_upload_jsonl, name="api_upload_jsonl"),
]
PY

# -------------------------
# Views (LLM + RAG + Upload JSONL)
# -------------------------
echo "==> Writing views.py"
cat > "${APP_NAME}/views.py" <<PY
import json
import os
import uuid
import requests

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Defaults (override via env vars)
OLLAMA_URL = os.getenv("OLLAMA_URL", "${OLLAMA_URL_DEFAULT}").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "${EMBED_MODEL_DEFAULT}")
GEN_MODEL = os.getenv("GEN_MODEL", "${GEN_MODEL_DEFAULT}")

QDRANT_URL = os.getenv("QDRANT_URL", "${QDRANT_URL_DEFAULT}").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "${QDRANT_COLLECTION_DEFAULT}")

# -------------------------
# Pages
# -------------------------
def index(request):
    return render(request, "${APP_NAME}/index.html")

def upload_page(request):
    return render(request, "${APP_NAME}/upload.html")

# -------------------------
# Ollama + Qdrant helpers
# -------------------------
def _ollama_embeddings_with_model(model: str, text: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]

def _ollama_embeddings(text: str):
    return _ollama_embeddings_with_model(EMBED_MODEL, text)

def _qdrant_search(vector, limit=6):
    r = requests.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True, "with_vector": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("result", []) or []

def _qdrant_upsert(collection: str, pid: str, vector, payload: dict):
    body = {"points": [{"id": pid, "vector": vector, "payload": payload}]}
    r = requests.put(
        f"{QDRANT_URL}/collections/{collection}/points?wait=true",
        json=body,
        timeout=120,
    )
    r.raise_for_status()

def _dedupe_hits(hits):
    # Simple dedupe by payload prompt prefix (helps when you re-upsert same rows)
    seen = set()
    out = []
    for h in hits:
        p = h.get("payload") or {}
        txt = (p.get("prompt") or "").strip()
        key = txt[:220]
        if key and key not in seen:
            seen.add(key)
            out.append(h)
    return out

def _ollama_generate(prompt: str):
    # Ollama 0.3.14 supports /api/generate for text generation
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 280},
        },
        timeout=300,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

# -------------------------
# API: LLM-only
# -------------------------
@csrf_exempt
def api_chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        user_message = (payload.get("message") or "").strip()
        history = payload.get("history") or []

        if not user_message:
            return JsonResponse({"error": "message is required"}, status=400)

        parts = [
            "You are a helpful assistant running privately on-prem.",
            "Answer clearly and concisely.",
            "",
            "Conversation:",
        ]

        for m in history[-6:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            parts.append(("User: " if role == "user" else "Assistant: ") + content)

        parts.append("User: " + user_message)
        parts.append("Assistant:")

        answer = _ollama_generate("\\n".join(parts))
        return JsonResponse({"answer": answer})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Upstream request failed: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# -------------------------
# API: RAG (embed -> qdrant -> generate)
# -------------------------
@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        question = (payload.get("question") or payload.get("message") or "").strip()
        top_k = int(payload.get("top_k", 6))

        if not question:
            return JsonResponse({"error": "question is required"}, status=400)

        vec = _ollama_embeddings(question)
        hits = _qdrant_search(vec, limit=top_k)
        hits = _dedupe_hits(hits)

        ctx_blocks = []
        sources = []
        for i, h in enumerate(hits, 1):
            p = h.get("payload") or {}
            txt = (p.get("prompt") or "").strip()
            if not txt:
                continue
            ctx_blocks.append(f"[{i}] {txt}")
            sources.append({"id": h.get("id"), "score": h.get("score")})

        context = "\\n\\n".join(ctx_blocks)

        rag_prompt = f"""
You answer ONLY from CONTEXT.
Do NOT mention real-time or external data.
If answer not in CONTEXT, say: Not found in provided data.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
""".strip()

        answer = _ollama_generate(rag_prompt)
        return JsonResponse({"answer": answer, "sources": sources})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Upstream request failed: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# -------------------------
# API: Upload JSONL -> embed -> upsert
# -------------------------
@csrf_exempt
@require_POST
def api_upload_jsonl(request):
    """
    Upload a JSONL file and upsert into Qdrant using Ollama embeddings.
    Each line: JSON with at least {"prompt":"..."}.
    Optional: model, id, dw_id. If no id provided => UUID.
    """
    try:
        if "file" not in request.FILES:
            return JsonResponse({"error": "file is required"}, status=400)

        f = request.FILES["file"]
        if not f.name.lower().endswith(".jsonl"):
            return JsonResponse({"error": "Only .jsonl is supported"}, status=400)

        collection = (request.POST.get("collection") or QDRANT_COLLECTION).strip()
        embed_model = (request.POST.get("embed_model") or EMBED_MODEL).strip()

        ok = 0
        failed = 0
        errors = []

        for ln, raw in enumerate(f, start=1):
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                doc = json.loads(line)

                text = (doc.get("prompt") or "").strip()
                if not text:
                    raise ValueError("missing 'prompt'")

                model = (doc.get("model") or embed_model).strip()
                pid = str(doc.get("id") or doc.get("dw_id") or uuid.uuid4())

                vec = _ollama_embeddings_with_model(model, text)
                _qdrant_upsert(collection, pid, vec, doc)

                ok += 1

            except Exception as e:
                failed += 1
                errors.append({"line": ln, "error": str(e)})
                if len(errors) >= 30:
                    errors.append({"line": ln, "error": "Too many errors; truncated."})
                    break

        return JsonResponse({
            "success": ok,
            "failed": failed,
            "errors": errors,
            "collection": collection,
            "embed_model": embed_model,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
PY

# -------------------------
# Templates
# -------------------------
echo "==> Writing templates"
mkdir -p "${APP_NAME}/templates/${APP_NAME}"

# index.html (chat + menu)
cat > "${APP_NAME}/templates/${APP_NAME}/index.html" <<'HTML'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Buro On-Prem Private LLM + RAG</title>
  <style>
    body { font-family: Arial, sans-serif; margin:0; background:#0b0f14; color:#e6edf3; }
    header { padding:16px 18px; background:#111823; border-bottom:1px solid #1e2a3a; display:flex; gap:12px; align-items:center; }
    header h1 { margin:0; font-size:16px; font-weight:600; flex:1; }
    header a { color:#9fb0c3; text-decoration:none; border:1px solid #1e2a3a; padding:6px 10px; border-radius:10px; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 16px; }
    .bar { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
    .toggle { display:flex; gap:8px; align-items:center; font-size:13px; color:#9fb0c3; }
    .pill { padding:4px 8px; border:1px solid #1e2a3a; border-radius:999px; font-size:12px; color:#9fb0c3; }
    .chatbox { background:#0f1621; border:1px solid #1e2a3a; border-radius:12px; height: 70vh; overflow:auto; padding: 14px; }
    .msg { margin: 10px 0; display:flex; }
    .msg.user { justify-content:flex-end; }
    .bubble { max-width: 78%; padding:10px 12px; border-radius:12px; white-space:pre-wrap; line-height:1.35; }
    .user .bubble { background:#1d4ed8; }
    .assistant .bubble { background:#152235; border:1px solid #22324a; }
    .composer { display:flex; gap:10px; margin-top:12px; }
    textarea { flex:1; resize:none; height:54px; padding:10px 12px; border-radius:12px;
               border:1px solid #1e2a3a; background:#0f1621; color:#e6edf3; }
    button { width:140px; border-radius:12px; border:1px solid #1e2a3a; background:#0b5cff; color:white;
             font-weight:600; cursor:pointer; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    .hint { color:#9fb0c3; font-size:12px; margin-top:10px; }
  </style>
</head>
<body>
<header>
  <a href="/upload">Upload JSONL</a>
  <h1>Buro On-Prem Private LLM + RAG</h1>
</header>

<div class="wrap">
  <div class="bar">
    <div class="toggle">
      <span class="pill" id="modePill">Mode: RAG</span>
      <label><input type="checkbox" id="modeToggle" checked /> Use Qdrant (RAG)</label>
    </div>
    <div class="pill">/api/ask (RAG) or /api/chat (LLM)</div>
  </div>

  <div id="chatbox" class="chatbox"></div>

  <div class="composer">
    <textarea id="input" placeholder="Type your message…"></textarea>
    <button id="sendBtn">Send</button>
  </div>

  <div class="hint">
    RAG mode: embed the question, search Qdrant, answer from retrieved context.
  </div>
</div>

<script>
  const chatbox = document.getElementById("chatbox");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const modeToggle = document.getElementById("modeToggle");
  const modePill = document.getElementById("modePill");

  let history = [];

  function addMessage(role, content) {
    const row = document.createElement("div");
    row.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;
    row.appendChild(bubble);
    chatbox.appendChild(row);
    chatbox.scrollTop = chatbox.scrollHeight;
    return bubble;
  }

  function updateModeLabel() {
    modePill.textContent = modeToggle.checked ? "Mode: RAG" : "Mode: LLM";
  }
  modeToggle.addEventListener("change", updateModeLabel);
  updateModeLabel();

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    sendBtn.disabled = true;

    addMessage("user", text);
    const thinking = addMessage("assistant", "Thinking...");

    try {
      const useRag = modeToggle.checked;
      const url = useRag ? "/api/ask" : "/api/chat";
      const body = useRag ? { question: text, top_k: 6 } : { message: text, history };

      const resp = await fetch(url, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Request failed");

      thinking.textContent = data.answer || "(empty response)";

      if (!useRag) {
        history.push({role:"user", content:text});
        history.push({role:"assistant", content:data.answer || ""});
      }
    } catch (err) {
      thinking.textContent = "Error: " + err.message;
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  addMessage("assistant", "System ready. Use Upload JSONL to ingest data, then ask questions.");
</script>
</body>
</html>
HTML

# upload.html
cat > "${APP_NAME}/templates/${APP_NAME}/upload.html" <<'HTML'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Upload JSONL for RAG Upsert</title>
  <style>
    body { font-family: Arial, sans-serif; margin:0; background:#0b0f14; color:#e6edf3; }
    header { padding:16px 18px; background:#111823; border-bottom:1px solid #1e2a3a; display:flex; gap:12px; align-items:center; }
    header a { color:#9fb0c3; text-decoration:none; border:1px solid #1e2a3a; padding:6px 10px; border-radius:10px; }
    header h1 { margin:0; font-size:16px; font-weight:600; flex:1; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 16px; }
    .card { background:#0f1621; border:1px solid #1e2a3a; border-radius:12px; padding:14px; }
    label { display:block; margin-top:10px; color:#9fb0c3; font-size:13px; }
    input[type="text"], input[type="file"] {
      width:100%; padding:10px 12px; border-radius:12px;
      border:1px solid #1e2a3a; background:#0b0f14; color:#e6edf3;
      margin-top:6px;
    }
    button { margin-top:12px; border-radius:12px; border:1px solid #1e2a3a; background:#0b5cff; color:white;
             font-weight:600; cursor:pointer; padding:10px 14px; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    pre { background:#0b0f14; border:1px solid #1e2a3a; padding:12px; border-radius:12px; overflow:auto; }
    .hint { color:#9fb0c3; font-size:12px; margin-top:10px; }
  </style>
</head>
<body>
<header>
  <a href="/">Chat</a>
  <h1>Upload JSONL → Upsert to Qdrant</h1>
</header>

<div class="wrap">
  <div class="card">
    <label>JSONL file (one JSON per line, must contain "prompt")</label>
    <input id="file" type="file" accept=".jsonl" />

    <label>Qdrant collection (optional)</label>
    <input id="collection" type="text" placeholder="dw_text" />

    <label>Embedding model (optional)</label>
    <input id="embed_model" type="text" placeholder="nomic-embed-text:latest" />

    <button id="btn">Upload & Upsert</button>

    <div class="hint">
      Upload time depends on file size. Each line calls Ollama embeddings + Qdrant upsert.
    </div>

    <h3>Result</h3>
    <pre id="out">{}</pre>
  </div>
</div>

<script>
  const fileEl = document.getElementById("file");
  const collectionEl = document.getElementById("collection");
  const modelEl = document.getElementById("embed_model");
  const btn = document.getElementById("btn");
  const out = document.getElementById("out");

  btn.addEventListener("click", async () => {
    if (!fileEl.files.length) {
      out.textContent = JSON.stringify({error:"Please choose a .jsonl file"}, null, 2);
      return;
    }

    btn.disabled = true;
    out.textContent = JSON.stringify({status:"Uploading..."}, null, 2);

    try {
      const fd = new FormData();
      fd.append("file", fileEl.files[0]);

      if (collectionEl.value.trim()) fd.append("collection", collectionEl.value.trim());
      if (modelEl.value.trim()) fd.append("embed_model", modelEl.value.trim());

      const resp = await fetch("/api/upload-jsonl", { method:"POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Upload failed");

      out.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      out.textContent = JSON.stringify({error: e.message}, null, 2);
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>
HTML

# -------------------------
# Migrate + run
# -------------------------
echo "==> Running migrations"
python manage.py migrate >/dev/null

echo
echo "==> SETUP COMPLETE"
echo "Project root: $(pwd)"
echo
echo "Defaults (override via env):"
echo "  export OLLAMA_URL=${OLLAMA_URL_DEFAULT}"
echo "  export EMBED_MODEL=${EMBED_MODEL_DEFAULT}"
echo "  export GEN_MODEL=${GEN_MODEL_DEFAULT}"
echo "  export QDRANT_URL=${QDRANT_URL_DEFAULT}"
echo "  export QDRANT_COLLECTION=${QDRANT_COLLECTION_DEFAULT}"
echo
echo "Open:"
echo "  http://<server-ip>:${PORT}/        (Chat)"
echo "  http://<server-ip>:${PORT}/upload  (Upload JSONL)"
echo

export OLLAMA_URL="${OLLAMA_URL_DEFAULT}"
export EMBED_MODEL="${EMBED_MODEL_DEFAULT}"
export GEN_MODEL="${GEN_MODEL_DEFAULT}"
export QDRANT_URL="${QDRANT_URL_DEFAULT}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION_DEFAULT}"

echo "==> Starting Django server on ${BIND_ADDR}:${PORT}"
python manage.py runserver "${BIND_ADDR}:${PORT}"
