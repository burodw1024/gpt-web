#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# COMPLETE CLEAN SH: CREATE DJANGO UI + RAG (OLLAMA+QDRANT)
# ==========================================================
# Creates:
#  - Django project + app
#  - Nice UI at /
#  - /api/chat (LLM only, optional)
#  - /api/ask  (RAG: embeddings -> Qdrant -> Ollama generate)
#
# Defaults for your cluster:
#  - Ollama:  http://10.109.199.247:11434   (Ollama 0.3.14)
#  - Qdrant:  http://10.105.26.13:6333
#  - Collection: dw_text
#  - Embed model: nomic-embed-text:latest
#  - Gen model: qwen2.5-1.5b:latest
#
# Run:
#   chmod +x create_buro_rag_ui.sh
#   ./create_buro_rag_ui.sh
#
# Override if needed:
#   PROJECT_DIR=buro_llm_chat2 PORT=8002 ./create_buro_rag_ui.sh
#   OLLAMA_URL_DEFAULT=http://<NODE_IP>:31134 QDRANT_URL_DEFAULT=http://<NODE_IP>:31633 ./create_buro_rag_ui.sh

# -------------------------
# CONFIG (override via env)
# -------------------------
PROJECT_DIR="${PROJECT_DIR:-buro_llm_chat}"
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
# PRECHECK
# -------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  exit 1
fi

echo "==> Creating project in: ${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

if [[ -f "manage.py" ]]; then
  echo "ERROR: manage.py already exists in $(pwd). Use a new PROJECT_DIR."
  exit 1
fi

echo "==> Creating venv"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip >/dev/null
pip install "django==4.2.*" requests >/dev/null

echo "==> Creating Django project: ${DJANGO_PROJECT}"
django-admin startproject "${DJANGO_PROJECT}" .

echo "==> Creating app: ${APP_NAME}"
python manage.py startapp "${APP_NAME}"

# -------------------------
# settings.py (CORRECT)
# -------------------------
echo "==> Updating settings.py"
SETTINGS_FILE="${DJANGO_PROJECT}/settings.py"

python - <<PY
import pathlib, re

app = "${APP_NAME}"
settings_path = pathlib.Path("${SETTINGS_FILE}")
s = settings_path.read_text(encoding="utf-8")

app_config = f"{app}.apps.{app.capitalize()}Config"

# Insert app_config as a literal string
if app_config not in s:
    s = re.sub(
        r"INSTALLED_APPS\s*=\s*\[\n",
        "INSTALLED_APPS = [\n    " + repr(app_config) + ",\n",
        s,
        count=1
    )

# Allow all hosts for LAN testing
if "ALLOWED_HOSTS" in s:
    s = re.sub(r"ALLOWED_HOSTS\s*=\s*\[[^\]]*\]", 'ALLOWED_HOSTS = ["*"]', s, count=1)
else:
    s += '\nALLOWED_HOSTS = ["*"]\n'

settings_path.write_text(s, encoding="utf-8")
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
# urls.py
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

echo "==> Writing app urls.py"
cat > "${APP_NAME}/urls.py" <<'PY'
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat", views.api_chat, name="api_chat"),
    path("api/ask", views.api_ask, name="api_ask"),
]
PY

# -------------------------
# views.py (NO BAD ESCAPES)
# -------------------------
echo "==> Writing views.py"
cat > "${APP_NAME}/views.py" <<PY
import json
import os
import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

OLLAMA_URL = os.getenv("OLLAMA_URL", "${OLLAMA_URL_DEFAULT}").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "${EMBED_MODEL_DEFAULT}")
GEN_MODEL = os.getenv("GEN_MODEL", "${GEN_MODEL_DEFAULT}")

QDRANT_URL = os.getenv("QDRANT_URL", "${QDRANT_URL_DEFAULT}").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "${QDRANT_COLLECTION_DEFAULT}")

def index(request):
    return render(request, "${APP_NAME}/index.html")

def _ollama_embeddings(text: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]

def _qdrant_search(vector, limit=6):
    r = requests.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        json={
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("result", []) or []

def _dedupe_hits(hits):
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
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 250},
        },
        timeout=300,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

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
            "Conversation:"
        ]
        for m in history[-6:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            parts.append(("User: " if role == "user" else "Assistant: ") + content)

        parts.append("User: " + user_message)
        parts.append("Assistant:")
        answer = _ollama_generate("\n".join(parts))
        return JsonResponse({"answer": answer})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Upstream request failed: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

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

        context = "\n\n".join(ctx_blocks)

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
PY

# -------------------------
# Template
# -------------------------
echo "==> Writing template"
mkdir -p "${APP_NAME}/templates/${APP_NAME}"

cat > "${APP_NAME}/templates/${APP_NAME}/index.html" <<'HTML'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Buro First On-Prem Kubernetes Deployed LLM Private</title>
  <style>
    body { font-family: Arial, sans-serif; margin:0; background:#0b0f14; color:#e6edf3; }
    header { padding:16px 18px; background:#111823; border-bottom:1px solid #1e2a3a; }
    header h1 { margin:0; font-size:16px; font-weight:600; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 16px; }
    .bar { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
    .toggle { display:flex; gap:8px; align-items:center; font-size:13px; color:#9fb0c3; }
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
    .pill { padding:4px 8px; border:1px solid #1e2a3a; border-radius:999px; font-size:12px; color:#9fb0c3; }
  </style>
</head>
<body>
  <header>
    <h1>Buro First On-Prem Kubernetes Deployed LLM Private</h1>
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
      RAG mode: embeds your question, searches Qdrant, then answers from retrieved context.
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

  addMessage("assistant", "Private on-prem LLM is ready. Ask a question. (RAG mode ON by default.)");
</script>
</body>
</html>
HTML

# -------------------------
# MIGRATE & RUN
# -------------------------
echo "==> Running migrations"
python manage.py migrate >/dev/null

echo
echo "==> DONE"
echo "Project created at: $(pwd)"
echo
echo "Defaults:"
echo "  OLLAMA_URL=${OLLAMA_URL_DEFAULT}"
echo "  EMBED_MODEL=${EMBED_MODEL_DEFAULT}"
echo "  GEN_MODEL=${GEN_MODEL_DEFAULT}"
echo "  QDRANT_URL=${QDRANT_URL_DEFAULT}"
echo "  QDRANT_COLLECTION=${QDRANT_COLLECTION_DEFAULT}"
echo
echo "==> Starting Django server on ${BIND_ADDR}:${PORT}"
echo

export OLLAMA_URL="${OLLAMA_URL_DEFAULT}"
export EMBED_MODEL="${EMBED_MODEL_DEFAULT}"
export GEN_MODEL="${GEN_MODEL_DEFAULT}"
export QDRANT_URL="${QDRANT_URL_DEFAULT}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION_DEFAULT}"

python manage.py runserver "${BIND_ADDR}:${PORT}"
