# chat/views.py

import json

import os

import uuid

import re

import requests



from django.conf import settings

from django.contrib.admin.views.decorators import staff_member_required

from django.http import JsonResponse, StreamingHttpResponse, HttpResponseServerError

from django.shortcuts import render, redirect

from django.views.decorators.csrf import csrf_exempt

from django.views.decorators.http import require_POST, require_http_methods



# ---------------------------------------------------------

# NiFi source (optional)

# ---------------------------------------------------------

NIFI_URL = os.getenv("NIFI_URL", "http://192.168.1.139:9999/actors?id=all")



# ---------------------------------------------------------

# Defaults (override via env vars)

# ---------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.1.99:31134").rstrip("/")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.1.176:31633").rstrip("/")



EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest").strip()

GEN_MODEL = os.getenv("GEN_MODEL", "qwen2.5-1.5b:local").strip()



QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dw_text").strip()



EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "220"))

QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "90"))

GEN_TIMEOUT = int(os.getenv("GEN_TIMEOUT", "900"))



# ---------------------------------------------------------

# Pages

# ---------------------------------------------------------

def index(request):

    return render(request, "chat/index.html")





def upload_page(request):

    return render(request, "chat/upload.html")





# ---------------------------------------------------------

# Ollama helpers

# ---------------------------------------------------------

def _ollama_embeddings_with_model(model: str, text: str):

    r = requests.post(

        f"{OLLAMA_URL}/api/embeddings",

        json={"model": model, "prompt": text},

        timeout=EMBED_TIMEOUT,

    )

    r.raise_for_status()

    data = r.json()

    if "embedding" not in data:

        raise RuntimeError(f"Ollama embeddings response missing 'embedding': {data}")

    return data["embedding"]





def _ollama_embeddings(text: str):

    return _ollama_embeddings_with_model(EMBED_MODEL, text)





def _ollama_generate(prompt: str):

    r = requests.post(

        f"{OLLAMA_URL}/api/chat",

        json={

            "model": GEN_MODEL,

            "messages": [

                {"role": "system", "content": "You are a helpful assistant."},

                {"role": "user", "content": prompt},

            ],

            "stream": False,

            "options": {

                "temperature": 0.2,

                "top_p": 0.9,

                "num_predict": 280,

            },

        },

        timeout=GEN_TIMEOUT,

    )

    r.raise_for_status()

    data = r.json()

    return (data.get("message", {}).get("content") or "").strip()





# ---------------------------------------------------------

# Qdrant helpers

# ---------------------------------------------------------

def _qdrant_search(vector, limit=6):

    """Vector similarity search (Top-K)."""

    r = requests.post(

        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",

        json={

            "vector": vector,

            "limit": int(limit),

            "with_payload": True,

            "with_vector": False,

        },

        timeout=QDRANT_TIMEOUT,

    )

    r.raise_for_status()

    return r.json().get("result", []) or []





def _qdrant_upsert(collection: str, pid: str, vector, payload: dict):

    body = {"points": [{"id": pid, "vector": vector, "payload": payload}]}

    r = requests.put(

        f"{QDRANT_URL}/collections/{collection}/points?wait=true",

        json=body,

        timeout=EMBED_TIMEOUT,

    )

    r.raise_for_status()





def _qdrant_scroll(batch_size=200, offset=None, with_payload=True, with_vectors=False):

    """Scroll API to fetch points without similarity ranking."""

    body = {

        "limit": int(batch_size),

        "with_payload": with_payload,

        "with_vectors": with_vectors,

    }

    if offset is not None:

        body["offset"] = offset



    r = requests.post(

        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",

        json=body,

        timeout=QDRANT_TIMEOUT,

    )

    r.raise_for_status()

    return r.json().get("result", {}) or {}





def _qdrant_scroll_all(max_points=None, batch_size=200):

    """Fetch up to max_points (None => all points) from the collection."""

    all_points = []

    offset = None



    while True:

        result = _qdrant_scroll(batch_size=batch_size, offset=offset, with_payload=True, with_vectors=False)

        points = result.get("points", []) or []

        all_points.extend(points)



        if max_points is not None and len(all_points) >= int(max_points):

            return all_points[: int(max_points)]



        offset = result.get("next_page_offset")

        if not offset or not points:

            break



    return all_points





def _dedupe_hits(hits):

    # Dedupe by prefix of payload.prompt (helps when same rows re-uploaded)

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





# ---------------------------------------------------------

# Math helpers (for "calculate using ALL employees")

# ---------------------------------------------------------

_salary_re = re.compile(r"\bBASICSALARY\s*:\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)



def _employee_key(payload: dict):

    for key in ("EMPLOYEEID", "employeeid", "employee_id", "EmployeeId"):

        v = payload.get(key)

        if v is not None and str(v).strip() != "":

            return str(v).strip()

    return None





def _extract_salary(payload: dict):

    # Prefer structured field

    for key in ("BASICSALARY", "basicSalary", "SALARY", "salary"):

        if key in payload and str(payload.get(key)).strip() != "":

            try:

                return float(str(payload.get(key)).replace(",", "").strip())

            except Exception:

                pass



    # Fallback: parse from prompt text

    txt = payload.get("prompt") or ""

    m = _salary_re.search(txt)

    if m:

        try:

            return float(m.group(1))

        except Exception:

            return None



    return None





def _compute_salary_stats_from_points(points):

    seen_emp = set()

    salary_sum = 0.0

    salary_count = 0

    min_salary = None

    max_salary = None



    for p in points:

        payload = p.get("payload") or {}



        emp_id = _employee_key(payload)

        if emp_id:

            if emp_id in seen_emp:

                continue

            seen_emp.add(emp_id)



        sal = _extract_salary(payload)

        if sal is None:

            continue



        salary_sum += sal

        salary_count += 1

        if min_salary is None or sal < min_salary:

            min_salary = sal

        if max_salary is None or sal > max_salary:

            max_salary = sal



    employee_count = len(seen_emp) if seen_emp else salary_count

    avg_salary = (salary_sum / employee_count) if employee_count else 0.0



    return {

        "employee_count": employee_count,

        "salary_values_found": salary_count,

        "total_salary": salary_sum,

        "avg_salary": avg_salary,

        "min_salary": min_salary,

        "max_salary": max_salary,

        "dedup_by_employeeid": bool(seen_emp),

    }





# ---------------------------------------------------------

# API: LLM-only chat (no RAG)

# ---------------------------------------------------------

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



        answer = _ollama_generate("\n".join(parts))

        return JsonResponse({"answer": answer})



    except requests.exceptions.RequestException as e:

        return JsonResponse({"error": f"Upstream request failed: {str(e)}"}, status=502)

    except Exception as e:

        return JsonResponse({"error": str(e)}, status=500)





# ---------------------------------------------------------

# API: Math mode (ALL employees) - no LLM truncation

# ---------------------------------------------------------

@csrf_exempt

def api_math(request):

    """

    POST JSON:

      {

        "op": "salary_stats" | "total_salary" | "avg_salary" | "count_employees",

        "max_points": 500

      }

    """

    if request.method != "POST":

        return JsonResponse({"error": "POST required"}, status=405)



    try:

        payload = json.loads(request.body.decode("utf-8"))

        op = (payload.get("op") or "salary_stats").strip().lower()

        max_points = int(payload.get("max_points", 500))



        points = _qdrant_scroll_all(max_points=max_points, batch_size=200)

        stats = _compute_salary_stats_from_points(points)

        stats["points_scanned"] = len(points)



        if op == "count_employees":

            return JsonResponse({"employee_count": stats["employee_count"], "meta": stats})



        if op == "total_salary":

            return JsonResponse({"total_salary": stats["total_salary"], "meta": stats})



        if op == "avg_salary":

            return JsonResponse({"avg_salary": stats["avg_salary"], "total_salary": stats["total_salary"], "meta": stats})



        return JsonResponse(stats)



    except requests.exceptions.RequestException as e:

        return JsonResponse({"error": f"Upstream request failed: {str(e)}"}, status=502)

    except Exception as e:

        return JsonResponse({"error": str(e)}, status=500)





# ---------------------------------------------------------

# API: RAG (embed -> qdrant search -> generate)

# + UI checkbox can request math_mode which routes to api_math

# ---------------------------------------------------------

@csrf_exempt

def api_ask(request):

    if request.method != "POST":

        return JsonResponse({"error": "POST required"}, status=405)



    try:

        payload = json.loads(request.body.decode("utf-8"))

        question = (payload.get("question") or payload.get("message") or "").strip()

        top_k = int(payload.get("top_k", 6))

        math_mode = bool(payload.get("math_mode", False))

        max_points = int(payload.get("max_points", 500))



        if not question:

            return JsonResponse({"error": "question is required"}, status=400)



        # If checkbox enabled: route "math questions" to api_math logic

        if math_mode:

            q = question.lower()

            if ("total" in q or "sum" in q) and ("salary" in q or "basic" in q):

                op = "total_salary"

            elif ("average" in q or "avg" in q) and ("salary" in q or "basic" in q):

                op = "avg_salary"

            elif "count" in q and ("employee" in q or "employees" in q):

                op = "count_employees"

            else:

                op = "salary_stats"



            points = _qdrant_scroll_all(max_points=max_points, batch_size=200)

            stats = _compute_salary_stats_from_points(points)

            stats["points_scanned"] = len(points)



            if op == "total_salary":

                return JsonResponse({"answer": f"Total salary (from all employees) = {stats['total_salary']}", "meta": stats})

            if op == "avg_salary":

                return JsonResponse({"answer": f"Average salary (from all employees) = {stats['avg_salary']}", "meta": stats})

            if op == "count_employees":

                return JsonResponse({"answer": f"Employee count = {stats['employee_count']}", "meta": stats})



            return JsonResponse({

                "answer": f"Employees={stats['employee_count']}, Total={stats['total_salary']}, Avg={stats['avg_salary']}, Min={stats['min_salary']}, Max={stats['max_salary']}",

                "meta": stats

            })



        # Normal RAG path (Top-K)

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

If answer not in CONTEXT, say: .



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





# ---------------------------------------------------------

# NiFi -> JSONL export helpers

# ---------------------------------------------------------

MAX_PROMPT_CHARS = 6000



def build_prompt_all_fields(obj: dict) -> str:

    parts = []

    for k, v in obj.items():

        if v is None:

            continue

        if isinstance(v, str) and v.strip() == "":

            continue

        if isinstance(v, (dict, list)):

            v = json.dumps(v, ensure_ascii=False)

        parts.append(f"{k}: {v}")



    prompt = " | ".join(parts)

    if len(prompt) > MAX_PROMPT_CHARS:

        prompt = prompt[:MAX_PROMPT_CHARS] + " &(truncated)"

    return prompt





def stable_uuid_from_employee(obj: dict) -> str:

    emp = obj.get("EMPLOYEEID")

    if emp is not None and str(emp).strip() != "":

        base = f"employee:{str(emp).strip()}"

    else:

        base = json.dumps(obj, sort_keys=True, ensure_ascii=False)

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))





def export_actors_jsonl(request):

    try:

        r = requests.get(NIFI_URL, timeout=180)

        r.raise_for_status()

        data = r.json()

    except Exception as e:

        return HttpResponseServerError(f"NiFi API error: {e}")



    if not isinstance(data, list):

        data = [data] if isinstance(data, dict) else []



    data = [x for x in data if isinstance(x, dict) and x]



    def stream():

        for obj in data:

            line = {

                "id": stable_uuid_from_employee(obj),

                "prompt": build_prompt_all_fields(obj),

                "source": "nifi_actors",

            }

            yield json.dumps(line, ensure_ascii=False) + "\n"



    resp = StreamingHttpResponse(stream(), content_type="application/x-ndjson; charset=utf-8")

    resp["Content-Disposition"] = 'attachment; filename="actors_uuid_for_rag.jsonl"'

    resp["Cache-Control"] = "no-store"

    return resp





# ---------------------------------------------------------

# API: Upload JSONL -> embed -> upsert to Qdrant

# ---------------------------------------------------------

@csrf_exempt

@require_POST

def api_upload_jsonl(request):

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



        return JsonResponse(

            {

                "success": ok,

                "failed": failed,

                "errors": errors,

                "collection": collection,

                "embed_model": embed_model,

                "ollama_url": OLLAMA_URL,

                "qdrant_url": QDRANT_URL,

                "gen_model": GEN_MODEL,

            }

        )



    except Exception as e:

        return JsonResponse({"error": str(e)}, status=500)





# ---------------------------------------------------------

# Admin: Qdrant collection management page

# ---------------------------------------------------------

QDRANT_BASE = getattr(settings, "QDRANT_BASE", QDRANT_URL)

ADMIN_COLLECTION = getattr(settings, "QDRANT_COLLECTION", QDRANT_COLLECTION)

QDRANT_VECTOR_SIZE = getattr(settings, "QDRANT_VECTOR_SIZE", 768)

QDRANT_DISTANCE = getattr(settings, "QDRANT_DISTANCE", "Cosine")





def _qdrant_get_collection_info():

    r = requests.get(f"{QDRANT_BASE}/collections/{ADMIN_COLLECTION}", timeout=30)

    r.raise_for_status()

    return r.json()





def _qdrant_delete_collection():

    r = requests.delete(f"{QDRANT_BASE}/collections/{ADMIN_COLLECTION}", timeout=30)

    if r.status_code not in (200, 404):

        r.raise_for_status()

    return r.json() if r.text else {"result": True}





def _qdrant_create_collection():

    payload = {"vectors": {"size": QDRANT_VECTOR_SIZE, "distance": QDRANT_DISTANCE}}

    r = requests.put(

        f"{QDRANT_BASE}/collections/{ADMIN_COLLECTION}",

        json=payload,

        timeout=30,

    )

    r.raise_for_status()

    return r.json()





@staff_member_required

@require_http_methods(["GET", "POST"])

def qdrant_admin(request):

    message = None

    error = None



    if request.method == "POST":

        action = request.POST.get("action")

        confirm = request.POST.get("confirm", "").strip()



        if confirm != ADMIN_COLLECTION:

            error = f"Confirmation failed. Type '{ADMIN_COLLECTION}' exactly to proceed."

        else:

            try:

                if action == "delete":

                    _qdrant_delete_collection()

                    message = f"Deleted collection '{ADMIN_COLLECTION}'."

                elif action == "create":

                    _qdrant_create_collection()

                    message = f"Created collection '{ADMIN_COLLECTION}'."

                elif action == "recreate":

                    _qdrant_delete_collection()

                    _qdrant_create_collection()

                    message = f"Recreated collection '{ADMIN_COLLECTION}'."

                else:

                    error = "Unknown action."

            except Exception as e:

                error = f"Qdrant operation failed: {e}"



        return redirect("qdrant_admin")



    info = None

    try:

        info = _qdrant_get_collection_info()

    except Exception as e:

        error = f"Could not read collection info: {e}"



    ctx = {

        "qdrant_base": QDRANT_BASE,

        "collection": ADMIN_COLLECTION,

        "vector_size": QDRANT_VECTOR_SIZE,

        "distance": QDRANT_DISTANCE,

        "info": info,

        "message": message,

        "error": error,

    }

    return render(request, "chat/qdrant_admin.html", ctx)

