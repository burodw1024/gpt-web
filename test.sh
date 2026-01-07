python - <<'PY'

import requests, json

OLLAMA="http://10.50.1.29:31134"

MODEL="qwen2.5-1.5b-instruct:q4_0"

r=requests.post(f"{OLLAMA}/api/chat",

                json={"model":MODEL,"messages":[{"role":"user","content":"Say OK"}],"stream":False},

                timeout=120)

print("STATUS:", r.status_code)

print(json.dumps(r.json(), indent=2)[:800])

PY

