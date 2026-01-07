python - <<'PY'

import requests, json



r = requests.get("http://10.50.1.29:31134/api/tags", timeout=30)

print(json.dumps(r.json(), indent=2))

PY



python - <<'PY'

import requests, json



r = requests.get("http://10.50.1.29:31134/api/tags", timeout=30)

print(json.dumps(r.json(), indent=2))

PY



python - <<'PY'

import requests, json



OLLAMA_URL = "http://10.50.1.29:31134"

MODEL = "qwen2.5-1.5b:latest"



r = requests.post(

    f"{OLLAMA_URL}/api/chat",

    json={

        "model": MODEL,

        "messages": [

            {"role": "user", "content": "Say OK"}

        ],

        "stream": False

    },

    timeout=120

)



print("STATUS:", r.status_code)

print(json.dumps(r.json(), indent=2)[:800])

PY

