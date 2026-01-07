
#!/usr/bin/env bash

set -euo pipefail



QDRANT_HOST="${QDRANT_HOST:-10.50.1.29}"

QDRANT_PORT="${QDRANT_PORT:-31633}"

COLLECTION_NAME="${COLLECTION_NAME:-dw_text}"

VECTOR_SIZE="${VECTOR_SIZE:-768}"

DISTANCE="${DISTANCE:-Cosine}"



echo "Creating Qdrant collection: ${COLLECTION_NAME}"

echo "Qdrant endpoint: http://${QDRANT_HOST}:${QDRANT_PORT}"

echo "Vector size: ${VECTOR_SIZE}, Distance: ${DISTANCE}"

echo "-----------------------------------------------"



JSON_BODY="{\"vectors\":{\"size\":${VECTOR_SIZE},\"distance\":\"${DISTANCE}\"}}"



curl -sS -X PUT "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${COLLECTION_NAME}" \

  -H "Content-Type: application/json" \

  --data-binary "${JSON_BODY}"



echo

echo "-----------------------------------------------"

echo "Verifying collections..."

curl -sS "http://${QDRANT_HOST}:${QDRANT_PORT}/collections"

echo

