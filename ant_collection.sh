curl -sS -X PUT "http://10.50.1.29:31633/collections/dw_text" -H "Content-Type: application/json" --data-binary '{"vectors":{"size":768,"distance":"Cosine"}}'

