#!/usr/bin/env sh
# Request walkthrough for Linux and macOS. Start the API first:
#   uvicorn app.main:create_app --factory --port 8000
BASE="${BASE:-http://127.0.0.1:8000}"
BUYER="X-API-Key: demo-harbor-buyer"

ask() {
  printf '\n=== %s ===\n' "$1"
  curl -s -X POST "$BASE/ask" -H "$BUYER" -H "Content-Type: application/json" \
    -d "{\"question\": \"$1\"}" | python3 -m json.tool
}

curl -s "$BASE/health" | python3 -m json.tool
ask "What approval is needed for a 7,000 AUD laptop order?"
ask "What is the status of PR-1042 and what did Bluegate quote for LAP-14-STD?"
ask "What is Redline Freight's negotiated price for LAP-14-STD?"
ask "When can the LAP-14-STD laptops for PR-1042 ship?"

printf '\n=== body cannot choose the tenant ===\n'
curl -s -X POST "$BASE/ask" -H "$BUYER" -H "Content-Type: application/json" \
  -d '{"question": "Redline price list?", "tenant_id": "redline"}' | python3 -m json.tool
