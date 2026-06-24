#!/bin/bash
# Smoke test against a running local or Railway server.
# Usage: ./scripts/smoke_test.sh [BASE_URL]
# Defaults to http://localhost:8080

BASE="${1:-http://localhost:8080}"
PASS=0
FAIL=0

check() {
  local desc="$1" expected="$2"
  shift 2
  local actual
  actual=$(curl -s -o /dev/null -w "%{http_code}" "$@")
  if [ "$actual" = "$expected" ]; then
    echo "  PASS  $desc ($actual)"
    ((PASS++))
  else
    echo "  FAIL  $desc (expected $expected, got $actual)"
    ((FAIL++))
  fi
}

echo "=== Smoke test: $BASE ==="

check "GET /health returns 200"       200 "$BASE/health"
check "GET /         returns 200"     200 "$BASE/"
check "GET /app.js   returns 200"     200 "$BASE/app.js"
check "GET /api/status/bad-id → 404"  404 "$BASE/api/status/does-not-exist"
check "GET /api/download/bad-id → 404" 404 "$BASE/api/download/does-not-exist"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
