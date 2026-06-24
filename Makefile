.PHONY: dev stop smoke test railway-logs railway-smoke

PORT ?= 8080
PID_FILE := /tmp/kmz-server.pid

# Start the FastAPI server locally (loads .env if present)
dev:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
	  echo "Server already running (PID $$(cat $(PID_FILE))). Run 'make stop' first."; \
	  exit 1; \
	fi
	@echo "Starting server on port $(PORT)..."
	@set -a; [ -f .env ] && . ./.env; set +a; \
	  PORT=$(PORT) python3 start.py > /tmp/kmz-server.log 2>&1 & echo $$! > $(PID_FILE)
	@echo "PID $$(cat $(PID_FILE)) — logs: tail -f /tmp/kmz-server.log"
	@sleep 2 && curl -s http://localhost:$(PORT)/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Health:', d)"

# Stop the local server
stop:
	@if [ -f $(PID_FILE) ]; then \
	  kill $$(cat $(PID_FILE)) 2>/dev/null && echo "Stopped." || echo "Already stopped."; \
	  rm -f $(PID_FILE); \
	else echo "No PID file found."; fi

# Run smoke tests against local server
smoke:
	@bash scripts/smoke_test.sh http://localhost:$(PORT)

# Run smoke tests against Railway production
railway-smoke:
	@bash scripts/smoke_test.sh https://kmz-location-scraper-production.up.railway.app

# Run unit tests
test:
	python3 -m pytest tests/ -v

# Tail Railway logs
railway-logs:
	railway logs
