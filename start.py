import os
import sys

print(f"Python {sys.version}", flush=True)
print("Starting uvicorn...", flush=True)

import uvicorn

port = int(os.environ.get("PORT", 8080))
print(f"Binding to 0.0.0.0:{port}", flush=True)

uvicorn.run("server.main:app", host="0.0.0.0", port=port, log_level="info")
