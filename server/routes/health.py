import os
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    from config import config
    return {"status": "ok", "version": config.APP_VERSION}


@router.get("/api/debug/env")
def debug_env():
    """Return env var names (not values) to diagnose Railway configuration."""
    keys_present = [k for k in os.environ if k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "PORT", "USE_GPT")]
    all_key_names = sorted(os.environ.keys())
    return {
        "target_keys": {
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        },
        "all_env_var_names": all_key_names,
    }
