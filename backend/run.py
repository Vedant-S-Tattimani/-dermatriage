"""
Uvicorn launcher — run from the backend/ directory:

    python run.py
    python run.py --host 0.0.0.0 --port 8000 --reload
"""
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Medical Assistant API")
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=8001)
    parser.add_argument("--reload",  action="store_true", help="Enable hot-reload (dev mode)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )
