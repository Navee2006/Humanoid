"""
orin1/runner.py
────────────────────────────────────────────────────────────────
Single-process launcher: runs the perception agent (main.py)
AND the snapshot HTTP server in the same Python process.

Usage:
    python -m orin1.runner

The snapshot server shares _latest_frame with the agent in-process
(no IPC needed). Uvicorn runs in a background thread.
"""

import asyncio, logging, threading
import uvicorn

log = logging.getLogger("runner")


def _start_snapshot_server():
    """Run snapshot server in background thread."""
    config = uvicorn.Config(
        "orin1.snapshot_server:app",
        host="0.0.0.0",
        port=5553,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Orin1] %(levelname)s %(message)s",
    )

    # Start snapshot HTTP server in background thread
    t = threading.Thread(target=_start_snapshot_server, daemon=True)
    t.start()
    log.info("Snapshot server started on :5553")

    # Run main perception agent (blocks)
    from orin1.main import Orin1Agent
    agent = Orin1Agent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.audio.close()
        log.info("Orin1 shut down")
