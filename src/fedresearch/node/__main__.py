"""Run federated data nodes.

    python -m fedresearch.node                  # all nodes in config/nodes.yaml
    python -m fedresearch.node hospital_central # a single node
"""

import argparse
import subprocess
import sys
from pathlib import Path

import uvicorn

from fedresearch.node.app import create_app
from fedresearch.node.config import DEFAULT_CONFIG, load_node_configs
from fedresearch.node.engine import NodeEngine
from fedresearch.node.privacy import AuditLog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated data nodes.")
    parser.add_argument("names", nargs="*", help="node names to run (default: all)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    configs = load_node_configs(args.config)
    names = args.names or list(configs)
    unknown = set(names) - set(configs)
    if unknown:
        parser.error(f"unknown node(s): {', '.join(sorted(unknown))}")

    if len(names) == 1:
        cfg = configs[names[0]]
        db = args.data_dir / "nodes" / f"{cfg.name}.duckdb"
        if not db.exists():
            parser.error(f"{db} not found — run `python -m fedresearch.data.generate` first")
        engine = NodeEngine(cfg.name, db, cfg.policy, cfg.description)
        audit = AuditLog(args.data_dir / "audit" / f"{cfg.name}.jsonl")
        uvicorn.run(create_app(engine, audit), host=args.host, port=cfg.port)
        return

    common = ["--config", str(args.config), "--data-dir", str(args.data_dir), "--host", args.host]
    procs = [subprocess.Popen([sys.executable, "-m", "fedresearch.node", name, *common]) for name in names]
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
