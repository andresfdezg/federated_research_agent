from pathlib import Path

import pytest

from fedresearch.data.generate import generate
from fedresearch.node.config import load_node_configs
from fedresearch.node.engine import NodeEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    generate(out)
    return out


@pytest.fixture(scope="session")
def engines(data_dir) -> dict[str, NodeEngine]:
    configs = load_node_configs(ROOT / "config" / "nodes.yaml")
    return {
        name: NodeEngine(name, data_dir / "nodes" / f"{name}.duckdb", cfg.policy, cfg.description)
        for name, cfg in configs.items()
    }
