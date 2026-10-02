from dataclasses import dataclass
from pathlib import Path

import yaml

from fedresearch.node.privacy import PrivacyPolicy

DEFAULT_CONFIG = Path("config/nodes.yaml")


@dataclass(frozen=True)
class NodeConfig:
    name: str
    description: str
    port: int
    policy: PrivacyPolicy


def load_node_configs(path: Path = DEFAULT_CONFIG) -> dict[str, NodeConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["nodes"]
    return {
        name: NodeConfig(
            name=name,
            description=cfg.get("description", ""),
            port=cfg["port"],
            policy=PrivacyPolicy(cfg["min_cell_count"], cfg.get("round_counts_to", 1)),
        )
        for name, cfg in raw.items()
    }
