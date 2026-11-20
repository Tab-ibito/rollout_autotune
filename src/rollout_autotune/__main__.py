"""Run the threshold controller as a subprocess using newline-delimited JSON over stdio."""

import argparse
import json
import sys
from dataclasses import asdict

from . import Config, Window, run

_WINDOW_FIELDS = frozenset(Window.__dataclass_fields__)


def exchange(op: str, config: Config):
    """Send one JSON request ``{"op": op, "k": ..., "g": ...}`` and parse one reply line."""
    print(json.dumps({"op": op, **asdict(config)}), flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError(f"runtime disconnected during {op}")
    return json.loads(line)


class StdioRuntime:
    """Adapter bridging the Python ``Runtime`` protocol to line-delimited JSON over stdio."""

    def configure(self, target: Config) -> bool:
        accepted = exchange("configure", target)
        if type(accepted) is not bool:
            raise ValueError("configure requires a JSON boolean acknowledgement")
        return accepted

    def observe(self, current: Config) -> Window | None:
        data = exchange("observe", current)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("observe requires a JSON object or null")
        # Ignore extra diagnostic fields emitted by the host.
        return Window(**{k: v for k, v in data.items() if k in _WINDOW_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-replicas", type=int, default=4)
    args = parser.parse_args()
    run(StdioRuntime(), limits=Config(args.max_workers, args.max_replicas))


if __name__ == "__main__":
    main()
