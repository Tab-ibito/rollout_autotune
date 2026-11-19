"""Run the controller as a subprocess using newline-delimited JSON over stdio."""

import argparse
import json
import sys
from dataclasses import asdict

from . import Config, Window, run


def exchange(op: str, config: Config):
    print(json.dumps({"op": op, **asdict(config)}), flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError(f"runtime disconnected during {op}")
    return json.loads(line)


class StdioRuntime:
    def configure(self, target: Config) -> bool:
        accepted = exchange("configure", target)
        if type(accepted) is not bool:
            raise ValueError("configure requires a JSON boolean acknowledgement")
        return accepted

    def observe(self, current: Config) -> Window | None:
        data = exchange("observe", current)
        return None if data is None else Window(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-replicas", type=int, default=4)
    args = parser.parse_args()
    run(StdioRuntime(), limits=Config(args.max_workers, args.max_replicas))


if __name__ == "__main__":
    main()
