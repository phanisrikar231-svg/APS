from __future__ import annotations

import argparse

from fedmsme.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FedMSME-PdM demo web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()

