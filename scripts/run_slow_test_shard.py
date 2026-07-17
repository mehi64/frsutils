# SPDX-License-Identifier: BSD-3-Clause
"""Run one deterministic contiguous shard of the exhaustive slow test suite."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import subprocess
import sys


def collect_slow_test_nodeids(pytest_args: Sequence[str] = ()) -> list[str]:
    """Collect node IDs for tests marked ``slow``.

    Parameters
    ----------
    pytest_args : sequence of str, default=()
        Additional arguments passed to the collection command.

    Returns
    -------
    nodeids : list of str
        Collected slow-test node IDs in pytest collection order.

    Raises
    ------
    RuntimeError
        If collection succeeds without returning any slow-test node IDs.
    subprocess.CalledProcessError
        If pytest collection fails.
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-m",
        "slow",
        "--collect-only",
        "-q",
        *pytest_args,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    nodeids = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if not nodeids:
        raise RuntimeError("No tests marked 'slow' were collected.")
    return nodeids


def select_contiguous_shard(
    nodeids: Sequence[str],
    *,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    """Select one contiguous, non-overlapping shard of collected node IDs.

    Parameters
    ----------
    nodeids : sequence of str
        Node IDs in deterministic pytest collection order.
    shard_index : int
        Zero-based index of the shard to select.
    shard_count : int
        Total number of shards.

    Returns
    -------
    shard : list of str
        Node IDs assigned to the requested shard.

    Raises
    ------
    ValueError
        If the shard count or index is invalid, or the requested shard is empty.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "shard_index must satisfy 0 <= shard_index < shard_count."
        )

    start = shard_index * len(nodeids) // shard_count
    stop = (shard_index + 1) * len(nodeids) // shard_count
    shard = list(nodeids[start:stop])
    if not shard:
        raise ValueError(
            "The requested shard is empty; use fewer shards than collected tests."
        )
    return shard


def run_slow_test_shard(
    *,
    shard_index: int,
    shard_count: int,
    pytest_args: Sequence[str] = (),
) -> int:
    """Collect and execute one slow-test shard.

    Parameters
    ----------
    shard_index : int
        Zero-based index of the shard to execute.
    shard_count : int
        Total number of deterministic shards.
    pytest_args : sequence of str, default=()
        Additional arguments passed to both collection and execution.

    Returns
    -------
    return_code : int
        Exit status returned by the pytest execution process.
    """
    nodeids = collect_slow_test_nodeids(pytest_args)
    shard = select_contiguous_shard(
        nodeids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    print(
        f"Running slow-test shard {shard_index + 1}/{shard_count}: "
        f"{len(shard)} of {len(nodeids)} tests.",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-q",
        *pytest_args,
        *shard,
    ]
    return subprocess.call(command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for deterministic slow-test sharding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-index",
        type=int,
        required=True,
        help="Zero-based index of the shard to execute.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        required=True,
        help="Total number of deterministic shards.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional extra pytest arguments after '--'.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested slow-test shard and return its pytest exit status."""
    args = parse_args(argv)
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return run_slow_test_shard(
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        pytest_args=pytest_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
