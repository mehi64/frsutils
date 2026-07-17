# SPDX-License-Identifier: BSD-3-Clause
"""Tests for deterministic sharding of the exhaustive slow test suite."""

from __future__ import annotations

import pytest

from scripts.run_slow_test_shard import select_contiguous_shard


def test_contiguous_shards_cover_collection_once_and_in_order() -> None:
    """All shards should reconstruct the collection without gaps or duplicates."""
    nodeids = [f"test::{index}" for index in range(10)]

    shards = [
        select_contiguous_shard(
            nodeids,
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    ]

    assert [item for shard in shards for item in shard] == nodeids
    assert [len(shard) for shard in shards] == [2, 3, 2, 3]


@pytest.mark.parametrize(
    "shard_index,shard_count,match",
    [
        (0, 0, "shard_count"),
        (-1, 4, "shard_index"),
        (4, 4, "shard_index"),
    ],
)
def test_invalid_shard_configuration_is_rejected(
    shard_index: int,
    shard_count: int,
    match: str,
) -> None:
    """Invalid counts and indexes should fail before invoking pytest."""
    with pytest.raises(ValueError, match=match):
        select_contiguous_shard(
            ["test::one", "test::two"],
            shard_index=shard_index,
            shard_count=shard_count,
        )


def test_empty_shard_is_rejected() -> None:
    """A shard count larger than the collection should not produce false passes."""
    with pytest.raises(ValueError, match="empty"):
        select_contiguous_shard(
            ["test::only"],
            shard_index=0,
            shard_count=2,
        )
