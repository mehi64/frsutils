# Glossary

This page defines the main execution and fuzzy-rough terms used in FRsutils.
Definitions are intentionally short so they can be used consistently across the
README, user docs, benchmark reports, and the JOSS paper.

## Dense execution

Execution mode that builds or uses the full pairwise similarity matrix in
memory. In FRsutils, the standard dense path uses NumPy.

## Blockwise execution

Execution mode that computes exact fuzzy-rough results by processing similarity
blocks instead of materializing the full similarity matrix. Blockwise execution
reduces pairwise-computation memory use; it does not currently provide disk-backed
or streaming dataset loading, so the input feature matrix `X` is expected to fit
in memory.

## Backend

The array/computation library used internally for supported operations. The
stable default backend is NumPy. CuPy is optional and currently limited to
selected backend-aware paths.

## Public output

The result returned by the public API. FRsutils returns public result arrays as
NumPy arrays, even when CuPy is used internally.

## GPU-backed similarity blocks

Similarity blocks computed with the CuPy backend during supported blockwise
execution.

## GPU-resident approximation accumulators

Model-specific intermediate approximation arrays that remain on the GPU during
supported blockwise computation before final conversion to NumPy public output.

## Engine

The high-level execution mode selected by the public API, usually `"dense"` or
`"blockwise"`.

## Similarity matrix

A pairwise matrix where entry `(i, j)` stores the fuzzy similarity between sample
`i` and sample `j`.

## Lower approximation

The fuzzy-rough degree to which each sample certainly belongs to a target class
or concept.

## Upper approximation

The fuzzy-rough degree to which each sample possibly belongs to a target class
or concept.

## Signed boundary

The un-clipped difference between upper and lower approximation degrees:
`upper - lower`. It can be negative when a model configuration does not
guarantee that the upper approximation is at least the lower approximation.
The older name `boundary region` remains available in the API for backward
compatibility.

## Positive region

A fuzzy-rough score summarizing how confidently samples can be associated with
their decision class.
