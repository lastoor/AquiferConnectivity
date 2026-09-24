#!/usr/bin/env python3
"""Reproduce the time-averaging effect discussed by Cowles (1960), p. 911."""

from __future__ import annotations

import argparse

import numpy as np


def sequences_reversals(levels: np.ndarray) -> tuple[int, int, float]:
    """Return sequences, reversals, and lag-one correlation of first differences."""
    changes = np.diff(levels)
    changes = changes[changes != 0]
    adjacent_products = changes[:-1] * changes[1:]
    sequences = int(np.count_nonzero(adjacent_products > 0))
    reversals = int(np.count_nonzero(adjacent_products < 0))
    correlation = float(np.corrcoef(changes[:-1], changes[1:])[0, 1])
    return sequences, reversals, correlation


def summaries(random_walk: np.ndarray) -> dict[str, tuple[int, int, float]]:
    blocks = random_walk.reshape(100, 12)
    return {
        "Every 12th point": sequences_reversals(blocks[:, -1]),
        "Midpoint of block high and low": sequences_reversals(
            (blocks.min(axis=1) + blocks.max(axis=1)) / 2
        ),
        "Mean of all 12 block values": sequences_reversals(blocks.mean(axis=1)),
    }


def print_single_run(seed: int) -> None:
    rng = np.random.default_rng(seed)
    random_walk = np.cumsum(rng.uniform(-1.0, 1.0, 1200))
    print(f"Single 1,200-step random walk (seed={seed})")
    print(f"{'Construction':36s} {'Ns':>4s} {'Nr':>4s} {'C=Ns/Nr':>10s} {'rho(1)':>9s}")
    for name, (ns, nr, correlation) in summaries(random_walk).items():
        print(f"{name:36s} {ns:4d} {nr:4d} {ns / nr:10.3f} {correlation:9.3f}")


def print_monte_carlo(repetitions: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    names = [
        "Every 12th point",
        "Midpoint of block high and low",
        "Mean of all 12 block values",
    ]
    totals = {name: np.zeros(3) for name in names}
    ratios = {name: np.empty(repetitions) for name in names}

    for repetition in range(repetitions):
        random_walk = np.cumsum(rng.uniform(-1.0, 1.0, 1200))
        for name, (ns, nr, correlation) in summaries(random_walk).items():
            totals[name] += ns, nr, correlation
            ratios[name][repetition] = ns / nr

    print(f"\nMonte Carlo averages ({repetitions:,} random walks, seed={seed})")
    print(
        f"{'Construction':36s} {'E[Ns]':>8s} {'E[Nr]':>8s} "
        f"{'E[C]':>8s} {'pooled C':>9s} {'E[rho]':>9s}"
    )
    for name in names:
        mean_ns, mean_nr, mean_correlation = totals[name] / repetitions
        mean_ratio = float(ratios[name].mean())
        pooled_ratio = mean_ns / mean_nr
        print(
            f"{name:36s} {mean_ns:8.3f} {mean_nr:8.3f} "
            f"{mean_ratio:8.3f} {pooled_ratio:9.3f} {mean_correlation:9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--single-seed", type=int, default=1)
    parser.add_argument("--monte-carlo-seed", type=int, default=520)
    args = parser.parse_args()

    print_single_run(args.single_seed)
    print_monte_carlo(args.repetitions, args.monte_carlo_seed)


if __name__ == "__main__":
    main()
