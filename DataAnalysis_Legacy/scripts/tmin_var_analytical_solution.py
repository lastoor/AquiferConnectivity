#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from visualization.stats.analytical_tmin import compute_analytical_tmin_curve


def main() -> int:
    # Example values matching current default visualization settings.
    field_size = 200
    length = field_size - 2 * (field_size // 20)
    velocity = 1.0 * 0.1  # k_g * dh
    i_y = 0.05 * field_size
    percentile = 0.01

    sigma2_list = np.arange(0.5, 5.1, 0.5)
    tmin_dimless, warnings = compute_analytical_tmin_curve(
        sigma2_list,
        length=length,
        velocity=velocity,
        i_y=i_y,
        percentile=percentile,
    )

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"- {msg}")

    plt.figure(figsize=(5, 4))
    plt.plot(sigma2_list, tmin_dimless, marker="o", linestyle="--", color="black")
    plt.xlabel(r"$\sigma_Y^2$")
    plt.ylabel(r"$t_{\mathrm{min}} U / I_Y$")
    plt.title("Analytical $t_{\\mathrm{min}}$ vs variance")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
