from __future__ import annotations

import numpy as np


def compute_pdf(distances: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pdf, edges = np.histogram(distances, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf


def average_histograms(list_of_distances: list[np.ndarray], bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pdfs = []
    for d in list_of_distances:
        _, pdf = compute_pdf(np.asarray(d), bins)
        pdfs.append(pdf)
    all_pdfs = np.vstack(pdfs)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, np.mean(all_pdfs, axis=0), np.std(all_pdfs, axis=0)
