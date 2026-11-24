"""Synthetic-only audit of SES scaling; never opens bearing data or trains models.

This is a diagnostic for ambiguous transform conventions, not an implementation
or validation of Mo et al.'s complete detector. It deliberately compares several
FFT conventions against the same transcription of Eq. (27).
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.stats import chi2


def run():
    rng = np.random.default_rng(20261112)
    output = {
        "scope": "synthetic_gaussian_noise_only",
        "paper_detector_validated": False,
        "note": "Neither a threshold calibration nor permission to alter the paper formula.",
        "trials_per_length": 1000,
        "cases": [],
    }
    for length in (6000, 12000, 24000):
        fs = 2 * length  # fixed half-second duration
        bandwidth = fs / 2
        bins = np.array([25, 50, 100, 200, 400])
        alpha = bins * fs / length
        count = np.zeros((4, len(bins)), dtype=int)
        ratio_sum = np.zeros((4, len(bins)))
        for start in range(0, 1000, 25):
            x = rng.normal(size=(min(25, 1000-start), length))
            x -= x.mean(axis=1, keepdims=True)
            z = hilbert(x, axis=1)
            # Standard real-signal Hilbert convention. Endpoint effects retained.
            q = np.abs(z) ** 2
            fft_power = np.abs(np.fft.rfft(q, axis=1)[:, bins]) ** 2
            energy = q.mean(axis=1, keepdims=True)
            scale = (bandwidth-alpha)[None, :] / (2*bandwidth) * energy**2
            threshold = scale * chi2.ppf(0.99, df=2)
            for i, denominator in enumerate((1, length, length**2, 2*length)):
                spectrum = fft_power / denominator
                count[i] += (spectrum > threshold).sum(axis=0)
                ratio_sum[i] += (spectrum / scale).sum(axis=0)
        output["cases"].append({
            "N": length,
            "fs": fs,
            "duration_seconds": length/fs,
            "frequencies_Hz": alpha.tolist(),
            "conventions": [
                {"power_divisor": name,
                 "pointwise_exceedance": (count[i]/1000).tolist(),
                 "mean_SES_over_S": (ratio_sum[i]/1000).tolist()}
                for i, name in enumerate(("1", "N", "N_squared", "2N_diagnostic_only"))
            ],
        })
    # Native FFT bin geometry for the present bearing's lowest/highest speeds.
    geometry = []
    for rpm in (1720, 1797):
        for family, order in (("BPFO", 3.5848), ("BPFI", 5.4152),
                              ("2BSF", 2*2.35675)):
            f0 = rpm/60*order
            half = .025*f0
            width = 2*half
            M = int(np.floor((.5*f0-.5*width)/width))
            lo, hi = (f0-half)/2, (f0+half)/2
            geometry.append({"rpm": rpm, "family": family, "f0_Hz": f0,
                             "half_width_Hz": half, "M_per_side": M,
                             "native_bins_in_first_target": int(np.floor(hi)-np.ceil(lo)+1)})
    output["native_2Hz_bin_geometry"] = geometry
    output["decision"] = "Do not lock the global detector until SES and Eq27 scaling are reconciled."
    path = Path(__file__).resolve().parent / "physics12_synthetic_conventions_v2.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
