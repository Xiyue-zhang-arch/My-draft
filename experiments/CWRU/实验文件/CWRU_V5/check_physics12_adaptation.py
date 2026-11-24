"""Deterministic functional checks on synthetic signals, without bearing files."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.stats import beta

from physics12_adapted_reference import (
    DescriptorConfig, bic_whiten, chnr_family, extract, ses, prediction_cap,
)


def interval(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]


def run():
    rng = np.random.default_rng(20261113)
    cfg = DescriptorConfig()
    report = {"scope": "synthetic_only_no_training_no_bearing_data",
              "seed": 20261113, "checks": {}, "white_noise": {}, "responses": {}}
    # Exact artificial spectra, independent known values for CHNR.
    toy_cfg = DescriptorConfig(fs=10000)
    freq = np.arange(2501)*2.
    power = np.ones(len(freq))
    power[50] = 10000
    c, number, detail = chnr_family(freq, power, 1., 100., toy_cfg)
    assert number == 1 and len(detail) == 3 and abs(c-40) < 1e-10
    power[150] = 10000  # an isolated missing harmonic should not terminate
    c2, n2, d2 = chnr_family(freq, power, 1., 100., toy_cfg)
    assert n2 == 2 and len(d2) == 5 and abs(c2-80) < 1e-10
    # A constant baseline has no locally prominent harmonics.
    c0, n0, d0 = chnr_family(freq, np.ones(len(freq)), 1., 100., toy_cfg)
    assert c0 == 0 and n0 == 0 and len(d0) == 2
    report["checks"]["known_40dB_score_and_missing_harmonic_stop"] = True
    zero, _ = extract(np.zeros(24000), 29., cfg)
    assert np.all(zero == 0)
    x = rng.normal(size=24000)
    original, original_meta = extract(x, 29., cfg)
    scaled, scaled_meta = extract(37*x+12, 29., cfg)
    assert np.allclose(original, scaled, rtol=1e-9, atol=1e-9)
    assert original_meta["ar_order"] == scaled_meta["ar_order"]
    report["checks"]["zero_and_amplitude_offset_invariance"] = True
    # Direct scalar Yule-Walker solve independently checks Levinson residuals.
    colored = lfilter([1.], [1., -1.5, .7], rng.normal(size=24000))
    colored -= colored.mean()
    residual, order, discard = bic_whiten(colored)
    rr = np.correlate(colored[:4096], colored[:4096], mode="full")  # not used for fitting
    del rr
    p = order
    r = np.fft.irfft(np.abs(np.fft.rfft(colored, n=48000))**2, n=48000)[:p+1]/24000
    if p:
        from scipy.linalg import toeplitz
        a = np.r_[1., np.linalg.solve(toeplitz(r[:p]), -r[1:p+1])]
    else:
        a = np.array([1.])
    independent = lfilter(a, [1.], colored)[discard:]
    independent -= independent.mean()
    assert np.allclose(independent, residual, rtol=1e-8, atol=1e-8)
    report["checks"]["levinson_matches_independent_linear_solve"] = True
    report["checks"]["known_AR2_selected_order"] = order
    # Pointwise null checks use all native frequencies in a fixed diagnostic range.
    # Pooling correlated frequencies is descriptive; do not attach binomial CI.
    for fs in (12000, 24000, 48000):
        length = fs//2
        exceed, point_total = 0, 0
        detected_families, any_window = np.zeros(3, dtype=int), 0
        orders = []
        trials = 300
        for _ in range(trials):
            x = rng.normal(size=length)
            value, meta = extract(x, 29., DescriptorConfig(fs=fs))
            orders.append(meta["ar_order"])
            found = value[9:] > 0
            detected_families += found
            any_window += bool(found.any())
            noise_cfg = DescriptorConfig(fs=fs)
            residual, _, _ = bic_whiten(x-x.mean(), prediction_cap(length, 29., noise_cfg))
            f, pwr, q_mean, _ = ses(residual, fs)
            ix = (f >= 50) & (f <= 1000)
            null_mean = q_mean**2*(1-f[ix]/(fs/2))
            exceed += int((pwr[ix] > null_mean*np.log(100)).sum())
            point_total += int(ix.sum())
        report["white_noise"][str(fs)] = {
            "windows": trials, "pointwise_pooled_exceedance": exceed/point_total,
            "families_with_any_detection": (detected_families/trials).tolist(),
            "any_family_window_rate": any_window/trials,
            "any_family_window_95pct_interval": interval(any_window, trials),
            "BIC_orders": dict(Counter(orders)),
        }
    fs, length, rot = 48000, 24000, 29.
    fault_freq = 2*cfg.bsf_order*rot
    impulse_count = int(length/fs*fault_freq)
    t_kernel = np.arange(int(.02*fs))/fs
    kernel = np.exp(-t_kernel/.0015)*np.sin(2*np.pi*4500*t_kernel)
    for kind in ("white_noise", "colored_AR2", "equal_impacts",
                 "alternating_impacts", "random_impacts"):
        values, chosen = [], []
        for _ in range(40):
            noise = rng.normal(size=length)
            if kind == "white_noise":
                signal = noise
            elif kind == "colored_AR2":
                signal = lfilter([1.], [1., -1.5, .7], noise)
            else:
                train = np.zeros(length)
                if kind == "random_impacts":
                    indices = np.sort(rng.choice(length, impulse_count, replace=False))
                else:
                    phase = rng.uniform(0, 1/fault_freq)
                    times = phase+np.arange(impulse_count)/fault_freq
                    times += rng.normal(scale=.01/fault_freq, size=impulse_count)
                    indices = np.clip(np.rint(times*fs).astype(int), 0, length-1)
                amplitude = np.ones(len(indices))
                if kind == "alternating_impacts":
                    amplitude[1::2] = .3
                np.add.at(train, indices, amplitude)
                signal = fftconvolve(train, kernel)[:length]
                signal /= max(np.sqrt(np.mean(signal**2)), 1e-12)
                signal += .5*noise
            value, meta = extract(signal, rot, cfg)
            values.append(value)
            chosen.append(meta["ar_order"])
        values = np.array(values)
        report["responses"][kind] = {
            "windows": len(values), "median_features": np.median(values, axis=0).tolist(),
            "mean_features": values.mean(axis=0).tolist(),
            "rolling_detected_rate": float((values[:, 11] > 0).mean()),
            "BIC_orders": dict(Counter(chosen)),
        }
    for periodic in ("equal_impacts", "alternating_impacts"):
        assert report["responses"][periodic]["median_features"][8] > report["responses"]["random_impacts"]["median_features"][8]
    report["checks"]["rolling_CHNR_stronger_for_periodic_than_random_impacts"] = True
    report["limitations"] = [
        "Synthetic response checks do not establish classification improvement.",
        "Gaussian null thresholds are approximate after adaptive AR fitting.",
        "1% is per searched harmonic under an approximate null, not overall family/window error.",
        "Within-window BIC prewhitening is a project adaptation, not an author setting.",
    ]
    path = Path(__file__).with_name("physics12_adaptation_check_v2.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
