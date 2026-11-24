"""CPU reference for the proposed 12 descriptors; not wired into frozen training.

CHNR neighborhood construction follows Mo et al. (2020). BIC prewhitening,
the explicit SES convention, and within-neighborhood Bonferroni adjustment
are project adaptations. No target labels or fitted dataset statistics used.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, lfilter


@dataclass(frozen=True)
class DescriptorConfig:
    fs: float = 48000.0
    bpfo_order: float = 3.5848
    bpfi_order: float = 5.4152
    bsf_order: float = 2.35675
    max_harmonics: int = 6
    search_fraction: float = 0.025
    harmonic_alpha: float = 0.01
    prediction_period_fraction: float = 0.5


def prediction_cap(length, rotation_hz, cfg):
    fastest = rotation_hz*max(cfg.bpfo_order, cfg.bpfi_order, 2*cfg.bsf_order)
    return min(int(np.floor(cfg.prediction_period_fraction*cfg.fs/fastest)), length//20)


def bic_whiten(x, cap=64):
    """Biased Yule-Walker/Levinson fits, BIC includes AR(0).

    Select p in [0, min(cap, N//20)]. Same effective sample count for all
    candidates. Filter and discard the common maximum-order prefix.
    """
    n = len(x)
    max_order = min(cap, n//20)
    n_eff = n-max_order
    spectrum = np.fft.rfft(x, n=2*n)
    r = np.fft.irfft(np.abs(spectrum)**2, n=2*n)[:max_order+1]/n
    if r[0] <= np.finfo(float).tiny:
        return x[max_order:].copy(), 0, max_order
    a = np.array([1.0])
    error = float(r[0])
    best = n_eff*np.log(error)
    best_a = a.copy()
    best_order = 0
    for p in range(1, max_order+1):
        reflection = -(r[p]+np.dot(a[1:], r[p-1:0:-1]))/error
        if not np.isfinite(reflection) or abs(reflection) >= 1:
            raise ArithmeticError("Unstable Yule-Walker recursion")
        previous = a
        a = np.empty(p+1)
        a[0] = 1
        a[p] = reflection
        if p > 1:
            a[1:p] = previous[1:]+reflection*previous[-1:0:-1]
        error *= 1-reflection**2
        bic = n_eff*np.log(max(error, np.finfo(float).tiny))+p*np.log(n_eff)
        if bic < best:
            best, best_a, best_order = bic, a.copy(), p
    residual = lfilter(best_a, [1.0], x)[max_order:]
    residual -= residual.mean()
    return residual, best_order, max_order


def ses(signal, fs):
    """Rectangular observation; no resampling, no FFT interpolation.

    For real Gaussian white noise and standard doubled-positive Hilbert z,
    E[SES(alpha)] ~ mean(|z|²)² * (1-alpha/(fs/2)) away from DC/Nyquist.
    """
    z = hilbert(signal)
    q = np.abs(z)**2
    power = np.abs(np.fft.rfft(q-q.mean()))**2/(2*len(q))
    freq = np.fft.rfftfreq(len(q), 1/fs)
    return freq, power, float(q.mean()), np.abs(z)


def interval_indices(freq, lo, hi):
    a = max(1, int(np.searchsorted(freq, lo, side="left")))
    b = min(len(freq)-1, int(np.searchsorted(freq, hi, side="left")))
    if b <= a:
        raise ValueError(f"No native FFT bins in [{lo}, {hi}); use a longer window")
    return np.arange(a, b)


def chnr_family(freq, power, mean_square_envelope, f0, cfg):
    delta = cfg.search_fraction
    half = delta*f0
    width = 2*half
    m = int(np.floor((.5*f0-.5*width)/width+1e-12))
    if m < 1:
        raise ValueError("Not enough background subintervals")
    center = f0
    misses, count, score = 0, 0, 0.0
    details = []
    for h in range(1, cfg.max_harmonics+1):
        lo, hi = center-half, center+half
        if lo-m*width <= 0 or hi+m*width >= cfg.fs/2:
            raise ValueError("Incomplete target/background interval at spectral edge")
        target = interval_indices(freq, lo, hi)
        peak_idx = target[np.argmax(power[target])]
        background = []
        for j in range(1, m+1):
            for sign in (-1, 1):
                ids = interval_indices(freq, lo+sign*j*width, hi+sign*j*width)
                background.append(float(power[ids].max()))
        base = float(np.mean(background))
        amplitude = float(power[peak_idx])
        chnr = 10*np.log10(amplitude/base) if amplitude > 0 and base > 0 else -np.inf
        # Derived for our SES convention, with original chi-square form:
        # S=.5*(1-alpha/B)*mean(q)^2; threshold=S*chi2_2(1-p/m).
        null_mean = mean_square_envelope**2*(1-freq[peak_idx]/(cfg.fs/2))
        threshold = null_mean*np.log(len(target)/cfg.harmonic_alpha)
        global_pass = bool(amplitude > threshold)
        detected = bool(global_pass and chnr > 0)
        if detected:
            count += 1
            score += chnr
            misses = 0
        else:
            misses += 1
        details.append({"h": h, "center_Hz": center,
                        "peak_Hz": float(freq[peak_idx]), "target_bins": len(target),
                        "chnr_dB": float(chnr) if np.isfinite(chnr) else None,
                        "global_pass": global_pass, "detected": detected})
        # Paper's midpoint fallback for failure of the global noise test.
        estimate = float(freq[peak_idx]) if global_pass else center
        center = estimate+f0
        if misses >= 2:
            break
    return score, count, details


def extract(x, rotation_hz, cfg=DescriptorConfig()):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 256 or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite 1D signal with at least 256 samples")
    if not np.isfinite(rotation_hz) or rotation_hz <= 0 or cfg.fs <= 0:
        raise ValueError("Invalid rotation or sampling frequency")
    x = x-x.mean()
    rms = np.sqrt(np.mean(x*x))
    if rms == 0:
        return np.zeros(12), {"zero_signal": True, "ar_order": 0, "families": []}
    x = x/rms
    freq, power, _, envelope = ses(x, cfg.fs)
    raw_families = rotation_hz*np.array([cfg.bpfo_order, cfg.bpfi_order, cfg.bsf_order])
    recurrence_families = raw_families.copy()
    recurrence_families[2] *= 2
    total = power[1:-1].sum()
    concentration = []
    for f0 in raw_families:
        covered = np.zeros(len(freq), dtype=bool)
        for h in range(1, cfg.max_harmonics+1):
            half = max(2., .015*h*f0)
            if h*f0+half >= cfg.fs/2:
                raise ValueError("Insufficient spectral range for fixed R harmonics")
            covered[interval_indices(freq, h*f0-half, h*f0+half)] = True
        concentration.append(float(power[covered].sum()/total) if total > 0 else 0.)
    e = envelope-envelope.mean()
    n = len(e)
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, n=2*n))**2, n=2*n)[:n]
    recurrence = []
    for f0 in recurrence_families:
        peaks = []
        for k in range(1, 4):
            lag = k*cfg.fs/f0
            half = max(1., .02*lag)
            lo, hi = max(1, int(np.ceil(lag-half))), min(n-1, int(np.floor(lag+half)))
            if hi < lo:
                raise ValueError("Window too short for recurrence")
            peaks.append(float(ac[lo:hi+1].max()/ac[0]) if ac[0] > 0 else 0.)
        recurrence.append(float(np.mean(peaks)))
    cap = prediction_cap(len(x), rotation_hz, cfg)
    residual, order, discard = bic_whiten(x, cap)
    f_white, p_white, q_mean, _ = ses(residual, cfg.fs)
    scores, counts, details = [], [], []
    for f0 in recurrence_families:
        c, number, detail = chnr_family(f_white, p_white, q_mean, f0, cfg)
        scores.append(c)
        counts.append(number)
        details.append(detail)
    features = np.array(concentration+recurrence+scores+counts)
    if not np.all(np.isfinite(features)):
        raise ArithmeticError("Non-finite descriptor")
    return features, {"zero_signal": False, "ar_order": order,
                      "discard_samples": discard, "chnr_samples": len(residual),
                      "ar_order_cap": cap, "ar_cap_hit": order == cap,
                      "families": details}
