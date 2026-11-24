"""Four fixed, locally normalized upper-sideband features; no envelope or labels."""
import math
import torch

LENGTH = 2560
SPR = 640
POLE_PAIRS = 4
ORDERS = (3.054290718038529, 4.945709281961471)
CENTERS = tuple(POLE_PAIRS + h * r for r in ORDERS for h in (1, 2))
NAMES = ('L_outer_h1', 'L_outer_h2', 'L_inner_h1', 'L_inner_h2')
CORE_HALF_WIDTH = .5
GUARD_HALF_WIDTH = .75
BACKGROUND_RADIUS = 4.
ELECTRICAL_GUARD = .5
TRANSFER_POWER_FLOOR = .25
RELATIVE_POWER_FLOOR = 1e-6


def specification():
    return dict(version='current_local_sideband_v2', dimension=4, names=NAMES,
                centers=CENTERS, mechanical_orders=ORDERS, pole_pairs=POLE_PAIRS,
                length=LENGTH, samples_per_revolution=SPR, order_spacing=SPR / LENGTH,
                input='PU_0 half-electrical-period-summed angle-domain phase current',
                core_half_width=CORE_HALF_WIDTH, guard_half_width=GUARD_HALF_WIDTH,
                background_radius=BACKGROUND_RADIUS, electrical_guard=ELECTRICAL_GUARD,
                spectrum_interval=[1., 20.], background_statistic='median',
                transfer_power_floor=TRANSFER_POWER_FLOOR,
                relative_power_floor=RELATIVE_POWER_FLOOR,
                formula='log1p(mean(corrected_power[core]) / max(median(corrected_power[local_background]), relative_floor))',
                labels_used=False, fit_parameters=False, window_or_band_search=False)


def transfer_power(q):
    # x(theta) + x(theta - 1/(2*p)) has this ideal steady-state power gain.
    return 4 * torch.cos(math.pi * q / (2 * POLE_PAIRS)).square()


def masks(device='cpu', dtype=torch.float64):
    q = torch.fft.rfftfreq(LENGTH, 1 / SPR, device=device, dtype=dtype)
    centers = torch.tensor(CENTERS, device=device, dtype=dtype)
    distance = (q[None] - centers[:, None]).abs()
    electrical = torch.arange(POLE_PAIRS, 21, POLE_PAIRS, device=device, dtype=dtype)
    electric_mask = (q[:, None] - electrical).abs().le(ELECTRICAL_GUARD).any(-1)
    eligible = (q >= 1) & (q <= 20) & ~electric_mask
    core = distance.le(CORE_HALF_WIDTH) & eligible
    all_guards = distance.le(GUARD_HALF_WIDTH).any(0)
    background = distance.gt(GUARD_HALF_WIDTH) & distance.le(BACKGROUND_RADIUS)
    background &= eligible & ~all_guards
    if not bool((core.sum(-1) >= 3).all()) or not bool((background.sum(-1) >= 5).all()):
        raise RuntimeError('Insufficient independent frequency bins in fixed masks')
    if bool((core.any(0) & background.any(0)).any()):
        raise RuntimeError('Signal/background overlap')
    return q, core, background, eligible


@torch.no_grad()
def describe_current(current):
    if (current.ndim != 2 or current.shape[-1] != LENGTH or not current.is_floating_point()
            or not bool(torch.isfinite(current).all())):
        raise ValueError('Expected finite floating [batch,2560] preprocessed current')
    output_dtype = current.dtype
    original = current.double()
    x = original - original.mean(-1, keepdim=True)
    rms = x.square().mean(-1, keepdim=True).sqrt()
    active = rms > 32 * torch.finfo(x.dtype).eps * original.abs().mean(-1, keepdim=True)
    x = torch.where(active, x / rms.clamp_min(torch.finfo(x.dtype).tiny), 0.)
    window = torch.hann_window(LENGTH, periodic=True, device=x.device, dtype=x.dtype)
    x = x - (x * window).sum(-1, keepdim=True) / window.sum()
    power = torch.fft.rfft(x * window).abs().square()
    q, core, background, eligible = masks(x.device, x.dtype)
    corrected = power / transfer_power(q).clamp_min(TRANSFER_POWER_FLOOR)
    floor = corrected[:, eligible].mean(-1) * RELATIVE_POWER_FLOOR
    floor = floor.clamp_min(torch.finfo(x.dtype).tiny)
    values = []
    for foreground, neighbors in zip(core, background):
        signal = corrected[:, foreground].mean(-1)
        # quantile(0.5) also averages the central pair for an even number of bins.
        local_background = torch.quantile(corrected[:, neighbors], .5, dim=-1)
        ratio = signal / torch.maximum(local_background, floor)
        values.append(torch.log1p(ratio))
    result = torch.stack(values, -1)
    result = torch.where(active, result, 0.)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite sideband descriptors')
    return result.to(output_dtype)


def frequency_manifest():
    q, core, background, _ = masks()
    return [dict(name=name, center=mu, core_orders=q[c].tolist(),
                 background_orders=q[b].tolist(), core_bins=int(c.sum()),
                 background_bins=int(b.sum()))
            for name, mu, c, b in zip(NAMES, CENTERS, core, background)]
