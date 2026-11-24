"""39 维物理描述符和模型接口的合成功能测试。"""

import math
import unittest

import torch  # Windows 环境须先加载 PyTorch。

from experiments.CWRU.实验文件.CWRU_V5.v5_model import CompactEvidence, SuiteModel, engine, v3_engine


def modulated_windows(carriers, modulation_hz, harmonics=(1,)):
    time = torch.arange(24000, dtype=torch.float64) / 48000
    envelope = 1 + sum(
        0.55 / h * torch.cos(2 * math.pi * h * modulation_hz * time)
        for h in harmonics
    )
    waves = [envelope * torch.sin(2 * math.pi * carrier * time) for carrier in carriers]
    return torch.stack(waves).unsqueeze(1).float()


class CompactPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.evidence = CompactEvidence(engine.R.U.CFG)

    def test_known_modulation_and_batch_order(self):
        # 80 Hz 正好是裁剪后包络谱的第 36 个非零频点。
        windows = modulated_windows([1500, 4500, 9000, 16000], 80, (1, 2))
        rotation = torch.full((4,), 80 / 3.5848)
        values, _ = self.evidence.extract(windows, rotation)
        self.assertEqual(values.shape, (4, 39))
        self.assertTrue(torch.isfinite(values).all())
        per_band = values[:, :36].reshape(4, 4, 9)
        envelope, centered, spectrum = self.evidence._unfloored_envelope_spectrum(windows)
        length = envelope.shape[-1]
        old_shape = self.evidence._shape_features(envelope, centered, spectrum.clamp_min(1e-16), length)
        torch.testing.assert_close(per_band[..., [6, 7]], old_shape[..., [0, 4]])
        for band in range(4):
            with self.subTest(band=band):
                dominant = per_band[band, band]
                self.assertGreater(float(dominant[2]), 0.6)
                self.assertGreater(float(dominant[2]), float(dominant[3]))
                self.assertAlmostEqual(float(dominant[8]), 80 / 1500, places=5)

    def test_batch_matches_single_for_nonempty_bands(self):
        waves = modulated_windows([1500, 4500, 9000, 16000], 80, (1, 2))
        # 各带都有调制功率，微扰打破理想周期波形中反复出现的等高事件。
        noise = torch.randn(waves.shape, generator=torch.Generator().manual_seed(431))
        windows = waves + 0.3 * waves.sum(dim=0) + 0.002 * noise
        rotation = torch.full((4,), 80 / 3.5848)
        values, _ = self.evidence.extract(windows, rotation)
        for i in range(4):
            with self.subTest(i=i):
                single, _ = self.evidence.extract(windows[i:i+1], rotation[i:i+1])
                self.assertEqual(single.shape, (1, 39))
                torch.testing.assert_close(single[0], values[i], atol=2e-5, rtol=2e-4)

    def test_zero_signal_and_out_of_range_harmonics(self):
        empty, _ = self.evidence.extract(torch.zeros(1, 1, 24000), torch.tensor([29.5]))
        self.assertEqual(empty.shape, (1, 39))
        per_band = empty[:, :36].reshape(1, 4, 9)
        # F5 沿用旧定义。其它量均有明确的零值约定。
        torch.testing.assert_close(per_band[..., 7], torch.ones(1, 4))
        self.assertEqual(torch.count_nonzero(per_band[..., [0, 1, 2, 3, 4, 5, 6, 8]]), 0)
        self.assertEqual(torch.count_nonzero(empty[:, 36:]), 0)
        wave = modulated_windows([4500], 80)
        for rotation in (0.001, 10000.):
            with self.subTest(rotation=rotation):
                values, _ = self.evidence.extract(wave, torch.tensor([rotation]))
                self.assertTrue(torch.isfinite(values).all())
                self.assertEqual(torch.count_nonzero(values[:, :36].reshape(1, 4, 9)[..., :6]), 0)

    def test_rejects_invalid_inputs(self):
        wave = torch.zeros(1, 1, 24000)
        for rotation in (0., -1., float('nan'), float('inf')):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                self.evidence.extract(wave, torch.tensor([rotation]))
        wave[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.evidence.extract(wave, torch.tensor([29.5]))

    def test_concentration_uses_union_and_ignores_dc(self):
        power = torch.zeros(1, 4, 676)
        power[..., [0, 1, 2, 3, 10, 675]] = torch.tensor([99., 1., 3., 2., 2., 4.])
        frequencies = torch.tensor([[1., 1500., 1501., .001]])
        concentration = self.evidence._spectral_concentration(power, frequencies)
        expected = torch.tensor([.5, 1 / 3, 0., 0.]).expand_as(concentration)
        torch.testing.assert_close(concentration, expected)
        self.assertEqual(self.evidence.orders.numel(), 4)
        torch.testing.assert_close(
            self.evidence.orders * 30., torch.tensor([11.9484, 70.7025, 107.544, 162.456])
        )

    def test_periodicity_aggregation_and_legacy_definitions(self):
        wave = modulated_windows([4500], 100, (1, 2))
        rotation = torch.tensor([30.])
        envelope, centered, _ = self.evidence._unfloored_envelope_spectrum(wave)
        legacy = v3_engine.R.Evidence(v3_engine.R.U.CFG)
        candidates = rotation[:, None, None] * legacy.orders[None, :, None] * legacy.multipliers
        frequencies = candidates[:, [2, 4, 5], 0]
        old_u, _ = legacy._recurrence_features(centered, centered.shape[-1], candidates)
        old_v, _, _ = legacy._impact_features(envelope, envelope.shape[-1], .45, candidates)
        torch.testing.assert_close(self.evidence._recurrence(centered, frequencies), old_u[..., [2, 4, 5], 0])
        torch.testing.assert_close(self.evidence._phase(envelope, frequencies), old_v[..., [2, 4, 5], 0])
        r = torch.tensor([[.9, .1, .3, .0], [.8, .0, .0, .0]])
        u = torch.tensor([[.2, .6, .7], [1., 1., 1.]])
        v = torch.tensor([[.4, .8, .9], [1., 1., 1.]])
        u_agg, v_agg = self.evidence._aggregate_periodicity(r, u, v)
        torch.testing.assert_close(u_agg, torch.tensor([.5, 0.]))
        torch.testing.assert_close(v_agg, torch.tensor([.7, 0.]))
        self.assertEqual(torch.count_nonzero(self.evidence._phase(torch.zeros_like(envelope), frequencies)), 0)

    def test_rms_ratios_use_cropped_envelopes(self):
        wave = modulated_windows([1500], 80) + 0.5 * modulated_windows([16000], 80)
        values, _ = self.evidence.extract(wave, torch.tensor([29.5]))
        envelope = self.evidence._envelope_spectrum(wave)[0]
        rms = envelope.square().mean(-1).sqrt()
        expected = ((rms[:, :3] + 1e-8) / (rms[:, 3:] + 1e-8)).log()
        torch.testing.assert_close(values[:, 36:], expected)

    def test_model_interface_and_train_only_standardization(self):
        torch.manual_seed(1706)
        model = SuiteModel('data_aux_ce', dict(aux_weight=.3))
        torch.manual_seed(1706)
        old = v3_engine.SuiteModel('data_aux_ce', dict(aux_weight=.3))
        self.assertEqual(old.network.physics[0].in_features, 212)
        self.assertEqual(model.network.physics[0].in_features, 39)
        self.assertEqual(model.network.physics(torch.zeros(2, 39)).shape, (2, 32))
        for key, value in old.network.state_dict().items():
            if key != 'physics.0.weight' and key != 'physics.0.bias':
                torch.testing.assert_close(model.network.state_dict()[key], value, atol=0, rtol=0)
        x = modulated_windows([1500, 4500, 9000, 16000], 80).numpy()
        rpm = torch.full((4,), 1750.).numpy()
        info = engine._fit_physics(model, x[:3], rpm[:3], 'clean_only', 'cpu', model.spec)
        features, _ = model.evidence.extract(torch.from_numpy(x[:3]), torch.from_numpy(rpm[:3]) / 60)
        torch.testing.assert_close(model.evidence.center, features.median(dim=0).values)
        center = model.evidence.center.clone()
        logits = model(torch.from_numpy(x[3:]), torch.from_numpy(rpm[3:]))
        self.assertEqual(logits.shape, (1, 4))
        self.assertEqual(model.aux_logits.shape, (1, 4))
        self.assertEqual(info['fit_unique_training_windows'], 3)
        torch.testing.assert_close(model.evidence.center, center)
        loss, _, _ = engine.loss_for(model, logits, torch.tensor([2]), model.spec)
        loss.backward()
        for module in (model.network.encoder, model.network.physics, model.network.gate, model.network.classifier):
            self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))
        self.assertEqual(list(model.evidence.parameters()), [])

    def test_baseline_data_path_is_identical(self):
        torch.manual_seed(1707)
        old = v3_engine.SuiteModel('baseline_data').eval()
        torch.manual_seed(1707)
        new = SuiteModel('baseline_data').eval()
        wave = modulated_windows([4500], 80)
        rpm = torch.tensor([1750.])
        torch.testing.assert_close(old(wave, rpm), new(wave, rpm), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
