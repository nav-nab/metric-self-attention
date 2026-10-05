import json
import math

import numpy as np
import pytest
import torch

from msa.calibrate import (MIN_KEYS, _collect, N_CALIBRATION_BATCHES, calibrate_model, calibration_combos,
                           run_calibration)
from msa.data import CALIBRATION_SEED, calibration_batches, load_corpus
from msa.model import build_model
from msa.scores import EPS_RATIO, INF, parse_arm

_cache = {}


def calibrated(arm, D, n_batches=8):
    key = (arm, D, n_batches)
    if key not in _cache:
        batches = calibration_batches(load_corpus(), n_batches)
        model = build_model(arm, D, CALIBRATION_SEED)
        _cache[key] = (model, batches, calibrate_model(model, batches))
    return _cache[key]


def pooled_scaled_sd(model, batches, layer):
    """Independent check: loop over queries with numpy and pool the variances."""
    attn = model.blocks[layer].attn
    attn.capture = True
    var_sum, count = 0.0, 0
    with torch.no_grad():
        for x, _ in batches:
            model.hidden(x, upto=layer + 1)
            scores = attn.captured["scores"].numpy().astype(np.float64)    # masked with -inf
            B, H, T, _ = scores.shape
            for b in range(B):
                for h in range(H):
                    for i in range(T):
                        s = scores[b, h, i]
                        s = s[np.isfinite(s)]
                        if len(s) >= MIN_KEYS:
                            var_sum += s.var()
                            count += 1
    attn.capture = False
    return math.sqrt(var_sum / count), count


@pytest.mark.parametrize("arm,D", [("dot_tied", 4), ("dot_untied", 16), ("lp_1", 2), ("lp_2", 8),
                                   ("lp_3", 12), ("lp_inf", 4)])
def test_scaled_scores_have_unit_spread(arm, D):
    batches = calibration_batches(load_corpus(), 2)
    model = build_model(arm, D, CALIBRATION_SEED)
    calibrate_model(model, batches)
    for layer in range(2):
        sd, _ = pooled_scaled_sd(model, batches, layer)
        assert sd == pytest.approx(1.0, abs=1e-4)


def test_unmasked_arm_is_calibrated_over_all_keys_including_the_position_itself():
    batches = calibration_batches(load_corpus(), 2)
    B, T, H = 64, 64, 4
    masked = calibrate_model(build_model("dot_untied", 16, CALIBRATION_SEED), batches)
    unmasked = calibrate_model(build_model("dot_untied_unmasked", 16, CALIBRATION_SEED), batches)
    # masked: queries i with i >= 8 allowed keys; unmasked: i + 1 >= 8
    assert masked[0]["n_queries"] == 2 * B * H * (T - 8)
    assert unmasked[0]["n_queries"] == 2 * B * H * (T - 7)
    model = build_model("dot_untied_unmasked", 16, CALIBRATION_SEED)
    calibrate_model(model, batches)
    sd, count = pooled_scaled_sd(model, batches, 0)
    assert count == unmasked[0]["n_queries"] and sd == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("D", [2, 4, 8, 12, 16])
def test_tied_dot_constant_follows_the_variance_of_z(D):
    # design section 6: c should be near 1 / (sigma_z^2 sqrt(D)); sigma_z^2 is measured
    _, _, layers = calibrated("dot_tied", D)
    for x in layers:
        ratio = x["c"] * x["z_var"] * math.sqrt(D)
        assert 0.75 < ratio < 1.25, (D, x["layer"], ratio)


@pytest.mark.parametrize("arm", ["dot_tied", "lp_1", "lp_1.5", "lp_4", "lp_8", "lp_inf", "dot_untied"])
def test_constants_are_stable_to_one_percent(arm):
    _, _, layers = calibrated(arm, 8, n_batches=N_CALIBRATION_BATCHES)
    for x in layers:
        a, b = x["sigma_halves"]
        assert abs(a / b - 1) < 0.01, (arm, x["layer"], a, b)


@pytest.mark.parametrize("arm", ["dot_tied", "lp_1", "lp_1.5", "lp_2", "lp_4", "lp_8", "lp_inf"])
def test_epsilon_is_relative_and_only_for_finite_p_above_one(arm):
    _, _, layers = calibrated(arm, 4)
    spec = parse_arm(arm)
    for x in layers:
        if spec.kind == "distance" and spec.p not in (1.0, INF):
            assert x["s_typ"] > 0 and x["eps"] == pytest.approx(EPS_RATIO * x["s_typ"])
        else:
            assert x["eps"] == 0.0 and x["s_typ"] is None


def test_calibration_is_repeatable_and_resets_old_constants():
    model, batches, first = calibrated("lp_2", 4)
    model.blocks[0].attn.c.fill_(7.0)
    model.blocks[1].attn.eps.fill_(3.0)
    second = calibrate_model(model, batches)
    for a, b in zip(first, second):
        assert a["c"] == pytest.approx(b["c"], rel=1e-6) and a["eps"] == pytest.approx(b["eps"], rel=1e-6)


def test_each_layer_is_measured_on_inputs_from_the_calibrated_layers_below():
    batches = calibration_batches(load_corpus(), 2)
    model = build_model("dot_tied", 16, CALIBRATION_SEED)
    layers = calibrate_model(model, batches)
    # In the finished model layer 0 carries its calibrated constant. Layer 1 sees its inputs,
    # and the raw spread measured there must equal the stored value.
    attn = model.blocks[1].attn
    attn.capture = True
    var_sum, count = 0.0, 0
    with torch.no_grad():
        for x, _ in batches:
            model.hidden(x, upto=2)
            v, n, _ = _collect(attn, attn.captured["raw"], MIN_KEYS)
            var_sum, count = var_sum + v, count + n
    assert math.sqrt(var_sum / count) == pytest.approx(layers[1]["sigma"], rel=1e-6)
    # With layer 0 left at c = 1 the layer-1 inputs differ, so the measurement would differ
    model.blocks[0].attn.c.fill_(1.0)
    var_sum, count = 0.0, 0
    with torch.no_grad():
        for x, _ in batches:
            model.hidden(x, upto=2)
            v, n, _ = _collect(attn, attn.captured["raw"], MIN_KEYS)
            var_sum, count = var_sum + v, count + n
    assert math.sqrt(var_sum / count) != pytest.approx(layers[1]["sigma"], rel=1e-6)


def test_stage_writes_files_and_skips_finished_cells(tmp_path):
    combos = [("dot_tied", 2), ("lp_1.5", 4)]
    messages = []
    run_calibration(tmp_path, n_batches=2, combos=combos, log=messages.append)
    path = tmp_path / "calibration" / "lp_1.5_D4.json"
    data = json.loads(path.read_text())
    assert data["seed"] == CALIBRATION_SEED and len(data["layers"]) == 2
    assert (tmp_path / "calibration" / "calibration_summary.csv").exists()
    stamp = path.stat().st_mtime_ns
    messages.clear()
    run_calibration(tmp_path, n_batches=2, combos=combos, log=messages.append)
    assert path.stat().st_mtime_ns == stamp and all("skipped" in m for m in messages[:2])
    assert not list((tmp_path / "calibration").glob("*.tmp"))


def test_combinations():
    combos = calibration_combos()
    assert len(combos) == 46 and ("dot_untied_unmasked", 16) in combos
    assert not any(arm == "uniform" for arm, _ in combos)
