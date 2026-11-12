"""Checks that need no downloaded data: configs, models and the paper checkpoints."""
from pathlib import Path

import numpy as np
import pytest

from crbreak.checkpoint import load_checkpoint
from crbreak.config.analysis import REPO_ROOT, load_config
from crbreak.fitting.models import PowerLaw, SmoothBrokenPowerLaw

CONFIGS = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
CHECKPOINTS = sorted((REPO_ROOT / "results" / "paper1").glob("*/checkpoint.pkl.gz"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_config_loads(path: Path):
    cfg = load_config(path)
    assert set(cfg.data.kinds) <= {"PASS8", "IEM", "DRAGON"}


def test_sbpl_without_break_is_a_power_law():
    model = load_config(REPO_ROOT / "configs" / "baseline.yaml").model
    E = np.geomspace(3.0, 700.0, 30)
    pl = PowerLaw.from_config(model)(E, 1.0, 2.5)
    # dgamma = 0 leaves K (E/E0)^-gamma1 times the constant 2^s.
    sbpl = SmoothBrokenPowerLaw.from_config(model)(E, 1.0, 2.5, 0.0, 26.0)
    np.testing.assert_allclose(sbpl / pl, 2.0 ** model.sbpl_s_fixed, rtol=1e-12)


@pytest.mark.parametrize("path", CHECKPOINTS, ids=lambda p: p.parent.name)
def test_paper_checkpoint_loads(path: Path):
    state = load_checkpoint(path)
    run = state["runs"][0]
    assert len(run.fitting.window_fits) == len(state["glon_edges"]) - 1
    assert np.isfinite(run.g.chi2)


def test_baseline_checkpoint_global_fit():
    run = load_checkpoint(REPO_ROOT / "results" / "paper1" / "baseline")["runs"][0]
    ebreak, dgamma = (run.g_prof.shared_params[k][0] for k in ("Ebreak", "dgamma"))
    assert ebreak == pytest.approx(23.91, abs=0.01)
    assert dgamma == pytest.approx(0.216, abs=0.001)
