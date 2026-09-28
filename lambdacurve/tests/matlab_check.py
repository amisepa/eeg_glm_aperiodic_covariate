"""Compare code/lib/oof_lambda_curve.m with lambda_curve on the same inputs
and the same bootstrap draws (exported from numpy). Not part of the pytest
run: MATLAB takes minutes to start. Needs MATLAB on the path, or its
executable in the MATLAB environment variable.

    python tests/matlab_check.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import lambdacurve as L  # noqa: E402

LIB = HERE.parents[1] / "code" / "lib"
NBOOT = 1000


def draws(n, nboot=NBOOT, seed=0):
    """The draws lambda_curve takes from default_rng(seed), in its order."""
    rng = np.random.default_rng(seed)
    idx, w = np.empty((n, nboot)), np.empty((n, nboot))
    for k in range(nboot):
        idx[:, k] = rng.integers(0, n, n) + 1
        w[:, k] = rng.dirichlet(np.ones(n))
    return idx, w


def main():
    with tempfile.TemporaryDirectory() as t:
        check(Path(t))


def check(tmp):
    rng = np.random.default_rng(12)
    n = 200
    x = rng.uniform(6, 18, n)
    cov = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    lb = 1 - 0.10 * x + rng.normal(0, 0.4, n)
    la = 0.5 - 0.05 * x + rng.normal(0, 0.5, n)
    cases = {"between": (la, lb, x, cov),
             "within": (rng.normal(0.8, 0.5, 40), rng.normal(0.3, 0.2, 40), None, None)}
    py = {}
    for name, (a, b, xx, cc) in cases.items():
        py[name] = L.lambda_curve(a, b, x=xx, covariates=cc, nboot=NBOOT)
        idx, w = draws(py[name].n)
        empty = np.zeros((0, 0))
        savemat(tmp / f"{name}_in.mat", dict(la=a, lb=b, x=empty if xx is None else xx,
                                             cov=empty if cc is None else cc, idx=idx, w=w))
    cmd = (f"addpath('{LIB.as_posix()}'); for c = {{'between','within'}}, "
           f"d = load(fullfile('{tmp.as_posix()}', [c{{1}} '_in.mat'])); "
           f"o = oof_lambda_curve(d.la(:), d.lb(:), d.x(:), d.cov, "
           f"struct('draws', struct('idx', d.idx, 'w', d.w))); "
           f"save(fullfile('{tmp.as_posix()}', [c{{1}} '_out.mat']), '-struct', 'o'); end")
    subprocess.run([os.environ.get("MATLAB", "matlab"), "-batch", cmd], check=True)
    worst = 0.0
    for name, r in py.items():
        m = loadmat(tmp / f"{name}_out.mat", squeeze_me=True)
        pairs = [(r.s_a, m["s_a"]), (r.s_b, m["s_b"]), (r.lam_star, m["lam_star"]),
                 (r.ci, m["ci"]), (r.hdi, m["hdi"]), (r.p_cross_in_01, m["p_cross_in_01"]),
                 (r.curve, m["curve"]), (r.lam0, m["lam0"]), (r.lam1, m["lam1"])]
        d = max(float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float)))) for p, q in pairs)
        worst = max(worst, d)
        print(f"{name}: max |MATLAB - Python| {d:.1e}; verdict {m['verdict']} / {L.verdict(r)}")
        assert str(m["verdict"]) == L.verdict(r)
    assert worst < 1e-9, worst


if __name__ == "__main__":
    main()
