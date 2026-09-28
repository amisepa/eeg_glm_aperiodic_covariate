"""Null distribution of a group statistic, computed through the real pipeline.

A group statistic built from fitted spectra (a map correlation, a mean
change in exponent, a crossover) can be produced by the analysis itself:
peak power leaks into any fitted aperiodic estimate. Its null has to be
simulated with the effect of interest removed and passed through the same
per-participant code. The null also depends on the number of participants,
because noise in group means pulls correlations towards zero: in HBN the
null of a flank-band map correlation was +0.45 with 200 simulated
participants and +0.81 with 1,800. So simulate as many participants as the
real sample, and build them from the whole sample rather than the first
files on disk.
"""
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class NullDistribution:
    """draws: the statistic in each null sample (n_draws, or n_draws x ...);
    n: simulated participants per draw; n_real: real participants."""
    draws: np.ndarray
    n: int
    n_real: int

    @property
    def n_draws(self):
        return self.draws.shape[0]

    def p_value(self, observed, tail="two-sided"):
        """Monte Carlo p-value, (1 + #draws at least as extreme) / (1 + n_draws).

        tail: "greater", "less", or "two-sided" (twice the smaller one-sided
        value, at most 1). Works elementwise for array statistics.
        """
        d = self.draws
        obs = np.asarray(observed, float)
        k = d.shape[0]
        ge = (1 + np.sum(d >= obs, axis=0)) / (1 + k)
        le = (1 + np.sum(d <= obs, axis=0)) / (1 + k)
        if tail == "greater":
            return ge
        if tail == "less":
            return le
        if tail == "two-sided":
            return np.minimum(1.0, 2 * np.minimum(ge, le))
        raise ValueError("tail must be 'greater', 'less' or 'two-sided'")

    def interval(self, mass=0.95):
        """Central interval containing `mass` of the null draws."""
        q = 50 * (1 - mass)
        return tuple(np.percentile(self.draws, [q, 100 - q], axis=0))


def _one(job):
    simulate, analyse, template, ss = job
    return analyse(simulate(template, np.random.default_rng(ss)))


def matched_null(simulate, analyse, statistic, templates, n_draws=100, n=None, seed=0,
                 resample=None, map_fn=map):
    """Null distribution of statistic(outputs), outputs from `analyse` on
    simulated participants with no effect.

    simulate(template, rng): null data for one participant, in the form the
        real data take when passed to `analyse`, with the effect of interest
        removed (e.g. both conditions' spectra built from the participant's
        own mean spectrum, with no background change, plus estimation noise).
    analyse(data): the per-participant analysis, the same function that is
        run on the real data.
    statistic(outputs): the group statistic from the list of per-participant
        outputs (a number or an array).
    templates: one entry per real participant (their mean spectra, fitted
        parameters, or anything simulate needs), or the number of real
        participants if simulate needs no template.
    n: simulated participants per draw; default: as many as real ones.
        Any other value gives the null of a different study and is warned
        about.
    resample: draw templates with replacement in each draw. Default: only
        when n exceeds the number of templates. Otherwise each template is
        used once per draw, or a random subset of them when n is smaller
        (never the first n).
    seed: int or numpy SeedSequence. Every simulated participant has its own
        random stream, so the result does not depend on map_fn.
    map_fn: map-like callable, e.g. multiprocessing.Pool(8).map; simulate
        and analyse must then be picklable (defined at module level).
    Returns a NullDistribution.
    """
    if isinstance(templates, (int, np.integer)):
        templates = [None] * int(templates)
    templates = list(templates)
    n_real = len(templates)
    if n_real == 0:
        raise ValueError("no templates")
    n = n_real if n is None else int(n)
    if n != n_real:
        warnings.warn(f"null simulated with n = {n} participants per draw, real sample "
                      f"n = {n_real}: the null of a group statistic depends on n",
                      stacklevel=2)
    if resample is None:
        resample = n > n_real
    if not resample and n > n_real:
        raise ValueError("n exceeds the number of templates; set resample=True")
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    draws = []
    for ss in root.spawn(n_draws):
        kids = ss.spawn(n + 1)
        g = np.random.default_rng(kids[-1])
        if resample:
            pick = g.integers(0, n_real, n)
        elif n < n_real:
            pick = g.choice(n_real, n, replace=False)
        else:
            pick = np.arange(n)
        jobs = [(simulate, analyse, templates[j], kids[k]) for k, j in enumerate(pick)]
        draws.append(statistic(list(map_fn(_one, jobs))))
    return NullDistribution(draws=np.asarray(draws, float), n=n, n_real=n_real)


def spectral_noise(S, n_segments, rng):
    """A Welch-like estimate of the power spectrum S (linear power, any shape
    ending in frequency): S times Gamma(K, 1/K) noise per bin, with K the
    number of averaged segments. Bins are independent here; tapering and
    overlap correlate neighbouring bins in real estimates, which a
    time-domain synthesis reproduces and this does not.
    """
    S = np.asarray(S, float)
    return S * rng.gamma(n_segments, 1.0 / n_segments, S.shape)
