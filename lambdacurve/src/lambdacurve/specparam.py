"""ln a and ln b from a table of specparam (FOOOF) parameters."""
import numpy as np


def from_specparam(offset, exponent, cf, pw, knee=None):
    """Periodic and aperiodic power at the peak, from specparam's output.

    offset, exponent, knee: aperiodic parameters, as returned by
    get_params('aperiodic') (knee is specparam's knee parameter, not the knee
    frequency; None for the fixed mode). cf, pw: centre frequency and power
    of the peak, as returned by get_params('peak'). Scalars or arrays, one
    entry per spectrum.

    specparam models log10 power, so pw is a log10 RATIO of the spectrum to
    the background at cf, not a difference in power:

        b = 10^offset / (knee + cf^exponent)       background at the peak
        a = (10^pw - 1) * b                        power above the background

    Returns (ln a, ln b), ready for lambda_curve. A spectrum with no detected
    peak (cf or pw nan) gives nan, and so does pw <= 0. lambda_curve refuses
    those rows unless dropna=True; report how many there are, because
    whether specparam finds a peak depends on the background.
    """
    offset, exponent, cf, pw = (np.asarray(v, float) for v in (offset, exponent, cf, pw))
    k = 0.0 if knee is None else np.asarray(knee, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = offset * np.log(10.0) - np.log(k + cf ** exponent)
        la = lb + np.log(np.where(pw > 0, 10.0 ** pw - 1.0, np.nan))
    return la, lb
