"""Does a periodic-power result depend on the assumed periodic/aperiodic coupling?

    fit_aperiodic, ap_eval, ap_band_power, fit_mask   aperiodic models (Whittle)
    band_power, peak_frequency                        a, b and total in a band
    lambda_curve, verdict                             effect as a function of lambda, lambda*
    coupling_two_conditions, coupling_levels          log-free estimates of lambda
    matched_null, spectral_noise                      null of a group statistic through
                                                      the real per-participant code
"""
from .aperiodic import MODELS, ap_band_power, ap_eval, fit_aperiodic, fit_mask
from .bandpower import band_power, peak_frequency
from .coupling import Coupling, coupling_levels, coupling_two_conditions
from .curve import VERDICTS, LambdaCurve, hdi, lambda_curve, verdict
from .null import NullDistribution, matched_null, spectral_noise

__version__ = "0.1.0"

__all__ = ["MODELS", "ap_band_power", "ap_eval", "fit_aperiodic", "fit_mask",
           "band_power", "peak_frequency", "Coupling", "coupling_levels",
           "coupling_two_conditions", "VERDICTS", "LambdaCurve", "hdi", "lambda_curve",
           "verdict", "NullDistribution", "matched_null", "spectral_noise"]
