#!/usr/bin/env python3
# Supplementary diagnostic for Reviewer 1, Comment 1.
# Standalone: does not modify AstroTuner or its original Experiment A outputs.
# Uses exact sinusoidal inputs, sampling at 0.1 m, and illustrative Gaussian
# vertical-response FWHMs. These FWHMs are sensitivity scenarios, not claims
# about any particular wireline instrument.
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_OUT = os.path.dirname(os.path.abspath(__file__))
FIG_OUT = os.path.join(BASE_OUT, 'supplement_figures')
RES_OUT = os.path.join(BASE_OUT, 'supplement_results')
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(RES_OUT, exist_ok=True)

PERIODS_KYR = (405.0, 95.0, 40.0, 21.0)
RATES = (2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
DZ = 0.1
RESPONSE_FWHM_M = (0.0, 0.3, 0.5, 1.0)
# A fine grid approximates the continuous instrument-response convolution.
FINE_DZ = 0.005
LENGTH_M = 600.0


def observed_period_kyr(period_kyr, sr):
    """Discrete-time alias period at fixed SR and sampled depth step DZ."""
    samples_per_cycle = period_kyr * sr / (1000.0 * DZ)
    discrete_frequency = 1.0 / samples_per_cycle  # cycles per sample
    folded = abs((discrete_frequency + 0.5) % 1.0 - 0.5)
    if folded < 1e-12:
        return np.inf
    return 1000.0 * DZ / (sr * folded)


def amplitude_from_harmonic_projection(depth, observed, wavelength_m):
    """Least-squares sine+cosine amplitude of a known sampled frequency."""
    phi = 2.0 * np.pi * depth / wavelength_m
    X = np.column_stack((np.sin(phi), np.cos(phi), np.ones_like(depth)))
    coef, *_ = np.linalg.lstsq(X, observed, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def run():
    fine_depth = np.arange(0, LENGTH_M + FINE_DZ / 2, FINE_DZ)
    depth = np.arange(0, LENGTH_M + DZ / 2, DZ)
    rows = []

    for sr in RATES:
        for p in PERIODS_KYR:
            lam = p * sr / 1000.0
            signal_fine = np.sin(2 * np.pi * fine_depth / lam)
            t_alias = observed_period_kyr(p, sr)
            for fwhm in RESPONSE_FWHM_M:
                if fwhm:
                    sigma_m = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    filtered = gaussian_filter1d(
                        signal_fine,
                        sigma_m / FINE_DZ,
                        mode='nearest',
                        truncate=5,
                    )
                else:
                    filtered = signal_fine

                sampled = np.interp(depth, fine_depth, filtered)
                # Fit at the true frequency; for aliased cases this fits an
                # indistinguishable discrete alias, NOT successful recovery.
                amp = amplitude_from_harmonic_projection(depth, sampled, lam)
                # Interior RMS avoids the end-effect of gaussian convolution.
                valid = (depth >= 5) & (depth <= LENGTH_M - 5)
                rms = float(np.sqrt(np.mean(sampled[valid] ** 2)))

                rows.append(
                    dict(
                        sr_m_per_Myr=sr,
                        period_kyr=p,
                        wavelength_m=lam,
                        nyquist_allowed=bool(lam >= 2 * DZ),
                        alias_period_kyr=t_alias,
                        response_FWHM_m=fwhm,
                        measured_harmonic_amplitude=amp,
                        measured_rms=rms,
                        theoretical_gaussian_amplitude_factor=(
                            np.exp(
                                -2
                                * np.pi**2
                                * (fwhm / (2 * np.sqrt(2 * np.log(2)))) ** 2
                                / lam**2
                            )
                            if fwhm
                            else 1.0
                        ),
                    )
                )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RES_OUT, 'expA_supp_sampling_resolution_diagnostic.csv')
    df.to_csv(csv_path, index=False)

    # Focused visual that illustrates aliasing and response attenuation.
    selected = df[(df.sr_m_per_Myr == 2.0) & (df.period_kyr == 95.0)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.32, bottom=0.30)

    x_fine = np.arange(0, 3.001, 0.005)
    x = np.arange(0, 3.01, DZ)
    wave = 0.19

    axs[0].plot(
        x_fine,
        np.sin(2 * np.pi * x_fine / wave),
        color='#41627E',
        label='Underlying 95-kyr sinusoid',
    )
    axs[0].plot(
        x,
        np.sin(2 * np.pi * x / wave),
        'o',
        ms=4,
        color='#A65A4A',
        label='0.1-m samples',
    )
    axs[0].set(
        xlabel='Depth (m)',
        ylabel='Signal amplitude',
        title='(a) Undersampling at SR = 2 m/Myr',
    )
    # Move legend below the left panel to avoid covering labels or plot content.
    axs[0].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
        ncol=2,
    )

    xx = selected.response_FWHM_m.to_numpy()
    axs[1].plot(
        xx,
        selected.theoretical_gaussian_amplitude_factor,
        'o-',
        color='#6E8B67',
        label='Gaussian-response transfer',
    )
    axs[1].set(
        xlabel='Assumed vertical-response FWHM (m)',
        ylabel='Relative input amplitude',
        title='(b) Instrument-response sensitivity',
        ylim=(-0.04, 1.06),
    )
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=True)

    fig_path = os.path.join(FIG_OUT, 'expA_supp_sampling_resolution_diagnostic.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print('Saved CSV to:', csv_path)
    print('Saved figure to:', fig_path)
    print(
        '95 kyr / SR=2 m/Myr: true wavelength = 0.19 m; sampled alias period =',
        round(observed_period_kyr(95, 2), 3),
        'kyr',
    )
    print('These are isolated-noiseless-tone sensitivity results, NOT revised Experiment A success rates.')
    return df


if __name__ == '__main__':
    run()
