"""
Implements bootstrap-based inference methods for model given in
Urban et al. (2023)

Implements methods for performing nonparametric bootstrap as well as
parametric likelihood-ratio bootstrap for various inferential objects
of interest.
"""

from multiprocessing import Pool
from functools import partial

import numpy as np

import latent_cn


def get_sigma_from_gamma(gamma):
    """
    Gets real-valued covariance matrix corresponding to gamma

    Turn the complex covariance matrix gamma into a corresponding real-valued
    covariance matrix for (Re X, Im X), where X is the random variable
    with distribution CN(mu, gamma, 0) (i.e., pseudo-covariance 0)

    gamma - a complex-valued covariance matrix

    Returns: real-valued covariance matrix created based on gamma
    """

    return np.concatenate([
        np.concatenate(
            [.5 * np.real(gamma), -.5 * np.imag(gamma)],
            axis=1
        ),
        np.concatenate(
            [.5 * np.imag(gamma), .5 * np.real(gamma)],
            axis=1
        )],
        axis=0
    )


def sample_from_gamma(gamma, n_samples):
    """
    Get n_samples of a complex normal vector distributed according to gamma

    Specifically, get n_samples of CN(0, gamma, 0)

    gamma - complex covariance matrix
    n_samples - positive integer

    Returns: array of size n_samples by gamma.shape[0] which contains the
             samples
    """

    sigma = get_sigma_from_gamma(gamma)
    latent_samples = np.random.multivariate_normal(
        np.zeros(gamma.shape[0] * 2), sigma, n_samples
    )

    return latent_samples[:, :gamma.shape[0]] + \
           1j * latent_samples[:, gamma.shape[0]:]


def gen_data(gamma, etas, betas, n_samples, r_counts):
    """
    Get n_samples from the latent variable model defined by parameters
    gamma, etas, betas

    gamma - complex covariance matrix of dimension len(r_counts)
    etas - list of complex covariance matrices of dimension r_counts[i]
    betas - array of length r_counts[i]
    r_counts - list of integers corresponding to number of entries per
               region
    n_samples - positive integer

    Returns: array of size n_samples by sum(r_counts) containing samples
    """

    sigma = latent_cn.get_full_mat(r_counts, etas, betas, gamma)
    observed_samples = sample_from_gamma(sigma, n_samples)

    return observed_samples


def _inner_bootstrap_ginv(data, r_counts, iters, i):

    np.random.seed()

    print(i, flush=True)

    data_sample = data[np.random.choice(data.shape[0], data.shape[0]), :]
    gamma_est, _, _ = latent_cn.fit(data_sample, r_counts, iters)

    gamma_inv_est = np.linalg.inv(gamma_est)

    return gamma_inv_est


def bootstrap_ginv(data, r_counts, iters=50, boot_iters=10000, procs=4):
    """
    Get bootstrap-based estimates of gamma inverse

    Do boot_iters bootstrap resamples of the rows of data, and estimate
    gamma inverse on each bootstrap resampled dataset.

    data - array of size n_samples by sum(r_counts) containing the data to be
        resampled
    r_counts - list of integers corresponding to number of entries per
               region
    iters - number of iterations to perform EM algorithm in fitting
            the latent covariance matrix
    boot_iters - number of bootstrap datasets to sample
    procs - number of cores to use for parallelized processing

    Returns: matrix of size boot_iters by data.shape[1] by data.shape[1]
             containing the boostrapped estimates of gamma inverse
    """

    partial_boot = partial(_inner_bootstrap_ginv, data, r_counts, iters)

    gamma_invs = []

    skip = 10000
    for i in range(0, boot_iters, skip):
        with Pool(procs) as p:
            tmp_ginvs = p.map(
                partial_boot, range(i, min(boot_iters, i + skip))
            )
        gamma_invs.extend(tmp_ginvs)

    gamma_invs = np.array(gamma_invs)

    return gamma_invs


def _inner_bootstrap_lr_test(r_counts, iters, orig_h0_gamma, orig_etas,
                             orig_betas, n_samples, region_i, region_j, i):

    np.random.seed()
    print(i, flush=True)
    data = gen_data(orig_h0_gamma, orig_etas, orig_betas, n_samples, r_counts)

    gamma, etas, betas = latent_cn.fit(data, r_counts, iters)

    h0_gamma, h0_etas, h0_betas = latent_cn.fit(
        data, r_counts, iters, set_inds_zero=(region_i, region_j)
    )

    h0_likelihood = latent_cn.log_likelihood(
        data, r_counts, h0_etas, h0_betas, h0_gamma
    )
    ha_likelihood = latent_cn.log_likelihood(
        data, r_counts, etas, betas, gamma
    )
    assert h0_likelihood <= ha_likelihood

    ratio = -2 * (h0_likelihood - ha_likelihood)

    return ratio


def bootstrap_lr_test(data, r_counts, iters,
                      region_i, region_j, boot_iters=100, procs=4):
    """
    Obtain likelihood ratio statistics with a parametric bootstrap under
    the constraint that the element at (region_i, region_j) of gamma inverse
    is zero.

    See Urban et al. (2023) for more information.

    data - array of size n_samples by sum(r_counts) containing the data to be
        resampled
    r_counts - list of integers corresponding to number of entries per
               region
    iters - number of iterations to perform EM algorithm in fitting
            the latent covariance matrix
    region_i - row of element of latent inverse covariance matrix to
               set to zero
    region_j - column of element of latent inverse covariance matrix to
               set to zero
    boot_iters - number of bootstrap datasets to sample
    procs - number of cores to use for parallelized processing

    Returns: tuple of (sorted list of likelihood ratio statistics from
             bootstrapped datasets, likelihood ratio statistic on original
             dataset)
    """

    gamma, etas, betas = latent_cn.fit(data, r_counts, iters)
    n_samples = data.shape[0]

    h0_gamma, h0_etas, h0_betas = latent_cn.fit(
        data, r_counts, iters, set_inds_zero=(region_i, region_j)
    )

    h0_likelihood = latent_cn.log_likelihood(
        data, r_counts, h0_etas, h0_betas, h0_gamma
    )
    ha_likelihood = latent_cn.log_likelihood(
        data, r_counts, etas, betas, gamma
    )

    assert h0_likelihood < ha_likelihood

    base_ratio = -2 * (h0_likelihood - ha_likelihood)

    boot_ratios = []

    orig_h0_gamma = np.copy(h0_gamma)
    orig_h0_etas = [np.copy(h0_eta) for h0_eta in h0_etas]
    orig_h0_betas = [np.copy(h0_beta) for h0_beta in h0_betas]

    inner_boot_fun = partial(
        _inner_bootstrap_lr_test, r_counts, iters, orig_h0_gamma, orig_h0_etas,
        orig_h0_betas, n_samples, region_i, region_j
    )

    skip = 10000
    for i in range(0, boot_iters, skip):
        with Pool(procs) as p:
            tmp_boot_ratios = p.map(
                inner_boot_fun, range(i, min(boot_iters, i + skip))
            )
        boot_ratios.extend(tmp_boot_ratios)

    boot_ratios = sorted(boot_ratios)
    return boot_ratios, base_ratio
