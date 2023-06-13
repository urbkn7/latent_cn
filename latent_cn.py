"""
Implements fitting functions for latent variable model in Urban et
al. (2023).

Main method provided is the fit() method which takes in a dataset and
returns fitted parameters according to latent variable model.
"""

import numpy as np
import scipy.linalg
import cvxpy as cp


def r_map(r_counts, r):
    """
    Returns consecutive integers corresponding to the rth entry in r_counts

    Assume that we have sum(r_counts) elements in some list and we want
    to obtain the entries of that list corresponding to the rth entry
    in r_counts.  This function returns the indices of those elements.

    r_counts - list of integers corresponding to number of entries per
               region
    r - number between 0 and len(r_counts)

    Returns: array of integers of length r_counts[r]
    """

    if r == 0:
        low = 0
        high = r_counts[r]
    else:
        low = sum(r_counts[:r])
        high = sum(r_counts[:r + 1])

    return range(low, high)


def get_full_mat(r_counts, etas, betas, gamma):
    """
    Returns the marginal covariance matrix given the parameters of
    the latent variable model

    r_counts - list of integers corresponding to number of entries per
               region
    etas - list of complex covariance matrices of dimension r_counts[i]
    betas - array of length r_counts[i]
    gamma - complex covariance matrix of dimension len(r_counts)

    Returns: complex covariance matrix of dimension sum(r_counts)
    """

    sigma = np.zeros((sum(r_counts), sum(r_counts))).astype(complex)
    for r1 in range(len(r_counts)):
        r1_inds = list(r_map(r_counts, r1))
        for r2 in range(len(r_counts)):
            r2_inds = list(r_map(r_counts, r2))
            sigma[np.ix_(r1_inds, r2_inds)] += \
                betas[r1].reshape((-1, 1)) @ \
                betas[r2].reshape((1, -1)).conj() * gamma[r1, r2]
            if r1 == r2:
                sigma[np.ix_(r1_inds, r2_inds)] += etas[r1]

    return sigma


def log_likelihood(data, r_counts, etas, betas, gamma):
    """
    Returns log likelihood of data (data) given parameters

    data - N times sum(r_counts) data matrix
    r_counts - list of integers corresponding to number of entries per
               region
    etas - list of complex covariance matrices of dimension r_counts[i]
    betas - array of length r_counts[i]
    gamma - complex covariance matrix of dimension len(r_counts)

    Returns: log likelihood
    """

    val = 0
    sigma = np.zeros((data.shape[1], data.shape[1])).astype(complex)
    for r1 in range(len(r_counts)):
        r1_inds = list(r_map(r_counts, r1))
        for r2 in range(len(r_counts)):
            r2_inds = list(r_map(r_counts, r2))
            sigma[np.ix_(r1_inds, r2_inds)] += \
                betas[r1].reshape((-1, 1)) @ \
                betas[r2].reshape((1, -1)).conj() * gamma[r1, r2]
            if r1 == r2:
                sigma[np.ix_(r1_inds, r2_inds)] += etas[r1]

    sigma_inv = scipy.linalg.pinvh(sigma)
    val -= data.shape[0] * np.linalg.slogdet(sigma)[1]
    val -= data.shape[0] * data.shape[1] * np.log(np.pi)
    val -= np.trace(data.T @ data.conj() @ sigma_inv)

    return np.real(val)


def fit(data, r_counts, iters, set_inds_zero=None):
    """
    Fits parameters given data

    data - N times sum(r_counts) data matrix
    r_counts - list of integers corresponding to number of entries per
               region
    iters - how many iterations of EM updates to perform
    set_inds_zero - Either tuple of two integers between 0 and
                    len(r_counts) - 1 (inclusive) which corresponds to an
                    entry of gamma inverse which is constrained to be zero
                    (used in computing likelihood ratio statistics) or None
                    if unconstrained optimization is performed

    Returns: gamma, etas, betas, corresponding to the parameters of the
             latent variable model
    """

    etas = [np.identity(rc).astype(complex) for rc in r_counts]
    gamma = np.zeros((len(r_counts), len(r_counts))).astype(complex)

    betas = [np.zeros(rc).astype(complex) for rc in r_counts]

    sigma = data.conj().T @ data / data.shape[0]

    # Get starting values via method of moments-based method
    for r1 in range(len(r_counts)):

        r1_inds = list(r_map(r_counts, r1))

        for r2 in range(len(r_counts)):

            if r1 == r2:
                continue

            r2_inds = list(r_map(r_counts, r2))
            chunk = sigma[np.ix_(r1_inds, r2_inds)]

            normed_vecs = [
                chunk[:, i] / np.linalg.norm(chunk[:, i])
                for i in range(chunk.shape[1])
            ]

            rot_vecs = [
                vec * np.exp(-1j * np.angle(vec[0]))
                for vec in normed_vecs
            ]

            assert (np.all([np.abs(np.imag(vec[0])) < 1e-8
                           for vec in rot_vecs]))

            mean_vec = np.mean(rot_vecs, axis=0)
            betas[r1] += mean_vec

            normed_vecs_2 = [
                chunk[i, :] / np.linalg.norm(chunk[i, :])
                for i in range(chunk.shape[0])
            ]

            rot_vecs_2 = [
                vec * np.exp(-1j * np.angle(vec[0]))
                for vec in normed_vecs_2
            ]

            mean_vec2 = np.mean(rot_vecs_2, axis=0)

            gamma_el = np.mean(
                chunk / (mean_vec.reshape((-1, 1)) @
                         mean_vec2.reshape((1, -1)).conj())
            )

            gamma[r1, r2] = gamma_el
            gamma[r2, r1] = gamma_el.conj()

        betas[r1] = betas[r1] / np.linalg.norm(betas[r1])
        betas[r1] = betas[r1] * np.exp(-1j * np.angle(betas[r1][0]))

        gamma[r1, r1] = np.abs(np.mean(
            sigma[np.ix_(r1_inds, r1_inds)] /
            (betas[r1].reshape((-1, 1)) @
             betas[r1].reshape((1, -1)).conj())
        ))

        etas[r1] = (np.identity(len(r1_inds)) * 1e-8).astype(complex)

    # Convex optimization problem is used for ensuring identifiability
    # constraints are met
    cvx_vars = []
    objectives = []
    constraints = []
    eta_params = []
    beta_outer_params = []
    problems = []

    for i in range(len(r_counts)):
        a = cp.Variable(1)
        cvx_vars.append(a)
        objectives.append(cp.Maximize(a))
        beta_outer_param = cp.Parameter(
            (betas[i].shape[0], betas[i].shape[0]), complex=True)
        beta_outer_params.append(beta_outer_param)
        eta_param = cp.Parameter(etas[i].shape, complex=True)
        eta_params.append(eta_param)
        constraint = [
            a >= 0,
            (eta_param - a * beta_outer_param) >> 0
        ]
        constraints.append(constraint)

        problems.append(cp.Problem(objectives[-1], constraint))

    # To optimize under H0: that the entry of gamma inverse
    # corresponding to set_inds_zero is 0
    if set_inds_zero is not None:
        zero_ind_gamma_inv = cp.Variable((len(r_counts), len(r_counts)),
                                         hermitian=True)
        zero_ind_constraints = [
            zero_ind_gamma_inv[set_inds_zero[0],
                               set_inds_zero[1]] == 0,
            zero_ind_gamma_inv == zero_ind_gamma_inv.H
        ]
        gamma_inv_mult_param = cp.Parameter(
            (len(r_counts), len(r_counts)), complex=True
        )
        zero_ind_obj = cp.Maximize(
            cp.log_det(zero_ind_gamma_inv) -
            cp.real(cp.trace(
                zero_ind_gamma_inv @ gamma_inv_mult_param)
            )
        )
        zero_ind_prob = cp.Problem(zero_ind_obj, zero_ind_constraints)

    # Perform EM updates
    for i in range(iters):

        mat_w_inv = scipy.linalg.pinvh(gamma)
        mat_v = np.zeros((data.shape[0], len(r_counts))).astype(complex)

        for r in range(len(r_counts)):

            inds = r_map(r_counts, r)

            eta_r_inv = scipy.linalg.pinvh(etas[r])

            mat_w_inv[r, r] += betas[r].conj().T @ eta_r_inv @ betas[r]
            mat_v[:, r] = (betas[r].conj().T @ eta_r_inv @ data[:, inds].T).T

        mat_w = scipy.linalg.pinvh(mat_w_inv)
        mu = (mat_w @ mat_v.T).T

        if set_inds_zero is not None:
            gamma_inv_mult_param.value = \
                mat_w + mu.T @ mu.conj() / mu.shape[0]
            zero_ind_prob.solve()
            gamma = np.linalg.inv(zero_ind_gamma_inv.value)
        else:
            gamma = mat_w + mu.T @ mu.conj() / mu.shape[0]

        assert min(np.linalg.eigh(gamma)[0]) >= 0
        assert np.sum(np.abs(gamma - gamma.conj().T)) < 1e-5

        for r in range(len(r_counts)):
            inds = list(r_map(r_counts, r))

            betas[r][:] = 0

            betas[r] += \
                (data[:, inds].T @
                 mu[:, r].reshape((-1, 1)).conj()).reshape((-1,))
            denom = mat_w[r, r] + np.linalg.norm(mu[:, r]) ** 2

            betas[r] = betas[r] / denom

        for r in range(len(r_counts)):
            inds = list(r_map(r_counts, r))

            etas[r][:, :] = 0

            etas[r] += data[:, inds].T @ data[:, inds].conj()
            etas[r] -= data[:, inds].T @ \
                (mu[:, r].conj().reshape((-1, 1)) @
                    betas[r].conj().reshape((1, -1)))
            etas[r] -= (data[:, inds].T @
                        (mu[:, r].conj().reshape((-1, 1)) @
                         betas[r].conj().reshape((1, -1)))).conj().T
            etas[r] += (
                data.shape[0] * mat_w[r, r] + mu[:, r].reshape((1, -1)).conj() @
                mu[:, r].reshape((-1, 1))
            ) * (
                betas[r].reshape((-1, 1)) @
                betas[r].reshape((1, -1)).conj()
            )

            etas[r] = etas[r] / data.shape[0]

            assert np.sum(np.abs(etas[r] - etas[r].conj().T)) < 1e-5

        # Adjust estimates to ensure identifiability
        for r in range(len(r_counts)):
            if np.linalg.norm(betas[r]) > 0:

                gamma[r, :] *= np.linalg.norm(betas[r])
                gamma[:, r] *= np.linalg.norm(betas[r])
                gamma[r, :] *= betas[r][0] / np.abs(betas[r][0])
                gamma[:, r] *= (betas[r][0] / np.abs(betas[r][0])).conj()

                betas[r] = betas[r] / np.linalg.norm(betas[r])
                betas[r] *= (betas[r][0] / np.abs(betas[r][0])).conj()

                eigs, vecs = np.linalg.eigh(etas[r])
                pre_eta = np.copy(etas[r])
                if min(eigs) < 0:
                    # Fix numerical issue with etas to make it PSD
                    eigs[eigs < 1e-10] = 1e-10
                    etas[r] = vecs @ np.diag(eigs) @ vecs.conj().T
                    assert np.sum(np.abs(etas[r] - pre_eta)) < 1

                eta_params[r].value = etas[r]
                beta_outer_params[r].value = (
                    betas[r].reshape((-1, 1)) @
                    betas[r].reshape((1, -1)).conj()
                )

                problems[r].solve()

                opt = cvx_vars[r].value

                etas[r] = (
                    etas[r] - opt * betas[r].reshape((-1, 1)) @
                    betas[r].reshape((1, -1)).conj()
                )

                gamma[r, r] += opt

    return gamma, etas, betas
