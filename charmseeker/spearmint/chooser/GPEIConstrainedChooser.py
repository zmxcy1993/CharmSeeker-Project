import os
import gp
import util
import tempfile
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps
import scipy.optimize as spo
import pickle
import multiprocessing
import copy

from helpers import *
from Locker import *


# Wrapper function to pass to parallel ei optimization calls
def optimize_pt(c, b, comp, pend, values, cos, model):
    ret = spo.fmin_l_bfgs_b(model.grad_optimize_ei_over_hyper_params,
                            c.flatten(), args=(comp, pend, values, cos), bounds=b, disp=0)
    return ret[0]


def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    if 'budget' in args.keys():
        args['budget'] = float(args['budget'])
    return GPEIConstrainedChooser(expt_dir, **args)

"""
Chooser module for Constrained Gaussian process expected improvement.
Candidates are sampled densely in the unit hypercube and then a subset
of the most promising points are optimized to maximize constrained EI
over hyper-parameter samples.  Slice sampling is used to sample
Gaussian process hyper-parameters for two GPs, one over the objective
function and the other a probit likelihood classification GP that estimates the
probability that a point is outside of the constraint space.
"""


class GPEIConstrainedChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iterations=20, pending_samples=100,
                 noiseless=False, burn_in=100, grid_subset=10, verbosity=0, budget=1):
        self.cov_func = getattr(gp, covar)
        self.locker = Locker()
        self.state_pkl = os.path.join(expt_dir, f"{self.__module__}.pkl")

        self.stats_file = os.path.join(expt_dir, f"{self.__module__}_hyper_parameters.txt")
        self.mcmc_iterations = mcmc_iterations
        self.burn_in = burn_in
        self.needs_burn_in = True
        self.pending_samples = pending_samples
        self.D = -1
        self.hyper_iterations = 1

        self.grid_subset = int(grid_subset)  # Number of points to optimize EI over
        self.noiseless = bool(int(noiseless))
        self.constraint_noiseless = False
        self.hyper_samples = []
        self.constraint_hyper_samples = []
        self.verbosity = int(verbosity)

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale = 1    # zero-mean log normal prior
        self.max_ls = 2    # top-hat prior on length scales

        self.constraint_noise_scale = 0.1  # horseshoe prior
        self.constraint_amp2_scale = 1    # zero-mean log normal prio
        self.constraint_max_ls = 2   # top-hat prior on length scales
        self.budget = np.log(budget)  # convert to log scale

    # A simple function to dump out hyper parameters to allow for a hot start
    # if the optimization is restarted.
    def dump_hyper_params(self):

        self.locker.lock_wait(self.state_pkl)

        # Write the hyper parameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump({'dims': self.D, 'ls': self.ls, 'amp2': self.amp2, 'noise': self.noise, 'mean': self.mean,
                     'constraint_ls': self.constraint_ls, 'constraint_amp2': self.constraint_amp2,
                     'constraint_noise': self.constraint_noise, 'constraint_mean': self.constraint_mean}, fh)
        fh.close()

        cmd = f'mv {fh.name} {self.state_pkl}'
        os.system(cmd)
        self.locker.unlock(self.state_pkl)

        # Write the hyper parameters out to a human readable file as well
        fh = open(self.stats_file, 'w')
        fh.write('Mean Noise Amplitude <length scales>\n')
        fh.write('-----------ALL SAMPLES-------------\n')
        mean_hyper_params = 0 * np.hstack(self.hyper_samples[0])
        for i in self.hyper_samples:
            hyper_params = np.hstack(i)
            mean_hyper_params += (1/float(len(self.hyper_samples))) * hyper_params
            for j in hyper_params:
                fh.write(str(j) + ' ')
            fh.write('\n')

        fh.write('-----------MEAN OF SAMPLES-------------\n')
        for j in mean_hyper_params:
            fh.write(str(j) + ' ')
        fh.write('\n')
        fh.close()

    def _real_init(self, dims, values, costs):

        self.locker.lock_wait(self.state_pkl)
        self.random_state = npr.get_state()

        if os.path.exists(self.state_pkl):
            fh = open(self.state_pkl, 'rb')
            state = pickle.load(fh)
            fh.close()

            self.D = state['dims']
            self.ls = state['ls']
            self.amp2 = state['amp2']
            self.noise = state['noise']
            self.mean = state['mean']
            self.constraint_ls = state['constraint_ls']
            self.constraint_amp2 = state['constraint_amp2']
            self.constraint_noise = state['constraint_noise']
            self.constraint_mean = state['constraint_mean']
            self.needs_burn_in = False
        else:
            self.D = dims
            self.ls = np.ones(self.D)
            self.constraint_ls = np.ones(self.D)

            self.amp2 = np.std(values) + 1e-4
            self.mean = np.mean(values)

            self.constraint_amp2 = np.std(costs) + 1e-4
            self.constraint_mean = np.mean(costs)

            self.noise = 1e-3
            self.constraint_noise = 1e-3

        self.locker.unlock(self.state_pkl)

    def cov(self, amp2, ls, x1, x2=None):
        if x2 is None:
            return amp2 * (self.cov_func(ls, x1, None) + 1e-6 * np.eye(x1.shape[0]))
        else:
            return amp2 * self.cov_func(ls, x1, x2)

    def next(self, grid, values, costs, candidates, pending, completes):

        # Don't bother using fancy GP stuff at first.
        if completes.shape[0] < 3:
            return int(candidates[0])

        # Grab out the relevant sets.
        complete_configs = grid[completes, :]
        candidate_configs = grid[candidates, :]
        pending_configs = grid[pending, :]
        complete_values = values[completes]
        complete_costs = costs[completes]

        # Convert from log values to normal scale
        print(f"current costs: {np.exp(complete_costs)}")
        print(f"self.budget: {np.exp(self.budget):.5f}")

        idx = np.logical_and(complete_costs <= self.budget, np.isfinite(complete_values))
        good_values = np.nonzero(idx)[0]
        bad_values = np.nonzero(np.logical_not(idx))[0]
        print(f'Found {bad_values.shape[0]} constraint violating jobs')
        print(f'Received {good_values.shape[0]} valid results')

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], complete_values, complete_costs)

        if good_values.shape[0] < 2:
            self.constraint_hyper_samples = []
            for _ in range(self.mcmc_iterations):
                self.sample_constraint_hyper_params(complete_configs, complete_costs)
            overall_ei = self.weight_over_hyper_params(complete_configs, candidate_configs, complete_costs)
            best_candidate = np.argmax(np.mean(overall_ei, axis=1))
            return int(candidates[best_candidate])

        # Spray a set of candidates around the min so far
        num_candidates = candidate_configs.shape[0]
        best_complete = np.argmin(complete_values[good_values])
        new_candidates = np.vstack((np.random.randn(10, complete_configs.shape[1]) * 0.001
                                    + complete_configs[best_complete, :], candidate_configs))

        if self.mcmc_iterations > 0:
            if self.needs_burn_in:
                self.amp2 = np.std(complete_values[good_values]) + 1e-4
                self.mean = np.mean(complete_values[good_values])
                for mcmc_iter in range(self.burn_in):
                    self.sample_constraint_hyper_params(complete_configs, complete_costs)
                    self.sample_hyper_params(complete_configs[good_values, :], complete_values[good_values])
                    log(f"BURN {mcmc_iter+1}/{self.burn_in}] mean: {self.mean:.2f} amp: {np.sqrt(self.amp2):.2f}"
                        f" noise: {self.noise:.4f} min_ls: {np.min(self.ls):.4f} max_ls: {np.max(self.ls):.4f}")

                self.needs_burn_in = False

            self.hyper_samples = []
            self.constraint_hyper_samples = []

            for mcmc_iter in range(self.mcmc_iterations):
                self.sample_constraint_hyper_params(complete_configs, complete_costs)
                self.sample_hyper_params(complete_configs[good_values, :], complete_values[good_values])

                if self.verbosity > 0:
                    log(f"{mcmc_iter+1}/{self.mcmc_iterations}] mean: {self.mean:.2f} amp: {np.sqrt(self.amp2):.2f}"
                        f" noise: {self.noise:.4f} min_ls: {np.min(self.ls):.4f} max_ls: {np.max(self.ls):.4f}")

                    log(f"{mcmc_iter+1}/{self.mcmc_iterations}] constraint_mean: {self.constraint_mean:.2f} "
                        f"constraint_amp: {np.sqrt(self.constraint_amp2):.2f}, constraint_noise: "
                        f"{self.constraint_noise:.4f}, constraint_min_ls: {np.min(self.constraint_ls):.4f} "
                        f"constraint_max_ls: {np.max(self.constraint_ls):.4f}")

            self.dump_hyper_params()

            # Pick the top candidates to optimize over
            overall_ei = self.ei_over_hyper_params(complete_configs, pending_configs, new_candidates,
                                                   complete_values, complete_costs)
            indices = np.argsort(np.mean(overall_ei, axis=1))[-self.grid_subset:]
            new_candidates = new_candidates[indices, :]

            # Adjust the candidates to hit ei peaks
            b = []
            for _ in range(0, candidate_configs.shape[1]):
                b.append((0, 1))

            # Optimize each point in parallel
            pool = multiprocessing.Pool(self.grid_subset)
            results = [pool.apply_async(optimize_pt, args=(c, b, complete_configs, pending_configs, complete_values,
                                                           complete_costs, copy.copy(self))) for c in new_candidates]
            for res in results:
                candidate_configs = np.vstack((candidate_configs, res.get(1024)))
            pool.close()

            overall_ei = self.ei_over_hyper_params(complete_configs, pending_configs, candidate_configs,
                                                   complete_values, complete_costs)
            best_candidate = np.argmax(np.mean(overall_ei, axis=1))

            self.dump_hyper_params()
            if best_candidate >= num_candidates:
                return int(num_candidates), candidate_configs[best_candidate, :]

            return int(candidates[best_candidate])

        else:
            print('This Chooser module permits only slice sampling with > 0 samples.')
            raise Exception('mcmc_iterations <= 0')

    def weight_over_hyper_params(self, complete_configs, candidate_configs, complete_costs):
        print("Compute weight over hyper parameters")
        overall_ei = np.zeros((candidate_configs.shape[0], self.mcmc_iterations))

        for mcmc_iter in range(self.mcmc_iterations):
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]
            self.constraint_mean = constraint_hyper[0]
            self.constraint_noise = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]
            overall_ei[:, mcmc_iter] = self.compute_constraint_satisfy(complete_configs, candidate_configs,
                                                                       complete_costs)
        return overall_ei

    def compute_constraint_satisfy(self, complete_full, candidate_configs, complete_costs):
        cons_comp_cov = self.cov(self.constraint_amp2, self.constraint_ls, complete_full)
        cons_cand_cross = self.cov(self.constraint_amp2, self.constraint_ls, complete_full, candidate_configs)

        cons_obsv_cov = cons_comp_cov + self.constraint_noise * np.eye(complete_full.shape[0])
        cons_obsv_chol = spla.cholesky(cons_obsv_cov, lower=True)

        t_alpha = spla.cho_solve((cons_obsv_chol, True), complete_costs - self.constraint_mean)
        t_beta = spla.solve_triangular(cons_obsv_chol, cons_cand_cross, lower=True)

        cons_func_m = np.dot(cons_cand_cross.T, t_alpha) + self.constraint_mean
        cons_func_v = self.constraint_amp2 * (1 + 1e-6) - np.sum(t_beta ** 2, axis=0)
        cons_func_s = np.sqrt(cons_func_v)
        weight = sps.norm.cdf(self.budget, loc=cons_func_m, scale=cons_func_s)
        return weight

    def ei_over_hyper_params(self, complete_configs, pending, candidate_configs, complete_values, complete_costs):

        overall_ei = np.zeros((candidate_configs.shape[0], self.mcmc_iterations))

        for mcmc_iter in range(self.mcmc_iterations):
            hyper = self.hyper_samples[mcmc_iter]
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]

            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            self.constraint_mean = constraint_hyper[0]
            self.constraint_noise = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]

            overall_ei[:, mcmc_iter] = self.compute_constrained_ei(complete_configs, pending, candidate_configs,
                                                                   complete_values, complete_costs)
        return overall_ei

    def compute_constrained_ei(self, complete_configs, pending, candidate_configs, complete_values, complete_costs):
        complete_full = complete_configs.copy()
        complete_configs = complete_configs[complete_costs <= self.budget, :]
        complete_values = complete_values[complete_costs <= self.budget]
        if pending.shape[0] == 0:
            return self.compute_constrained_ei_no_pending(complete_configs, complete_values, complete_full,
                                                          complete_costs, candidate_configs)
        else:
            return self.compute_constrained_ei_pending(complete_configs, complete_values, complete_full, complete_costs,
                                                       candidate_configs, pending)

    def compute_constrained_ei_no_pending(self, completes, values, complete_full, costs, candidates):
        if np.all(costs <= self.budget) or np.all(costs > self.budget):
            weight = 1
        else:
            cons_comp_cov = self.cov(self.constraint_amp2, self.constraint_ls, complete_full)
            cons_cand_cross = self.cov(self.constraint_amp2, self.constraint_ls, complete_full, candidates)

            cons_obsv_cov = cons_comp_cov + self.constraint_noise * np.eye(complete_full.shape[0])
            cons_obsv_chol = spla.cholesky(cons_obsv_cov, lower=True)

            t_alpha = spla.cho_solve((cons_obsv_chol, True), costs - self.constraint_mean)
            t_beta = spla.solve_triangular(cons_obsv_chol, cons_cand_cross, lower=True)

            # Predict marginal mean times and (possibly) variances
            cons_func_m = np.dot(cons_cand_cross.T, t_alpha) + self.constraint_mean
            cons_func_v = self.constraint_amp2 * (1 + 1e-6) - np.sum(t_beta ** 2, axis=0)
            cons_func_s = np.sqrt(cons_func_v)
            weight = sps.norm.cdf(self.budget, loc=cons_func_m, scale=cons_func_s)

        best = np.min(values)

        comp_cov = self.cov(self.amp2, self.ls, completes)
        cand_cross = self.cov(self.amp2, self.ls, completes, candidates)

        obsv_cov = comp_cov + self.noise * np.eye(completes.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        alpha = spla.cho_solve((obsv_chol, True), values - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        func_s = np.sqrt(func_v)
        u = (best - func_m) / func_s
        n_cdf = sps.norm.cdf(u)
        n_pdf = sps.norm.pdf(u)
        ei = func_s * (u * n_cdf + n_pdf)

        # print("weight: ", weight)
        constrained_ei = ei * weight
        return constrained_ei

    # Adjust points by optimizing EI over a set of hyper parameter samples
    def grad_optimize_ei_over_hyper_params(self, candidate, comp, pend, values, cos, compute_grad=True):
        summed_ei = 0
        summed_grad_ei = np.zeros(candidate.shape).flatten()

        for mcmc_iter in range(self.mcmc_iterations):
            hyper = self.hyper_samples[mcmc_iter]
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]

            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            self.constraint_mean = constraint_hyper[0]
            self.constraint_noise = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]

            if compute_grad:
                (ei, g_ei) = self.grad_optimize_ei(candidate, comp, pend, values, cos, compute_grad)
                summed_grad_ei = summed_grad_ei + g_ei
            else:
                ei = self.grad_optimize_ei(candidate, comp, pend, values, cos, compute_grad)

            summed_ei += ei

        if compute_grad:
            return summed_ei, summed_grad_ei
        else:
            return summed_ei

    def grad_optimize_ei(self, candidate, comp, pend, values, cos, compute_grad=True):
        complete_full = comp.copy()
        comp = comp[cos <= self.budget, :]
        values = values[cos <= self.budget]

        if pend.shape[0] == 0:
            return self.grad_optimize_ei_no_pending(comp, values, complete_full, cos, candidate, compute_grad)
        else:
            return self.grad_optimize_ei_pending(comp, values, complete_full, cos, candidate, pend, compute_grad)

    def grad_optimize_ei_pending(self, comp, vals, compfull, cos, cand, pend, compute_grad):

        # Create a composite vector of complete and pending.
        comp_pend = np.concatenate((comp, pend))
        cons_comp_pend = np.concatenate((compfull, pend))

        comp_pend_cov = self.cov(self.amp2, self.ls, comp_pend) + self.noise * np.eye(comp_pend.shape[0])
        cons_comp_pend_cov = self.cov(self.constraint_amp2, self.constraint_ls,
                                      cons_comp_pend) + self.constraint_noise * np.eye(cons_comp_pend.shape[0])
        comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)
        cons_comp_pend_chol = spla.cholesky(cons_comp_pend_cov, lower=True)

        obsv_chol = comp_pend_chol[:comp.shape[0], :comp.shape[0]]
        cons_obsv_chol = cons_comp_pend_chol[:compfull.shape[0], :compfull.shape[0]]

        pend_cross = self.cov(self.amp2, self.ls, comp, pend)
        cons_pend_cross = self.cov(self.constraint_amp2, self.constraint_ls, compfull, pend)
        pend_kappa = self.cov(self.amp2, self.ls, pend)
        cons_pend_kappa = self.cov(self.constraint_amp2, self.constraint_ls, pend)

        alpha = spla.cho_solve((obsv_chol, True), vals - self.mean)
        cons_alpha = spla.cho_solve((cons_obsv_chol, True), cos - self.constraint_mean)
        beta = spla.cho_solve((obsv_chol, True), pend_cross)
        cons_beta = spla.cho_solve((cons_obsv_chol, True), cons_pend_cross)

        # Finding predictive means and variances.
        pend_m = np.dot(pend_cross.T, alpha) + self.mean
        cons_pend_m = np.dot(cons_pend_cross.T, cons_alpha) + self.constraint_mean
        pend_k = pend_kappa - np.dot(pend_cross.T, beta)
        cons_pend_k = cons_pend_kappa - np.dot(cons_pend_cross.T, cons_beta)
        pend_chol = spla.cholesky(pend_k, lower=True)
        cons_pend_chol = spla.cholesky(cons_pend_k, lower=True)

        # Make predictions.
        npr.set_state(self.randomstate)
        pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0], self.pending_samples)) + pend_m[:, None]
        cons_pend_fant = np.dot(cons_pend_chol, npr.randn(pend.shape[0], self.pending_samples)) + cons_pend_m[:, None]

        m_pend_fant = np.mean(pend_fant, axis=1)
        m_cons_pend_fant = np.mean(cons_pend_fant, axis=1)
        print("m_pend_fant shape: ", m_pend_fant.shape)
        print("m_cons_pend_fant shape: ", m_cons_pend_fant.shape)

        valid_pend = m_pend_fant[m_cons_pend_fant <= self.budget]

        if len(valid_pend) > 0:
            new_comp = comp.vstack(comp, pend[m_cons_pend_fant <= self.budget, :])
            new_vals = np.concatenate((vals, valid_pend))
        else:
            new_comp = comp
            new_vals = vals

        new_compfull = cons_comp_pend
        new_cos = np.concatenate((cos, m_cons_pend_fant))
        return self.grad_optimize_ei_nopend(new_comp, new_vals, new_compfull, new_cos, cand, compute_grad)

    def grad_optimize_ei_no_pending(self, comp, vals, compfull, cos, cand, compute_grad):
        cand = np.reshape(cand, (-1, comp.shape[1]))
        best = np.min(vals)

        cons_comp_cov = self.cov(self.constraint_amp2, self.constraint_ls, compfull)
        cons_cand_cross = self.cov(self.constraint_amp2, self.constraint_ls, compfull, cand)

        cons_obsv_cov = (cons_comp_cov + self.constraint_noise * np.eye(compfull.shape[0]))
        cons_obsv_chol = spla.cholesky(cons_obsv_cov, lower=True)

        t_alpha = spla.cho_solve((cons_obsv_chol, True), cos - self.constraint_mean)
        t_beta = spla.solve_triangular(cons_obsv_chol, cons_cand_cross, lower=True)

        cons_func_m = np.dot(cons_cand_cross.T, t_alpha) + self.constraint_mean
        cons_func_v = self.constraint_amp2 * (1 + 1e-6) - np.sum(t_beta ** 2, axis=0)
        cons_func_s = np.sqrt(cons_func_v)
        weight = sps.norm.cdf(self.budget, loc=cons_func_m, scale=cons_func_s)

        comp_cov = self.cov(self.amp2, self.ls, comp)
        cand_cross = self.cov(self.amp2, self.ls, comp, cand)

        obsv_cov = comp_cov + self.noise * np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)

        alpha = spla.cho_solve((obsv_chol, True), vals - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta**2, axis=0)

        func_s = np.sqrt(func_v)
        u = (best - func_m) / func_s
        n_cdf = sps.norm.cdf(u)
        n_pdf = sps.norm.pdf(u)
        ei = func_s * (u * n_cdf + n_pdf)

        cons_ei = -np.sum(ei * weight)

        if not compute_grad:
            return cons_ei

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -n_cdf
        g_ei_s2 = 0.5 * n_pdf / func_s

        cov_grad_func = getattr(gp, f'grad_{self.cov_func.__name__}')
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve((obsv_chol, True), cand_cross).transpose(), grad_cross)

        # grad_xp shape: (1, cand.shape(0))
        grad_xp = 0.5 * self.amp2 * (grad_xp_m * g_ei_m + grad_xp_v * g_ei_s2)

        if np.all(cos <= self.budget) or np.all(cos > self.budget):
            return -np.sum(ei), grad_xp.flatten()
        else:
            cons_cand_cross_grad = cov_grad_func(self.constraint_ls, compfull, cand)
            cons_grad_cross_t = np.squeeze(cons_cand_cross_grad)

            cons_grad_xp_m = np.dot(t_alpha.transpose(), cons_grad_cross_t)
            cons_grad_xp_s = np.dot(-2 * spla.cho_solve(
                (cons_obsv_chol, True), cons_cand_cross).transpose(), cons_grad_cross_t)

            cons_pdf = sps.norm.pdf(self.budget, loc=cons_func_m, scale=cons_func_s)

            cons_g_ei_m = - 1 / cons_func_v
            cons_g_ei_s = 0.5 * (cons_func_m - self.budget) / cons_func_s**3

            cons_grad_xp = 0.5 * self.constraint_amp2 * cons_pdf * (
                    cons_g_ei_m * cons_grad_xp_m + cons_g_ei_s * cons_grad_xp_s)

            # cons_grad_xp shape: (1, cand.shape(0))
            grad_xp = weight * grad_xp + ei * cons_grad_xp

            return cons_ei, grad_xp.flatten()

    def compute_constrained_ei_pending(self, comp, vals, compfull, cos, cand, pend):
        comp_pend = np.concatenate((comp, pend))
        comp_pend_full = np.concatenate((compfull, pend))

        comp_pend_cov = self.cov(self.amp2, self.ls, comp_pend) + self.noise * np.eye(comp_pend.shape[0])
        cons_comp_pend_cov = self.cov(self.constraint_amp2, self.constraint_ls,
                                      comp_pend_full) + self.constraint_noise * np.eye(comp_pend_full.shape[0])

        comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)
        cons_comp_pend_chol = spla.cholesky(cons_comp_pend_cov, lower=True)

        obsv_chol = comp_pend_chol[:comp.shape[0], :comp.shape[0]]
        cons_obsv_chol = cons_comp_pend_chol[:compfull.shape[0], :compfull.shape[0]]

        pend_cross = self.cov(self.amp2, self.ls, comp, pend)
        cons_pend_cross = self.cov(self.constraint_amp2, self.constraint_ls, compfull, pend)

        pend_kappa = self.cov(self.amp2, self.ls, pend)
        cons_pend_kappa = self.cov(self.constraint_amp2, self.constraint_ls, pend)

        alpha = spla.cho_solve((obsv_chol, True), vals - self.mean)
        cons_alpha = spla.cho_solve((cons_obsv_chol, True), cos - self.constraint_mean)

        beta = spla.cho_solve((obsv_chol, True), pend_cross)
        cons_beta = spla.cho_solve((cons_obsv_chol, True), cons_pend_cross)

        pend_m = np.dot(pend_cross.T, alpha) + self.mean
        cons_pend_m = np.dot(cons_pend_cross.T, cons_alpha) + self.constraint_mean
        pend_k = pend_kappa - np.dot(pend_cross.T, beta)
        cons_pend_k = cons_pend_kappa - np.dot(cons_pend_cross.T, cons_beta)

        pend_chol = spla.cholesky(pend_k, lower=True)
        cons_pend_chol = spla.cholesky(cons_pend_k, lower=True)

        # Make predictions.
        pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0], self.pending_samples)) + pend_m[:, None]
        cons_pend_fant = np.dot(cons_pend_chol,
                                npr.randn(pend.shape[0], self.pending_samples)) + cons_pend_m[:, None]

        m_pend_fant = np.mean(pend_fant, axis=1)
        m_cons_pend_fant = np.mean(cons_pend_fant, axis=1)
        print("m_pend_fant shape: ", m_pend_fant.shape)
        print("m_cons_pend_fant shape: ", m_cons_pend_fant.shape)

        valid_pend = m_pend_fant[m_cons_pend_fant <= self.budget]

        if len(valid_pend) > 0:
            new_comp = comp.vstack(comp, pend[m_cons_pend_fant <= self.budget, :])
            new_vals = np.concatenate((vals, valid_pend))
        else:
            new_comp = comp
            new_vals = vals

        new_compfull = comp_pend_full
        new_cos = np.concatenate((cos, m_cons_pend_fant))
        return self.compute_constrained_ei_nopend(new_comp, new_vals, new_compfull, new_cos, cand)

    def sample_constraint_hyper_params(self, comp, cos):
        if self.constraint_noiseless:
            self.constraint_noise = 1e-3
            self._sample_constraint_noiseless(comp, cos)
        else:
            self._sample_constraint_noisy(comp, cos)

        self._sample_constraint_ls(comp, cos)
        self.constraint_hyper_samples.append((self.constraint_mean, self.constraint_noise,
                                              self.constraint_amp2, self.constraint_ls))

    def sample_hyper_params(self, comp, values):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, values)
        else:
            self._sample_noisy(comp, values)
        self._sample_ls(comp, values)

        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))

    def _sample_ls(self, comp, values):
        def log_prob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov = (self.amp2 * (self.cov_func(ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                   self.noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), values - self.mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(values - self.mean, solve))
            return lp

        self.ls = util.slice_sample(self.ls, log_prob, compwise=True)

    def _sample_constraint_ls(self, comp, cos):
        def log_prob(ls):
            if np.any(ls < 0) or np.any(ls > self.constraint_max_ls):
                return -np.inf

            cov = (self.constraint_amp2 * (self.cov_func(ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                   self.constraint_noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - self.constraint_mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - self.constraint_mean, solve))
            return lp

        self.constraint_ls = util.slice_sample(self.constraint_ls, log_prob, compwise=True)

    def _sample_noisy(self, comp, values):
        def log_prob(hyper_params):
            mean = hyper_params[0]
            amp2 = hyper_params[1]
            noise = hyper_params[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(values) or mean < np.min(values):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                          noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), values - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(values - mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale / noise) ** 2))
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        new_hyper_params = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), log_prob, compwise=False)
        self.mean = new_hyper_params[0]
        self.amp2 = new_hyper_params[1]
        self.noise = new_hyper_params[2]

    def _sample_constraint_noisy(self, comp, cos):
        def log_prob(hyper_params):
            mean = hyper_params[0]
            amp2 = hyper_params[1]
            noise = hyper_params[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(cos) or mean < np.min(cos):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.constraint_ls, comp, None) +
                           1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.constraint_noise_scale / noise) ** 2))
            lp -= 0.5 * (np.log(amp2) / self.constraint_amp2_scale) ** 2
            return lp

        new_hyper_params = util.slice_sample(np.array([self.constraint_mean, self.constraint_amp2,
                                                       self.constraint_noise]), log_prob, compwise=False)
        self.constraint_mean = new_hyper_params[0]
        self.constraint_amp2 = new_hyper_params[1]
        self.constraint_noise = new_hyper_params[2]

    def _sample_noiseless(self, comp, values):
        def log_prob(hyper_params):
            mean = hyper_params[0]
            amp2 = hyper_params[1]
            noise = 1e-3

            if mean > np.max(values) or mean < np.min(values):
                return -np.inf

            if amp2 < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                          noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), values - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(values - mean, solve)
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2
            return lp

        new_hyper_params = util.slice_sample(np.array([self.mean, self.amp2, self.noise]),
                                             log_prob, compwise=False)
        self.mean = new_hyper_params[0]
        self.amp2 = new_hyper_params[1]
        self.noise = 1e-3

    def _sample_constraint_noiseless(self, comp, cos):
        def log_prob(hyper_params):
            mean = hyper_params[0]
            amp2 = hyper_params[1]
            noise = 1e-3

            if mean > np.max(cos) or mean < np.min(cos):
                return -np.inf

            if amp2 < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.constraint_ls, comp, None) +
                           1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - mean, solve)

            lp -= 0.5 * (np.log(amp2) / self.constraint_amp2_scale) ** 2

            return lp

        new_hyper_params = util.slice_sample(np.array([self.constraint_mean, self.constraint_amp2,
                                                       self.constraint_noise]), log_prob, compwise=False)
        self.constraint_mean = new_hyper_params[0]
        self.constraint_amp2 = new_hyper_params[1]
        self.constraint_noise = 1e-3

    def get_budget(self):
        return self.budget
