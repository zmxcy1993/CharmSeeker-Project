##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os
import gp
import util
import tempfile
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import cPickle

from Locker  import *
from helpers import *


def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    if 'budget' in args.keys():
        args['budget'] = float(args['budget'])
    return GPEIIBOChooser(expt_dir, **args)

"""
Chooser module for Inner Bayesian Optimization layer
"""
class GPEIIBOChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10, total_workload=1950,
                 pending_samples=100, noiseless=False, budget=0.16, cold_start=False):
        self.cov_func        = getattr(gp, covar)
        self.locker          = Locker()
        self.state_pkl       = os.path.join(expt_dir, self.__module__ + ".pkl")

        self.mcmc_iters      = int(mcmc_iters)
        self.pending_samples = pending_samples
        self.D               = -1
        self.hyper_iters     = 1
        self.noiseless       = bool(int(noiseless))

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 2    # top-hat prior on length scales
        self.budget      = budget
        self.cold_start  = bool(int(cold_start))
        self.total_workload = int(total_workload)
        self.completes = [line.rstrip('\n') for line in open(expt_dir + '/complete_proposal')]
        print("total_workload: ", self.total_workload)

    def __del__(self):
        self.locker.lock_wait(self.state_pkl)

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({'dims'   : self.D,
                       'ls'     : self.ls,
                       'amp2'   : self.amp2,
                       'noise'  : self.noise,
                       'mean'   : self.mean},
                     fh)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.state_pkl)
        os.system(cmd)
        # TODO: Should check system-dependent return status.
        self.locker.unlock(self.state_pkl)

    def _real_init(self, dims, values):
        self.locker.lock_wait(self.state_pkl)

        if os.path.exists(self.state_pkl):
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D     = state['dims']
            self.ls    = state['ls']
            self.amp2  = state['amp2']
            self.noise = state['noise']
            self.mean  = state['mean']
        else:

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values)+1e-4

            # Initial observation noise.
            self.noise = 1e-3

            # Initial mean.
            self.mean = np.mean(values)

        self.locker.unlock(self.state_pkl)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None) + 1e-6 * np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

    def next(self, expt_grid, grid, values, candidates, pending, complete):
        n_comp = complete.shape[0]
        if n_comp < 7:
            propose = self.completes[n_comp].split(',')
            propose = [int(i) for i in propose]
            new_grid = expt_grid.vmap.convert_to_unit(propose)
            return 0, new_grid

        if self.D == -1:
            self._real_init(grid.shape[1], values[complete])

        comp = grid[complete, :]
        cand = grid[candidates, :]
        pend = grid[pending, :]
        vals = values[complete]
        complete_list = expt_grid.get_complete_expt_params()
        print("complete list:", complete_list)

        if self.mcmc_iters > 0:
            overall_ei = np.zeros((cand.shape[0], self.mcmc_iters))
            for mcmc_iter in range(self.mcmc_iters):
                self.sample_hypers(comp, vals)
                log("mean: %f  amp: %f  noise: %f  min_ls: %f  max_ls: %f" % (
                    self.mean, np.sqrt(self.amp2), self.noise, np.min(self.ls), np.max(self.ls)))
                overall_ei[:, mcmc_iter] = self.compute_ei(expt_grid, comp, pend, cand, vals)

            while True:
                best_cand = np.argmax(np.mean(overall_ei, axis=1))
                tmp_idx = int(candidates[best_cand])
                params = []
                for param in expt_grid.get_params(tmp_idx):
                    params.extend(param.int_val)
                params = map(int, params)
                print('params: ', params)
                if params not in complete_list:
                    return tmp_idx
                overall_ei = np.delete(overall_ei, best_cand, 0)
                candidates = np.delete(candidates, best_cand, 0)
                print("delete!!!!!")
        else:
            try:
                self.optimize_hypers(comp, vals)
            except:
                self.ls = np.ones(self.D)
                self.amp2 = np.std(vals)
                self.noise = 1e-3

            log("mean: %f  amp: %f  noise: %f  min_ls: %f  max_ls: %f" % (self.mean,
                                                                          np.sqrt(self.amp2), self.noise,
                                                                          np.min(self.ls), np.max(self.ls)))

            ei = self.compute_ei(comp, pend, cand, vals)
            best_cand = np.argmax(ei)
            return int(candidates[best_cand])

    def compute_ei(self, expt_grid, comp, pend, cand, vals):
        memory_size = 64 * expt_grid.get_param_values(cand, 0)
        workload = np.power(2, expt_grid.get_param_values(cand, 1))
        limit = np.log(((self.budget / self.total_workload * workload  - 0.0000002)/(
                0.0000166667/10240 * memory_size)).astype(int))-np.log(10)

        if pend.shape[0] == 0:
            comp_cov   = self.cov(comp)
            cand_cross = self.cov(comp, cand)
            obsv_cov  = comp_cov + self.noise * np.eye(comp.shape[0])
            obsv_chol = spla.cholesky(obsv_cov, lower=True)
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2 * (1 + 1e-6) - np.sum(beta**2, axis=0)
            func_s = np.sqrt(func_v)

            if self.cold_start:
                best = np.min(vals)
                u = (best - func_m) / func_s
                ncdf = sps.norm.cdf(u)
                npdf = sps.norm.pdf(u)
                ei = func_s * (u * ncdf + npdf)
                return ei
            else:
                weight = sps.norm.cdf(limit, loc=func_m, scale=func_s)
                cur_min, idx = expt_grid.get_constraint_best(np.log(self.budget))
                print("cur_min:", cur_min)

                if cur_min is not np.nan:
                    best = cur_min
                    u = (best - func_m) / func_s
                    ncdf = sps.norm.cdf(u)
                    npdf = sps.norm.pdf(u)
                    ei = func_s * (u * ncdf + npdf)
                else:
                    print("best value that satisfies current budget is nan.")
                    ei = np.ones(weight.shape[0])

                return ei * weight
        else:
            # If there are pending experiments, fantasize their outcomes.
            # Create a composite vector of complete and pending.
            comp_pend = np.concatenate((comp, pend))

            # Compute the covariance and Cholesky decomposition.
            comp_pend_cov  = self.cov(comp_pend) + self.noise*np.eye(comp_pend.shape[0])
            comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)

            # Compute submatrices.
            pend_cross = self.cov(comp, pend)
            pend_kappa = self.cov(pend)

            # Use the sub-Cholesky.
            obsv_chol = comp_pend_chol[:comp.shape[0], :comp.shape[0]]

            # Solve the linear systems.
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.cho_solve((obsv_chol, True), pend_cross)

            # Finding predictive means and variances.
            pend_m = np.dot(pend_cross.T, alpha) + self.mean
            pend_K = pend_kappa - np.dot(pend_cross.T, beta)

            # Take the Cholesky of the predictive covariance.
            pend_chol = spla.cholesky(pend_K, lower=True)

            # Make predictions.
            pend_fant = (np.dot(pend_chol, npr.randn(pend.shape[0], self.pending_samples)) + pend_m[:, None])

            # Include the fantasies.
            fant_vals = np.concatenate((np.tile(vals[:, np.newaxis], (1, self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(comp_pend, cand)

            # Solve the linear systems.
            alpha  = spla.cho_solve((comp_pend_chol, True), fant_vals - self.mean)
            beta   = spla.solve_triangular(comp_pend_chol, cand_cross, lower=True)

            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)
            func_s = np.sqrt(func_v[:, np.newaxis])

            weight = sps.norm.cdf(np.tile(limit[:, np.newaxis], (1, self.pending_samples)),
                                  loc=func_m, scale=np.tile(func_s, (1, self.pending_samples)))

            # Expected improvement
            u      = (bests[np.newaxis, :] - func_m) / func_s
            ncdf   = sps.norm.cdf(u)
            npdf   = sps.norm.pdf(u)
            ei     = func_s*(u*ncdf + npdf)

            # return np.mean(ei, axis=1)
            ei = ei * weight
            return np.mean(ei, axis=1)

    def sample_hypers(self, comp, vals):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, vals)
        else:
            self._sample_noisy(comp, vals)
        self._sample_ls(comp, vals)

    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov   = self.amp2 * (self.cov_func(ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + self.noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-self.mean, solve)
            return lp

        self.ls = util.slice_sample(self.ls, logprob, compwise=True)

    def _sample_noisy(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov   = amp2 * (self.cov_func(self.ls, comp, None) +
                            1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale/noise)**2))

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]),
                                   logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = 1e-3

            if amp2 < 0:
                return -np.inf

            cov   = amp2 * (self.cov_func(self.ls, comp, None) +
                            1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), logprob,
                                   compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = 1e-3

    def optimize_hypers(self, comp, vals):
        mygp = gp.GP(self.cov_func.__name__)
        mygp.real_init(comp.shape[1], vals)
        mygp.optimize_hypers(comp, vals)
        self.mean = mygp.mean
        self.ls = mygp.ls
        self.amp2 = mygp.amp2
        self.noise = mygp.noise
        return

    def get_budget(self):
        return np.log(self.budget)
