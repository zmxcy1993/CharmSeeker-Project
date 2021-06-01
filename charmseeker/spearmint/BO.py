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
import importlib
import time
import os
import sys
import numpy as np
from ExperimentGrid  import *
from helpers         import *
import math


try: import simplejson as json
except ImportError: import json

sys.path.append(os.path.realpath(__file__))

class BO:

    def __init__(self, exp_config, chooser, chooser_args, grid_size, cold_start=False, single_round=1,
                 driver="local", polling_time=30, max_finished_jobs=70,
                 max_concurrent=1):
        self.exp_config = exp_config
        self.chooser_module = chooser
        self.chooser_args = chooser_args
        self.driver = driver
        self.polling_time = polling_time
        self.grid_size = grid_size
        self.grid_seed = 1
        self.max_finished_jobs = max_finished_jobs
        self.max_concurrent = max_concurrent
        self.current_best = None
        self.single_round = single_round
        self.current_iterate = 0
        self.cold_start = cold_start
        self.cur_best_cost = 10000

    def write_trace(self, expt_dir, best_val, best_job,
                    n_candidates, n_pending, n_complete, expt_grid):
        '''Append current experiment state to trace file.'''
        trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
        params = []
        for best_params in expt_grid.get_params(best_job):
            params.extend(best_params.int_val)

        trace_fh.write("%d,%f,%d,%d,%d,%d, %s\n"
                       % (time.time(), best_val, best_job,
                          n_candidates, n_pending, n_complete, str(params)))
        trace_fh.close()

    def write_best_job(self, expt_dir, best_val, best_job, expt_grid):
        '''Write out the best_job_and_result.txt file containing the top results.'''

        best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
        best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" %
                          (best_val, best_job))
        for best_params in expt_grid.get_params(best_job):
            best_job_fh.write(str(best_params))
        best_job_fh.close()

    def write_proposals(self, job_id, expt_dir, params):
        '''Append current experiment proposal to proposal file.'''
        proposal_fh = open(os.path.join(expt_dir, 'proposal.txt'), 'a')
        proposal_fh.write("job id %s %s\n" % (str(job_id), str(params)))
        proposal_fh.close()

    def check_experiment_dirs(self, expt_dir):
        '''Make output and jobs sub directories.'''
        output_subdir = os.path.join(expt_dir, 'output')
        check_dir(output_subdir)

        job_subdir = os.path.join(expt_dir, 'jobs')
        check_dir(job_subdir)

    def run_bo(self, res_q):
        experiment_config = self.exp_config
        expt_dir = os.path.dirname(os.path.realpath(experiment_config))

        # redirect intermediate outputs to file
        path = os.path.join(expt_dir, 'total_log.txt')
        redirect_output(path)

        log("Using experiment configuration: " + experiment_config)
        log("experiment dir: " + expt_dir)

        if not os.path.exists(expt_dir):
            log("Cannot find experiment directory '%s'. Aborting." % expt_dir)
            sys.exit(-1)
        self.check_experiment_dirs(expt_dir)

        module = importlib.import_module('chooser.' + self.chooser_module)
        chooser = module.init(expt_dir, self.chooser_args)

        module = importlib.import_module('driver.' + self.driver)
        driver = module.init()

        while self.attempt_dispatch(experiment_config, expt_dir, chooser, driver):
            time.sleep(self.polling_time)

        res_q.put(self.current_best)

        return self.current_best

    def attempt_dispatch(self, expt_config, expt_dir, chooser, driver):
        log("\n" + "-" * 40)
        expt = load_experiment(expt_config)
        expt_grid = ExperimentGrid(expt_dir, expt.variable, self.grid_size, self.grid_seed)
        print("current grid size: ", self.grid_size)

        best_val, best_job = expt_grid.get_best()
        if best_job >= 0:
            log("Inner Bayesian current best without constraints: %f (job %d)" % (best_val, best_job))
        else:
            log("Inner Bayesian current best without constraints: No results returned yet.")

        candidates = expt_grid.get_candidates()
        pending = expt_grid.get_pending()
        complete = expt_grid.get_complete()

        n_candidates = candidates.shape[0]
        n_pending = pending.shape[0]
        n_complete = complete.shape[0]

        log("%d candidates   %d pending   %d complete" % (n_candidates, n_pending, n_complete))
        self.write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_complete, expt_grid)
        self.write_best_job(expt_dir, best_val, best_job, expt_grid)

        grid, values, durations = expt_grid.get_grid()

        for job_id in pending:
            proc_id = expt_grid.get_proc_id(job_id)
            if not driver.is_proc_alive(job_id, proc_id):
                log("Set job %d back to pending status." % job_id)
                expt_grid.set_candidate(job_id)

        if self.current_iterate >= self.single_round:
            log("Maximum number of iterations (%d) reached. Exiting." % self.single_round)

            if self.cold_start:
                costs, values, indices = expt_grid.get_best_cost()
                params = np.zeros((costs.shape[0], grid.shape[1]))
                idx = 0
                for job_id in indices:
                    item = []
                    for param in expt_grid.get_params(job_id):
                        item.extend(param.int_val)
                    params[idx, :] = item
                    idx += 1

                self.current_best = (costs, values, params)
            else:
                cur_budget = chooser.get_budget()
                cons_bests, jobs_bests = expt_grid.get_constrained_bests(cur_budget)
                cons_params = expt_grid.get_all_constrained_params(jobs_bests)
                self.current_best = (cons_bests, cons_params)
            return False

        if n_complete >= self.max_finished_jobs:
            log("Maximum number of finished jobs (%d) reached. Exiting" % self.max_finished_jobs)

            cons_bests, jobs_bests = expt_grid.get_constrained_best(chooser.get_budget())
            cons_params = expt_grid.get_all_constrained_params(jobs_bests)
            self.current_best = (cons_bests, cons_params)
            return False

        if n_candidates == 0:
            log("There are no candidates left.  Exiting.")
            return False

        if n_pending >= self.max_concurrent:
            log("Maximum number of jobs (%d) pending." % self.max_concurrent)
            return True

        else:
            self.current_iterate += 1
            log("Choosing next candidate... ")
            job_id = chooser.next(expt_grid, grid, values, candidates, pending, complete)

            if isinstance(job_id, tuple):
                (job_id, candidate) = job_id
                job_id = expt_grid.add_to_grid(candidate)

            log("selected job %d from the grid." % job_id)
            job = Job()
            job.id = job_id
            job.expt_dir = expt_dir
            job.name = expt.name
            job.language = expt.language
            job.status = 'submitted'
            job.submit_t = int(time.time())
            job.param.extend(expt_grid.get_params(job_id))

            print_params = []
            new_list = []
            for param in expt_grid.get_params(job_id):
                print_params.extend(param.int_val)
            print_params = map(int, print_params)

            idx = 0
            for i in print_params:
                if idx % 2 == 0:
                    new_list.append(i * 64)
                else:
                    new_list.append(int(math.pow(2, i)))
                idx += 1
                
            log("The next params: ", new_list)
            self.write_proposals(job_id, expt_dir, new_list)

            save_job(job)
            pid = driver.submit_job(job)
            if pid is not None:
                log("submitted - pid = %d" % pid)
                expt_grid.set_submitted(job_id, pid)
            else:
                log("Failed to submit job!")
                log("Deleting job file.")
                os.unlink(job_file_for(job))

        return True
