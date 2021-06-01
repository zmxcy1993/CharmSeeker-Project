import importlib
import time
import os
import sys
import numpy as np
from ExperimentGrid import *
from helpers import *
import math
import json

sys.path.append(os.path.realpath(__file__))


class ConsBO:

    def __init__(self, exp_config, chooser, chooser_args, grid_size, grid_seed=1, cost_optimize=False, single_round=1,
                 driver="local", polling_time=30, max_finished_jobs=70, max_concurrent=1):
        self.exp_config = exp_config
        self.chooser_module = chooser
        self.chooser_args = chooser_args
        self.driver = driver
        self.polling_time = polling_time
        self.grid_size = grid_size
        self.grid_seed = grid_seed
        self.max_finished_jobs = max_finished_jobs
        self.max_concurrent = max_concurrent
        self.current_best = None
        self.single_round = single_round
        self.current_iterate = 0
        self.cost_optimize = cost_optimize
        self.cur_best_cost = 10000

    def write_trace(self, expt_dir, best_val, best_job, n_candidates, n_pending, n_completes, expt_grid):
        # Append current experiment state to trace file.
        trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
        params = []
        for best_params in expt_grid.get_params(best_job):
            params.extend(best_params.int_val)

        trace_fh.write(f"{time.time()},{best_val},{best_job},"
                       f"{n_candidates},{n_pending},{n_completes},{params}\n")
        trace_fh.close()

    def write_best_job(self, expt_dir, best_val, best_job, expt_grid):
        # Write out the best_job_and_result.txt file containing the top results.
        best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
        best_job_fh.write(f"Best result: {best_val}\nJob-id: {best_job}\nParameters: \n")
        for best_params in expt_grid.get_params(best_job):
            best_job_fh.write(str(best_params))
        best_job_fh.close()

    def write_proposals(self, job_id, expt_dir, params):
        # Append current experiment proposal to proposal file.
        proposal_fh = open(os.path.join(expt_dir, 'proposal.txt'), 'a')
        proposal_fh.write(f"job id {job_id} {params}\n")
        proposal_fh.close()

    def check_experiment_dirs(self, expt_dir):
        # Make output and jobs sub directories.
        output_subdir = os.path.join(expt_dir, 'output')
        check_dir(output_subdir)

        job_subdir = os.path.join(expt_dir, 'jobs')
        check_dir(job_subdir)

    def run_bo(self, res_q):
        experiment_config = self.exp_config
        expt_dir = os.path.dirname(os.path.realpath(experiment_config))

        # redirect intermediate outputs (stdout and stderr) to file
        path = os.path.join(expt_dir, 'total_log.txt')
        redirect_output(path)

        log(f"Using experiment configuration: {experiment_config}")
        log(f"experiment dir: {expt_dir}")

        if not os.path.exists(expt_dir):
            log(f"Cannot find experiment directory {expt_dir}. Aborting.")
            sys.exit(-1)
        self.check_experiment_dirs(expt_dir)

        module = importlib.import_module('chooser.' + self.chooser_module)
        chooser = module.init_chooser(expt_dir, self.chooser_args)

        module = importlib.import_module('driver.' + self.driver)
        driver = module.init_driver()

        while self.attempt_dispatch(experiment_config, expt_dir, chooser, driver):
            time.sleep(self.polling_time)

        # pass results to the parent process
        res_q.put(self.current_best)

    def attempt_dispatch(self, expt_config, expt_dir, chooser, driver):
        # Return True for successful dispatch, and return false for exist.
        log("\n" + "-" * 40)
        expt = load_experiment(expt_config)
        expt_grid = ExperimentGrid(expt_dir, expt.variable, self.grid_size, self.grid_seed)
        print(f"current grid size: {self.grid_size}")

        best_val, best_job = expt_grid.get_second_small_cost()
        best_val = np.exp(best_val)
        self.write_best_job(expt_dir, best_val, best_job, expt_grid)
        if best_job >= 0:
            log(f"Inner Bayesian current best without constraints: {best_val:.5f} (job {best_job})")
        else:
            log("Inner Bayesian current best without constraints: No results returned yet.")

        # the data type of candidates, pending, and complete is np.array (num,)
        candidates = expt_grid.get_candidates()
        pending = expt_grid.get_pending()
        completes = expt_grid.get_complete()

        n_candidates = candidates.shape[0]
        n_pending = pending.shape[0]
        n_completes = completes.shape[0]

        log(f"{n_candidates} candidates    {n_pending} pending    {n_completes} complete\n")
        self.write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_completes, expt_grid)

        grid, values, costs = expt_grid.get_grid()

        for job_id in pending:
            proc_id = expt_grid.get_proc_id(job_id)
            if not driver.is_proc_alive(job_id, proc_id):
                log(f"Set job {job_id} back to candidate status.")
                expt_grid.set_candidate(job_id)

        if self.current_iterate >= self.single_round:
            log(f"Maximum number of iterations {self.single_round} reached. Exiting.")

            if self.cost_optimize:
                # Used for cost optimization stage. Covert from log scale to normal scale.
                second_small_cost, index = expt_grid.get_second_small_cost()
                self.current_best = np.exp(second_small_cost)
            else:
                # Used for budget-constrained time optimization stage. cur_budget is log-scaled.
                cur_budget = chooser.get_budget()
                cons_bests, jobs_bests = expt_grid.get_constrained_bests(cur_budget)
                cons_params = expt_grid.get_all_constrained_params(jobs_bests)
                self.current_best = (cons_bests, cons_params)
            return False

        if n_completes >= self.max_finished_jobs:
            log(f"Maximum number of finished jobs {self.max_finished_jobs} reached. Exiting")

            cur_budget = chooser.get_budget()
            cons_bests, jobs_bests = expt_grid.get_constrained_bests(cur_budget)
            cons_params = expt_grid.get_all_constrained_params(jobs_bests)
            self.current_best = (cons_bests, cons_params)
            return False

        if n_candidates == 0:
            log("There are no candidates left. Exiting.")
            return False

        if n_pending >= self.max_concurrent:
            log(f"Maximum number of jobs {self.max_concurrent} pending.")
            return True

        else:
            self.current_iterate += 1
            log("Choosing next candidate... ")
            job_id = chooser.next(expt_grid, grid, values, costs, candidates, completes)

            if isinstance(job_id, tuple):
                (job_id, candidate) = job_id
                job_id = expt_grid.add_to_grid(candidate)

            log(f"selected job {job_id} from the grid.")
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
                
            log(f"The next params: {new_list}")
            self.write_proposals(job_id, expt_dir, new_list)

            save_job(job)
            pid = driver.submit_job(job)
            if pid is not None:
                log(f"submitted - pid = {pid}")
                expt_grid.set_submitted(job_id, pid)
            else:
                log("Failed to submit job!")
                log("Deleting job file.")
                os.unlink(job_file_for(job))
        return True
