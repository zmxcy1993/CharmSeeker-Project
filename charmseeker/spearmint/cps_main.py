import optparse
import importlib
import time
import os
import sys
import signal
import math
import json

sys.path.append(os.path.realpath(__file__))

from ExperimentGrid import *
from helpers import *
from runner import job_runner


# --grid_size=7683000 for random and sequential methods
def parse_args():
    parser = optparse.OptionParser(usage="\n\tspearmint [options] <experiment/config.pb>")

    parser.add_option("--max-concurrent", dest="max_concurrent",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=1000)
    parser.add_option("--pipeline-stages", dest="stages",
                      type="int", default=3)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments [SequentialChooser, RandomChooser, "
                           "GPEIOptChooser, GPEIOptChooser, GPEIperSecChooser, GPEIChooser]",
                      type="string", default="RandomChooser")
    parser.add_option("--driver", dest="driver",
                      help="Runtime driver for jobs (local, or sge)",
                      type="string", default="local")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="noiseless=1,budget=1")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=30000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--search-budget", dest="search_budget",
                      help="The maximum searching cost.",
                      type="float", default=14.72461)
    parser.add_option("--run-job", dest="job",
                      help="Run a job in wrapper mode.",
                      type="string", default="")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=6.0)
    parser.add_option("-v", "--verbose", action="store_true",
                      help="Print verbose debug output.")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    return options, args


def main():
    (options, args) = parse_args()

    if options.job:
        job_runner(load_job(options.job))
        exit(0)

    experiment_config = args[0]
    expt_dir = os.path.dirname(os.path.realpath(experiment_config))
    log(f"Using experiment configuration: {experiment_config}")
    log(f"experiment dir: {expt_dir}")

    if not os.path.exists(expt_dir):
        log(f"Cannot find experiment directory {expt_dir}. Aborting.")
        sys.exit(-1)

    check_experiment_dirs(expt_dir)

    # Load up the chooser module.
    module = importlib.import_module('chooser.' + options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    module = importlib.import_module('driver.' + options.driver)
    driver = module.init_driver()

    search_object = SearchCost(expt_dir, options.search_budget, options.stages)

    # Loop until we run out of jobs.
    while attempt_dispatch(experiment_config, expt_dir, chooser, driver, options, search_object):
        time.sleep(options.polling_time)


def attempt_dispatch(expt_config, expt_dir, chooser, driver, options, search_object):
    log("\n" + "-" * 40)
    expt = load_experiment(expt_config)

    expt_grid = ExperimentGrid(expt_dir, expt.variable, options.grid_size, options.grid_seed)
    best_val, best_job = expt_grid.get_constraint_best(chooser.get_budget())
    best_val = math.exp(best_val)

    if best_job >= 0:
        log(f"Current best: {best_val:.5f} (job {best_job})")
    else:
        log("Current best: No results returned yet.")

    # Gets you everything - NaN for unknown values & durations.
    grid, values, costs = expt_grid.get_grid()

    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending = expt_grid.get_pending()
    completes = expt_grid.get_complete()

    n_candidates = candidates.shape[0]
    n_pending = pending.shape[0]
    n_completes = completes.shape[0]
    log(f"{n_candidates} candidates   {n_pending} pending   {n_completes} complete")

    # Verify that pending jobs are actually running, and add them back to the
    # candidate set if they have crashed or gotten lost.
    for job_id in pending:
        proc_id = expt_grid.get_proc_id(job_id)
        if not driver.is_proc_alive(job_id, proc_id):
            log(f"Set job {job_id} back to candidate status.")
            expt_grid.set_candidate(job_id)

    # Track the time series of optimization.
    write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_completes)

    # Print out the best job results
    write_best_job(expt_dir, best_val, best_job, expt_grid)

    if n_completes >= options.max_finished_jobs:
        log(f"Maximum number of finished jobs ({options.max_finished_jobs}) reached. Exiting")
        return False

    if n_candidates == 0:
        log("There are no candidates left. Exiting.")
        return False

    if n_pending >= options.max_concurrent:
        log(f"Maximum number of jobs ({options.max_concurrent}) pending.")
        return True

    else:
        if search_object.finish_flag:
            log("The budget has been depleted. Exiting.")
            return False

        # Ask the chooser to pick the next candidate
        log("Choosing next candidate... ")
        job_id = chooser.next(grid, values, costs, candidates, pending, completes)

        # If the job_id is a tuple, then the chooser picked a new job.
        # We have to add this to our grid
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            job_id = expt_grid.add_to_grid(candidate)

        log(f"selected job {job_id} from the grid.")

        # Convert this back into an interpretable job and add metadata.
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
        write_proposals(job_id, expt_dir, new_list)

        save_job(job)
        pid = driver.submit_job(job)
        if pid is not None:
            log(f"submitted - pid = {pid}")
            expt_grid.set_submitted(job_id, pid)
        else:
            log("Failed to submit job!")
            log("Deleting job file.")
            os.unlink(job_file_for(job))

        keys = []
        for i in range(int(len(new_list) / 2)):
            key = f'{new_list[i * 2]},{new_list[i * 2 + 1]}'
            keys.append(key)

        print(f'search_cost: {search_object.used_cost:.6f}')
        search_object.check_complete(keys)
        if search_object.finish_flag:
            print(f"search_cost: {search_object.used_cost:.6f}")
            budget = f"{math.exp(chooser.get_budget()):.6f}"
            write_search_cost(expt_dir, search_object.used_cost, budget)

    return True


def write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_completes):
    # Append current experiment state to trace file.
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')

    trace_fh.write(f"{time.time()}, {best_val}, {best_job}, "
                   f"{n_candidates}, {n_pending}, {n_completes}\n")
    trace_fh.close()


def write_best_job(expt_dir, best_val, best_job, expt_grid):
    # Write out the best_job_and_result.txt file containing the top results.
    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write(f"Best result: {best_val:.6f}\nJob-id: {best_job}\nParameters: \n")
    print_params = []
    new_list = []
    for param in expt_grid.get_params(best_job):
        print_params.extend(param.int_val)
    print_params = map(int, print_params)

    idx = 0
    for i in print_params:
        if idx % 2 == 0:
            new_list.append(i * 64)
        else:
            new_list.append(int(math.pow(2, i)))
        idx += 1
    best_job_fh.write(f'{new_list}\n')
    best_job_fh.close()


def check_experiment_dirs(expt_dir):
    # Make output and jobs sub directories.
    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)
    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)


def write_proposals(job_id, expt_dir, params):
    # Append current experiment proposal to proposal file.
    proposal_fh = open(os.path.join(expt_dir, 'proposal.txt'), 'a')
    proposal_fh.write(f"job_id {job_id}, {params}\n")
    proposal_fh.close()


def write_search_cost(expt_dir, search_cost, budget):
    proposal_fh = open(os.path.join(expt_dir, f'search-cost-{budget}'), 'a')
    proposal_fh.write(f'{search_cost:.6f}\n')
    proposal_fh.close()


class SearchCost:
    def __init__(self, expt_dir, search_cost, stages):
        self.dic_lists = []
        for i in range(stages):
            file_name = f'averages-{i + 1}'
            fh = open(os.path.join(expt_dir, file_name), 'r')
            self.dic_lists.append(self.read_stage_configs(fh))
        self.search_budget = search_cost
        self.used_cost = 0
        self.searched_configs = []
        self.stages = stages
        self.finish_flag = False
        for _ in range(stages):
            self.searched_configs.append([])

    def check_complete(self, configs):
        for i in range(self.stages):
            if configs[i] not in self.searched_configs[i]:
                self.searched_configs[i].append(configs[i])
                self.used_cost += float(self.dic_lists[i][configs[i]])
        if self.used_cost >= self.search_budget:
            self.finish_flag = True

    def read_stage_configs(self, fh):
        file_dic = {}
        for line in fh:
            params = line.rstrip('\n').split(' ')
            key = f'{params[0]},{params[1]}'
            file_dic[key] = params[2]
        return file_dic


# Cleanup locks and processes on ctl-c
def sigint_handler():
    sys.exit(0)


if __name__ == '__main__':
    print("setting up signal handler...")
    signal.signal(signal.SIGINT, sigint_handler)
    main()

