import optparse
import numpy as np
from pyDOE import lhs
from ConsBO import *
from multiprocessing import Process, Queue

grid_sizes = [512, 512, 1024]


def parse_args():
    parser = optparse.OptionParser(usage="\n\tspearmint [options] <experiment/config.pb>")

    parser.add_option("--max-expt", dest="max_expt",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=20)
    parser.add_option("--pipeline-budget", dest="pipe_budget",
                      type="float", default=0.3771)
    parser.add_option("--pipeline-stages", dest="pipe_stages",
                      type="int", default=2)
    parser.add_option("--initial-cost-steps", dest="initial_cost_steps",
                      type="int", default=6)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)

    return parser.parse_args()


def two_levels_optimization(opts):
    expt_num, budget_min = start_cost_optimization(opts.pipe_stages, opts.max_expt, opts.initial_cost_steps,
                                                   opts.pipe_budget, opts.grid_seed)
    if budget_min.size == 0:
        return
    left_budget = opts.pipe_budget - sum(budget_min)
    np.set_printoptions(precision=5)
    print(f"budget_min: {budget_min}")
    print(f"left_budget: {left_budget:.5f}")

    budget_min = np.array(budget_min)
    left_rounds = opts.max_expt - expt_num
    if left_rounds <= 0:
        return
    outer_loops = int(left_rounds / (opts.max_expt * 0.2))
    if outer_loops < 1:
        outer_loops = 1

    budgets = latin_hypercube_sampling(opts.pipe_stages, budget_min, left_budget, outer_loops)
    rounds = [int(left_rounds / outer_loops) for _ in range(outer_loops)]
    remainder = left_rounds % outer_loops
    if remainder != 0:
        idx = 0
        while remainder > 0:
            rounds[idx] += 1
            remainder -= 1
            idx += 1
    print(f"rounds: {rounds}")

    for i in range(outer_loops):
        budget_vector = budgets[i, :]
        print(f"budget_vector: {budget_vector}")
        constrained_time_optimization(opts.pipe_stages, budget_vector, rounds[i], opts.grid_seed)


def start_cost_optimization(stages, max_expt, initial_cost_steps, pipeline_budget, grid_seed):
    iterate_num = initial_cost_steps
    expt_num = initial_cost_steps
    print(f"max_expt: {max_expt}")

    while True:
        budget_min = np.array([])
        if expt_num > max_expt:
            return max_expt, budget_min

        bos = []
        shared_queues = []
        process = []

        for i in range(stages):
            config_path = f'../inner_bo_s{i + 1}/config.pb'
            method_args = 'cost_optimize=1'
            bos.append(
                ConsBO(exp_config=config_path, chooser='GPEISBOChooser', chooser_args=method_args,
                       grid_size=grid_sizes[i], grid_seed=grid_seed, single_round=iterate_num, cost_optimize=True))
            shared_queues.append(Queue())
            process.append(Process(target=bos[i].run_bo, args=(shared_queues[i],)))
            process[i].start()

        for i in range(stages):
            process[i].join()

        results = [res.get() for res in shared_queues]
        for i in range(stages):
            second_small_cost = results[i]
            budget_min = np.append(budget_min, second_small_cost)

        print(f"inner loop budget_min: {budget_min}")
        if sum(budget_min) <= pipeline_budget:
            return expt_num, budget_min

        iterate_num = 1
        expt_num += iterate_num


def latin_hypercube_sampling(stages, budget_min, lefts, outer_loop):
    lhd = lhs(stages, samples=outer_loop)
    budgets = budget_min + lhd * lefts
    return budgets


def constrained_time_optimization(stages, budget_vector, rounds, grid_seed):
    bos = []
    shared_queues = []
    process = []

    for i in range(stages):
        config_path = f'../inner_bo_s{i + 1}/config.pb'
        method_args = f'budget={budget_vector[i]}'
        bos.append(
            ConsBO(exp_config=config_path, chooser='GPEISBOChooser', chooser_args=method_args,
                   grid_size=grid_sizes[i], grid_seed=grid_seed, single_round=rounds))
        shared_queues.append(Queue())
        process.append(Process(target=bos[i].run_bo, args=(shared_queues[i],)))
        process[i].start()

    for i in range(stages):
        process[i].join()


if __name__ == '__main__':
    options, args = parse_args()
    if options.pipe_stages == 5:
        # Different from 2-stage or 3-stage pipelines, the memory step of 5-stage pipeline is 256MB instead of 64MB
        grid_sizes = [64, 64, 128, 64, 128]
        options.max_expt = 12
    two_levels_optimization(options)

