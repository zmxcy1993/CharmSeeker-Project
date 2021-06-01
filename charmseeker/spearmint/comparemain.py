import numpy as np
from BO import *
from multiprocessing import Process, Queue


# comparison scheme to Bayesian Optimization (can be Random and sequential schemes)

def random_dispatch(stages, sizes, budget):
    bos = []
    shared_queues = []
    procs = []

    for i in xrange(stages):
        expr_dir = '../examples/inner_bo_' + str(i+1) + '/config.pb'
        method_args = 'noiseless=1'
        bos.append(BO(exp_config=expr_dir, chooser='SequentialChooser', chooser_args=method_args,
                      grid_size=sizes[i], single_round=80))
        shared_queues.append(Queue())
        procs.append(Process(target=bos[i].run_bo, args=(shared_queues[i], )))
        procs[i].start()

    print("Inner Bayesian pids:")
    for stage in xrange(stages):
        print(procs[stage].pid)
        procs[stage].join()

    results = [res.get() for res in shared_queues]
    durations = None
    costs = None

    for stage in xrange(stages):
        single_stage = results[stage]
        # single stage shape: (4, single_round)
        if costs is None:
            costs = np.exp(single_stage[1])
        else:
            costs += np.exp(single_stage[1])

        if durations is None:
            durations = np.exp(single_stage[0])
        else:
            durations += np.exp(single_stage[0])

    durations = np.log(durations)
    legal_values = durations[np.nonzero(costs <= budget)]

    print(" legal values: ", legal_values)

    if len(legal_values) > 0:
        best_value = np.min(legal_values)
        best_ind = np.nonzero(durations == best_value)[0][0]
        print("best_value: ", best_value)
        best_memory = np.array([])
        best_workload = np.array([])
        for i in xrange(stages):
            best_memory = np.append(best_memory, results[i][2, best_ind])
            best_workload = np.append(best_workload, results[i][3, best_ind])
        print(" Current best configuration: best memory ", best_memory.astype(int),
              " best workload: ", best_workload.astype(int))
    else:
        print(" Cannot find a configuration satisfying the constraints.")


if __name__ == "__main__":
    print("enter the main function!")
    grid_sizes = [26890, 25610, 25610]
    random_dispatch(3, grid_sizes, 300)


