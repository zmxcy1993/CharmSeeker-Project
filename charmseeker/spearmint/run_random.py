from ExperimentGrid  import *
from helpers         import *
from BO import *
from multiprocessing import Process, Queue

bos = []
shared_queues = []
procs = []
grid_sizes = [168, 210, 240]
total_workload = [390, 1950, 3376]
stages = 2

for i in range(stages):
    expr_dir = '../examples/inner_bo_' + str(i + 1) + '_random/config.pb'
    method_args = 'total_workload=' + str(total_workload[i])
    bos.append(
        BO(exp_config=expr_dir, chooser='RandomChooser', chooser_args=method_args, grid_size=grid_sizes[i]))
    shared_queues.append(Queue())
    procs.append(Process(target=bos[i].run_bo, args=(shared_queues[i],)))
    procs[i].start()

for stage in range(stages):
    print(procs[stage].pid)
    procs[stage].join()


