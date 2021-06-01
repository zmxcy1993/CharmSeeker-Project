import math
from multiprocessing import Process, Queue

function1_name = 'pipe-decoder'
function2_name = 'pipe-yolo'

durations_1 = {}
costs_1 = {}
durations_2 = {}
costs_2 = {}

with open('./averages-1') as file:
    for line in file:
        line = line.strip('\n')
        params = line.split(' ')
        index = f'{params[0]},{params[1]}'
        durations_1[index] = float(params[3])
        costs_1[index] = float(params[2])

with open('./averages-2') as file:
    for line in file:
        line = line.strip('\n')
        params = line.split(' ')
        index = f'{params[0]},{params[1]}'
        durations_2[index] = float(params[3])
        costs_2[index] = float(params[2])


def get_value_1(config_memory_1, config_workload_1, q):
    # print(f'function name: {function1_name}, memory_size {config_memory_1}, workload {config_workload_1}')
    idx = f'{config_memory_1},{config_workload_1}'
    # print(f"Config Result: {idx}, processing time: {durations_1[idx]}, monetary cost: {costs_1[idx]}")
    q.put((durations_1[idx] / 1000.0, costs_1[idx]))


def get_value_2(config_memory_2, config_workload_2, q):
    # print(f'function name: {function2_name}, memory_size {config_memory_2}, workload {config_workload_2}')
    idx = f'{config_memory_2},{config_workload_2}'
    # print(f"Config Result: {idx}, processing time: {durations_2[idx]}, monetary cost: {costs_2[idx]}")
    q.put((durations_2[idx] / 1000.0, costs_2[idx]))


def compute_values(config):
    q1 = Queue()
    proc1 = Process(target=get_value_1, args=(config[0], config[1], q1))
    proc1.start()

    q2 = Queue()
    proc2 = Process(target=get_value_2, args=(config[2], config[3], q2))
    proc2.start()

    proc1.join()
    proc2.join()

    stage1_res = q1.get()
    stage2_res = q2.get()

    duration = stage1_res[0] + stage2_res[0]
    cost = stage1_res[1] + stage2_res[1]

    print(f"log duration: {math.log(duration)}, log costs: {math.log(cost)}")
    return math.log(duration), math.log(cost)


def main(job_id, para):
    print(f"Job id {job_id} enter CPS 2-stage pipeline main function")
    config = [para['memory_1'][0] * 64, int(math.pow(2, para['workload_1'][0])), para['memory_2'][0] * 64,
              int(math.pow(2, para['workload_2'][0]))]
    return compute_values(config)
