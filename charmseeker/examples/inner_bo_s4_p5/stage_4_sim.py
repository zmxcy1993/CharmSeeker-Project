import math

function_name = 'yolo-s4-p5'
durations = {}
costs = {}

with open('../inner_bo_s4_p5/averages') as file:
    for line in file:
        line = line.strip('\n')
        params = line.split(' ')
        idx = f"{params[0]},{params[1]}"
        durations[idx] = float(params[3])
        costs[idx] = float(params[2])


def get_function_value(memory, workload):
    print(f'function name: {function_name}, memory_size {memory}, workload {workload}')
    index = f"{memory},{workload}"
    print(f"Config {memory},{workload} Result. Processing time: {durations[index]} milliseconds,"
          f" monetary cost: {costs[index]} us dollars.")
    return math.log(durations[index] / 1000), math.log(costs[index])


def main(job_id, parameters):
    print(f"Job id {job_id} enter invoke_yolo_4 main function")
    config_memory = parameters['memory_size'][0] * 256
    config_workload = int(math.pow(2, parameters['workload'][0]))
    return get_function_value(config_memory, config_workload)

