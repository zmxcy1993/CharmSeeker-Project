import boto3
import botocore
import base64
import math
import numpy
import json
from multiprocessing.pool import ThreadPool

config = botocore.config.Config(connect_timeout=900, read_timeout=900, retries={'max_attempts': 0})
client = boto3.client('lambda', config=config)
function_name = 'pipe-alpr'
time_out = 900
total_workload = 3376

file_names = []
with open('../inner_bo_s3_p5/profile-names') as file:
    for line in file:
        line = line.strip('\n')
        file_names.append(line)


def invoke_function_once(payload, memory_size):
    invoke_res = client.invoke(FunctionName=function_name, Payload=payload, LogType='Tail')
    if invoke_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        log_res = base64.b64decode(invoke_res['LogResult']).split('\n')[2].split('\t')
        duration = float(log_res[1].split(' ')[1])
        billed_duration = int(math.ceil(duration))
        cost = 0.0000166667 / 1024 / 1000 * memory_size * billed_duration + 0.0000002
        return duration / 1000, cost


def get_function_value(memory, workload):
    if memory < 512 or memory > 3008:
        print('Illegal memory size. please reset!')
        return math.log(time_out)
    config_res = client.update_function_configuration(FunctionName=function_name, Timeout=time_out, MemorySize=memory)
    print(f'memory_size {memory}, workload {workload}')

    invoke_numbers = total_workload / workload
    pool = ThreadPool(invoke_numbers)
    results = []

    if config_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        for i in range(invoke_numbers):
            start_id = i * workload
            end_id = (i + 1) * workload
            res = file_names[start_id:end_id]
            pay_load = f'{"ids": {json.dumps(res)}}'
            results.append(pool.apply_async(invoke_function_once, (pay_load, memory)))

        durations = []
        costs = []
        results = [r.get() for r in results]
        for i in results:
            durations.append(i[0])
            costs.append(i[1])
        total_cost = numpy.sum(costs)
        max_duration = numpy.max(durations)
        return math.log(max_duration), math.log(total_cost)

    return math.log(time_out), math.log((0.0000166667 / 1024 * memory * time_out + 0.0000002) * invoke_numbers)


def main(job_id, parameters):
    print(f"Job id {job_id} enter invoke_ALPR main function")
    config_memory = parameters['memory_size'][0] * 256
    config_workload = int(math.pow(2, parameters['workload'][0]))
    return get_function_value(config_memory, config_workload)
