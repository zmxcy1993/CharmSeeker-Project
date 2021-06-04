import boto3
import botocore
import base64
import math
import numpy
from multiprocessing.pool import ThreadPool

config = botocore.config.Config(connect_timeout=900, read_timeout=900, retries={'max_attempts': 0})
client = boto3.client('lambda', config=config)
function_name = 'pipe-decoder-p'
time_out = 900
total_workload = 390


def invoke_function_once(payload, memory_size):
    invoke_res = client.invoke(FunctionName=function_name, Payload=payload, LogType='Tail')
    if invoke_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        log_res = base64.b64decode(invoke_res['LogResult']).split('\n')[2].split('\t')
        duration = float(log_res[1].split(' ')[1])
        billed_duration = int(math.ceil(duration))
        cost = 0.0000166667 / 1024 / 1000 * memory_size * billed_duration + 0.0000002
        return duration / 1000, cost


def get_function_value(memory, workload):
    config_res = client.update_function_configuration(FunctionName=function_name, Timeout=time_out, MemorySize=memory)
    print(f'Update function {function_name} Configuration: memory_size {memory}, workload {workload}')

    invoke_numbers = math.ceil(total_workload / workload)
    pool = ThreadPool(invoke_numbers)
    durations = []
    costs = []
    results = []

    if config_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        for i in range(invoke_numbers):
            start = i * workload
            end = (i + 1) * workload - 1
            pay_load = f'{"start": {start},"end": {end}}'
            results.append(pool.apply_async(invoke_function_once, (pay_load, memory)))

        results = [r.get() for r in results]
        for i in results:
            durations.append(i[0])
            costs.append(i[1])
        total_cost = numpy.sum(costs)
        max_duration = numpy.max(durations)
        return math.log(max_duration), math.log(total_cost)

    return math.log(time_out), math.log((0.0000166667 / 1024 * memory * time_out + 0.0000002) * invoke_numbers)


def main(job_id, parameters):
    print(f"Job id {job_id} enter invoke_decoder main function")
    config_memory = parameters['memory_size'][0] * 256
    config_workload = int(math.pow(2, parameters['workload'][0]))
    return get_function_value(config_memory, config_workload)
