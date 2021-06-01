import boto3
import botocore
import base64
import math
import numpy
import json
from multiprocessing.pool import ThreadPool

config = botocore.config.Config(connect_timeout=900, read_timeout=900, retries={'max_attempts': 0})
client = boto3.client('lambda', config=config)
function_name = 'pipe-alpr-p'
time_out = 900
total_workload = 3376


def invoke_function_once(payload, memory_size):
    invoke_res = client.invoke(FunctionName=function_name, Payload=payload, LogType='Tail')
    if invoke_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        log_res = base64.b64decode(invoke_res['LogResult']).split('\n')[2].split('\t')
        duration = log_res[1].split(' ')[1]
        billed_duration = int(math.ceil(float(duration) / 100))
        cost = 0.0000166667 / 1024 / 10 * memory_size * billed_duration + 0.0000002
        return float(duration) / 1000, cost


def get_function_value(memory, workload):
    if memory < 512 or memory > 3008:
        print('Illegal memory size. please reset!')
        return math.log(900)
    config_res = client.update_function_configuration(FunctionName=function_name, Timeout=time_out, MemorySize=memory)
    print('memory_size %d, workload %d' % (memory, workload))

    if config_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        invoke_numbers = 256 / workload
        pool = ThreadPool(invoke_numbers)

        with open("/home/miao/Desktop/python2-venv/spearmint/spearmint/examples/invoke_alpr/profile-names") as myfile:
            for i in range(invoke_numbers):
                res = [next(myfile).strip() for x in xrange(workload)]
                pay_load = '{"ids": ' + json.dumps(res) + '}'
                pool.apply(invoke_function_once, (pay_load, memory))

        results = []
        with open("/home/miao/Desktop/python2-venv/spearmint/spearmint/examples/invoke_alpr/profile-names") as myfile:
            for i in range(invoke_numbers):
                res = [next(myfile).strip() for x in xrange(workload)]
                pay_load = '{"ids": ' + json.dumps(res) + '}'
                print(pay_load)
                results.append(pool.apply_async(invoke_function_once, (pay_load, memory)))

            durations = []
            costs = []
            results = [r.get() for r in results]
            for i in results:
                durations.append(i[0])
                costs.append(i[1])
            mean_cost = numpy.mean(costs) / workload * 3376
            mean_duration = numpy.mean(durations)
            print(mean_duration, mean_cost)
            return math.log(mean_duration), math.log(mean_cost)

    return math.log(900), math.log(0.0000166667 / 1024 * memory * 900 + 0.0000002)


def main(params):
    print("enter invoke_alpr main function")
    memory = params['memory_size'][0] * 64
    workload = int(math.pow(2, params['workload'][0]))
    return get_function_value(memory, workload)
