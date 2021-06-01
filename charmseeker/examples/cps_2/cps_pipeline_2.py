import boto3
import botocore
import base64
import math
import numpy
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

lambda_config = botocore.config.Config(connect_timeout=900, read_timeout=900, retries={'max_attempts': 0})
client = boto3.client('lambda', config=lambda_config)
function1_name = 'pipe-decoder-p'
function2_name = 'pipe-yolo-p'
time_out = 900


def invoke_function_once(function_name, payload, memory_size):
    invoke_res = client.invoke(FunctionName=function_name, Payload=payload, LogType='Tail')
    if invoke_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        log_res = base64.b64decode(invoke_res['LogResult']).split('\n')[2].split('\t')
        duration = log_res[1].split(' ')[1]
        billed_duration = int(math.ceil(float(duration)/100))
        cost = 0.0000166667/1024/10 * memory_size * billed_duration + 0.0000002
        return float(duration)/1000, cost


def get_value_1(memory, workload, q):
    config_res = client.update_function_configuration(FunctionName=function1_name, Timeout=time_out, MemorySize=memory)
    print('function name: %s, memory_size %d, workload %d' % (function1_name, memory, workload))

    invoke_numbers = 32 / workload
    pool = ThreadPool(invoke_numbers)
    durations = []
    costs = []
    results = []
    res = []
    if config_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        if workload != 1:
            for i in range(invoke_numbers):
                start = i * workload
                end = (i + 1) * workload - 1
                pay_load = '{"start": ' + str(start) + ',"end": ' + str(end) + '}'
                res.append(pool.apply_async(invoke_function_once, (function1_name, pay_load, memory)))
            res = [r.get() for r in res]

        for i in range(invoke_numbers):
            start = i * workload
            end = (i + 1) * workload - 1
            pay_load = '{"start": ' + str(start) + ',"end": ' + str(end) + '}'
            results.append(pool.apply_async(invoke_function_once, (function1_name, pay_load, memory)))

        results = [r.get() for r in results]
        results.extend(res)
        for i in results:
            durations.append(i[0])
            costs.append(i[1])
        mean_cost = numpy.mean(costs) / workload * 390
        mean_duration = numpy.mean(durations)
        print('decoder: ', mean_duration, mean_cost)
        q.put((mean_duration, mean_cost))
    else:
        q.put((90, 0.0000166667/1024 * memory * 900 + 0.0000002))


def get_value_2(memory, workload, q):
    config_res = client.update_function_configuration(FunctionName=function2_name, Timeout=time_out, MemorySize=memory)
    print('function name: %s, memory_size %d, workload %d' % (function2_name, memory, workload))
    invoke_numbers = 128 / workload
    pool = ThreadPool(invoke_numbers)
    durations = []
    costs = []
    results = []
    res = []

    if config_res['ResponseMetadata']['HTTPStatusCode'] == 200:
        if workload != 2:
            for i in range(invoke_numbers):
                start = i * workload + 1
                end = (i + 1) * workload
                pay_load = '{"start": ' + str(start) + ',"end": ' + str(end) + '}'
                res.append(pool.apply_async(invoke_function_once, (function2_name, pay_load, memory)))
            res = [r.get() for r in res]

        for i in range(invoke_numbers):
            start = i * workload + 1
            end = (i + 1) * workload
            pay_load = '{"start": ' + str(start) + ',"end": ' + str(end) + '}'
            results.append(pool.apply_async(invoke_function_once, (function2_name, pay_load, memory)))

        results = [r.get() for r in results]
        results.extend(res)
        for i in results:
            durations.append(i[0])
            costs.append(i[1])
        mean_cost = numpy.mean(costs) / workload * 1950
        mean_duration = numpy.mean(durations)
        print('yolo: ', mean_duration, mean_cost)
        q.put((mean_duration, mean_cost))
    else:
        q.put((90, 0.0000166667 / 1024 * memory * 900 + 0.0000002))


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

    print("log duration: ", math.log(duration), "log costs: ", math.log(cost))
    return math.log(duration), math.log(cost)


def main(job_id, params):
    print(f"Job id {job_id} enter CPS 2-stage pipeline main function")
    config = [params['memory_1'][0] * 64, int(math.pow(2, params['workload_1'][0])), params['memory_2'][0] * 64,
              int(math.pow(2, params['workload_2'][0]))]
    return compute_values(config)
