import sys
import os
import math
import traceback
import numpy as np

from spearmint_pb2 import *
from ExperimentGrid import *
from helpers import *

# System dependent modules
DEFAULT_MODULES = ['packages/epd/7.1-2', 'packages/matlab/r2011b', 'mpi/openmpi/1.2.8/intel',
                   'libraries/mkl/10.0', 'packages/cuda/4.0']

MCR_LOCATION = "/home/matlab/v715"


def job_runner(job):
    # This function runs in a new process.  Now we are going to do a little bookkeeping
    # and then spin off the actual job that does whatever it is we're trying to achieve.

    redirect_output(job_output_file(job))
    log(f"Running in wrapper mode for job id: {job.id}")

    ExperimentGrid.job_running(job.expt_dir, job.id)

    # Update metadata and save the job file, which will be read by the job wrappers.
    job.start_t = int(time.time())
    job.status = 'running'
    save_job(job)

    success = False
    start_time = time.time()

    try:
        if job.language == PYTHON:
            run_python_job(job)
        else:
            raise Exception("That function type has not been implemented.")
        success = True
    except:
        log("-" * 40)
        log("Problem running the job:")
        log(sys.exc_info())
        log(traceback.print_exc(limit=1000))
        log("-" * 40)

    duration = time.time() - start_time

    # The job output is written back to the job file, so we read it back in to get the results.
    job_file = job_file_for(job)
    job = load_job(job_file)
    log("Job file reloaded.")

    if not job.HasField("value"):
        log("Could not find value in output file.")
        success = False

    job.end_t = int(time.time())
    job.duration = duration

    if success:
        log(f"Completed successfully in {duration:.2f} seconds. "
            f" Configuration running duration: {math.exp(job.value):.5f} seconds.")

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id, job.value, job.cost, job.budgets,
                                    job.values, job.config, duration)
        job.status = 'complete'
    else:
        log(f"Job failed in {duration:.2f} seconds.")

        # Update the experiment status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
        job.status = 'broken'

    save_job(job)


# regarding the python path, experiment directory, etc...
def run_python_job(job):
    # Run a python function
    log("Running python job.\n")

    # Add experiment directory to the system path.
    sys.path.append(os.path.realpath(job.expt_dir))

    # Convert the PB object into useful parameters.
    params = {}
    for param in job.param:
        dbl_values = param.dbl_val
        int_values = param.int_val
        str_values = param.str_val

        if len(dbl_values) > 0:
            params[param.name] = np.array(dbl_values)
        elif len(int_values) > 0:
            params[param.name] = np.array(int_values, dtype=int)
        elif len(str_values) > 0:
            params[param.name] = str_values
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    module = __import__(job.name)
    result = module.main(job.id, params)

    if not isinstance(result, tuple):
        job.value = result
        log(f"Got result: {result:.5f} \n")
    else:
        if len(result) == 2:
            # indicate this is inner bayesian
            job.value = result[0]
            job.cost = result[1]
        elif len(result) == 3:
            # indicate this is a outer layer cold start
            job.value = -1
            job.values.extend(result[0])
            job.budgets.extend(result[1])
            job.config.extend(result[2])
            job.cost = 2000.0
        elif len(result) == 4:
            # indicate this is outer bayesian
            job.value = result[0]
            job.budgets.extend(result[1])
            job.values.extend(result[2])
            job.config.extend(result[3])
        else:
            job.value = result
        log(f"Got job result {result[0]:.5f} \n")

    save_job(job)
