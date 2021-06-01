import os
import tempfile
import pickle
import numpy as np

from spearmint_pb2 import *
from Locker import *
from sobol_lib import *
from helpers import *

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE = 2
COMPLETE_STATE = 3
BROKEN_STATE = -1

EXPERIMENT_GRID_FILE = 'expt-grid.pkl'


class ExperimentGrid:

    @staticmethod
    def job_running(expt_dir, job_id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_running(job_id)

    @staticmethod
    def job_complete(expt_dir, job_id, value, cost, budgets, values, config, duration):
        log(f"setting job {job_id} complete")
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_complete(job_id, value, cost, budgets, values, config, duration)
        log("set...")

    @staticmethod
    def job_broken(expt_dir, job_id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_broken(job_id)

    def __init__(self, expt_dir, variables=None, grid_size=None, grid_seed=1):
        self.expt_dir = expt_dir
        self.jobs_pkl = os.path.join(expt_dir, EXPERIMENT_GRID_FILE)
        self.locker = Locker()

        # Only one process at a time is allowed to have access to the grid.
        self.locker.lock_wait(self.jobs_pkl)

        # Set up the grid for the first time if it doesn't exist.
        if variables is not None and not os.path.exists(self.jobs_pkl):
            self.seed = grid_seed
            self.grid_size = grid_size
            self.vmap = GridMap(variables)

            if grid_size == 1200:
                self.grid, self.grid_size = self._hypercube_grid(self.vmap.card(), grid_size, True)
            else:
                self.grid = self._hypercube_grid(self.vmap.card(), grid_size, False)

            print(f"current grid size: {self.grid_size}")
            self.status = np.zeros(self.grid_size, dtype=int) + CANDIDATE_STATE
            self.proc_ids = np.zeros(self.grid_size, dtype=int)
            self.values = np.zeros(self.grid_size) + np.nan
            self.durations = np.zeros(self.grid_size) + np.nan
            self.complete_params = []

            # This is for inner bayesian
            self.costs = np.zeros(self.grid_size) + np.nan
            self.budgets = np.array([np.inf], dtype=float)

            # This is for outer bayesian. to record the order of jobs
            self.job_ids = np.array([], dtype=int)
            self.best_params = []

            self._save_jobs()

        # Or load in the grid from the pickled file.
        else:
            self._load_jobs()

    def __del__(self):
        self._save_jobs()
        if self.locker.unlock(self.jobs_pkl):
            pass
        else:
            raise Exception("Could not release lock on job grid.\n")

    def get_grid(self):
        return self.grid, self.values, self.costs

    def get_grid_size(self):
        return self.grid_size

    def get_candidates(self):
        return np.nonzero(self.status == CANDIDATE_STATE)[0]

    def get_pending(self):
        return np.nonzero((self.status == SUBMITTED_STATE) | (self.status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(np.isfinite(self.values))[0]

    def get_broken(self):
        return np.nonzero(self.status == BROKEN_STATE)[0]

    def get_params(self, index):
        return self.vmap.get_params(self.grid[index, :])

    # TODO: Miao:add
    def get_param_values(self, candidate, index):
        return self.vmap.get_specific_params(candidate, index)

    # TODO: MIAO add
    def get_all_params(self, candidate):
        return self.vmap.get_all_params(candidate)

    def get_complete_expt_params(self):
        return self.complete_params

    # todo: miao. IBO: get the vector of minimum satisfying a budget and corresponding job ids
    # todo: budget is the budget for the current iteration, float type
    def get_constrained_bests(self, budget, cold_start=False):
        if cold_start and len(self.budgets) == 2:
            self.budgets[1] = budget
        else:
            self.budgets = np.append(self.budgets, budget)

        limits = self.budgets

        costs_finite = self.costs[np.isfinite(self.costs)]
        print(f"self.costs: {np.exp(costs_finite)}")
        print(f"self.budgets: {np.exp(limits)}")
        values_finite = self.values[np.isfinite(self.values)]
        cons_bests = np.zeros([limits.shape[0], 1]) + np.nan
        jobs_bests = np.zeros(limits.shape[0]) + np.nan

        if len(costs_finite) > 0:
            for i in range(limits.shape[0]):
                satisfied = (costs_finite <= limits[i])
                if np.sum(satisfied) != 0:
                    cons_bests[i] = np.min(values_finite[np.nonzero(satisfied)])
                    jobs_bests[i] = np.nonzero(self.values == cons_bests[i])[0][0]

        print(f"cons_bests: {cons_bests}")
        return cons_bests, jobs_bests

    def get_all_constrained_params(self, jobs_bests):
        print("jobs_best: ", jobs_bests, jobs_bests.shape[0])
        params = np.zeros([jobs_bests.shape[0], self.vmap.card()]) + np.nan

        # arrays used as indices must be of integer (or boolean) type
        cons_index = jobs_bests[np.isfinite(jobs_bests)].astype(int)
        if len(cons_index) > 0:
            real_params = self.get_all_params(self.grid[cons_index, :])
            index = np.nonzero(np.isfinite(np.tile(jobs_bests[:, np.newaxis], [1, self.vmap.cardinality])))
            # np.reshape(a,-1) can reshape a=[[1,2],[3,4]] to a = [1,2,3,4]
            params[index] = np.reshape(real_params, -1)
        print("return value:", params)
        return params

    def get_best(self):
        finite = self.values[np.isfinite(self.values)]
        if len(finite) > 0:
            cur_min = np.min(finite)
            index = np.nonzero(self.values == cur_min)[0][0]
            return cur_min, index
        else:
            return np.nan, -1

    def get_best_cost(self):
        cost_finite = self.costs[np.isfinite(self.costs)]
        if len(cost_finite) > 0:
            min_cost = np.min(cost_finite)
            index = np.nonzero(self.costs == min_cost)[0][0]
            return min_cost, index
        else:
            return np.nan, -1

    def get_second_small_cost(self):
        # get the grid index of the configuration with the second smallest cost.
        cost_finite = self.costs[np.isfinite(self.costs)]
        if len(cost_finite) > 1:
            second_small_cost = np.partition(cost_finite, 1)[1]
            index = np.nonzero(self.costs == second_small_cost)[0][0]
            return second_small_cost, index
        else:
            return np.nan, -1

    def get_two_layer_best(self):
        if self.best_params and np.isfinite(self.best_params[0]):
            cur_min = self.best_params[0]
            params = [int(i) for i in self.best_params[1:]]
            index = np.nonzero(self.values == cur_min)[0][0]
            return cur_min, index, params
        else:
            return np.nan, -1, []

    def get_current_best(self, budget):
        complete = self.get_complete()
        candidate = self.get_all_params(self.grid[complete, :])
        valid_complete = complete[np.sum(candidate, axis=1) <= np.log(budget)]
        return np.min(self.values[valid_complete])

    def get_real_complete(self):
        complete = self.get_complete()
        return np.sum(self.values[complete] < 10)

    def get_constraint_best(self, budget):
        costs = self.costs.copy()
        costs[np.isnan(costs)] = 20000.0

        idx = np.logical_and(np.isfinite(self.values), costs <= budget)
        good_vals = np.nonzero(idx)[0]
        if len(good_vals) > 0:
            cur_min = np.min(self.values[good_vals])
            index = np.nonzero(self.values == cur_min)[0][0]
            return cur_min, index
        else:
            return np.nan, -1

    def get_proc_id(self, job_id):
        return self.proc_ids[job_id]

    def add_to_grid(self, candidate):
        # Checks to prevent numerical over/underflow from corrupting the grid
        candidate[candidate > 1.0] = 1.0
        candidate[candidate < 0.0] = 0.0

        self.grid = np.vstack((self.grid, candidate))
        self.status = np.append(self.status, np.zeros(1, dtype=int) + int(CANDIDATE_STATE))

        self.values = np.append(self.values, np.zeros(1) + np.nan)
        self.costs = np.append(self.costs, np.zeros(1) + np.nan)
        self.durations = np.append(self.durations, np.zeros(1) + np.nan)
        self.proc_ids = np.append(self.proc_ids, np.zeros(1, dtype=int))

        self._save_jobs()
        return self.grid.shape[0] - 1

    def set_candidate(self, job_id):
        self.status[job_id] = CANDIDATE_STATE
        self._save_jobs()

    def set_submitted(self, job_id, proc_id):
        self.status[job_id] = SUBMITTED_STATE
        self.proc_ids[job_id] = proc_id
        self._save_jobs()

    def set_running(self, job_id):
        self.status[job_id] = RUNNING_STATE
        self._save_jobs()

    def set_complete(self, job_id, value, cost, budgets, values, config, duration):
        if budgets:
            # indicate this is outer layer bayesian
            if cost:
                # indicate this is a cold start
                self.update_cold_start(job_id, values, budgets, config)
            else:
                self.job_ids = np.append(self.job_ids, job_id)
                print(f"Having finished {len(self.job_ids)} jobs.  Finishing in order: {self.job_ids}")
                self.update_complete(value, budgets, values, config, duration)
                self.best_params = config
        else:
            if cost:
                self.costs[job_id] = cost

            self.values[job_id] = value
            tmp_job = self.grid[job_id, :]
            tmp_job = tmp_job[np.newaxis, :]
            params = self.get_all_params(tmp_job)
            params = list(map(int, params[0].tolist()))
            self.complete_params.append(params)
            print("params: ", params)
            print("add complete: ", self.complete_params)

        self.status[job_id] = COMPLETE_STATE
        self.durations[job_id] = duration
        self._save_jobs()

    def set_broken(self, job_id):
        self.status[job_id] = BROKEN_STATE
        self._save_jobs()

    def _load_jobs(self):
        fh = open(self.jobs_pkl, 'rb')
        jobs = pickle.load(fh)
        fh.close()

        self.vmap = jobs['vmap']
        self.grid = jobs['grid']
        self.status = jobs['status']
        self.values = jobs['values']
        self.durations = jobs['durations']
        self.proc_ids = jobs['proc_ids']

        self.costs = jobs['costs']
        self.budgets = jobs['budgets']
        self.job_ids = jobs['job_ids']
        self.best_params = jobs['best_params']
        self.complete_params = np.reshape(jobs['comp_params'], (-1, self.vmap.cardinality)).tolist()
        self.grid_size = jobs['grid_size']

    def _save_jobs(self):
        # Write everything to a temporary file first.
        fh = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        # Convert the complete parameters into a list
        tmp_list = np.reshape(self.complete_params, (1, -1))[0].tolist()
        pickle.dump({'vmap': self.vmap, 'grid': self.grid, 'status': self.status, 'values': self.values,
                     'durations': self.durations, 'costs': self.costs, 'proc_ids': self.proc_ids,
                     'job_ids': self.job_ids, 'best_params': self.best_params, 'budgets': self.budgets,
                     'comp_params': tmp_list, 'grid_size': self.grid_size}, fh, protocol=-1)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = f'mv {fh.name} {self.jobs_pkl}'
        os.system(cmd)

    def _hypercube_grid(self, dims, size, flag):
        # Generate from a Sobol sequence. Generate random numbers distributed more uniformly in the space.
        sobol_grid = np.transpose(i4_sobol_generate(dims, size, self.seed))

        if flag:
            params = self.get_all_params(sobol_grid)
            new_grid = sobol_grid[np.sum(params, axis=1) <= 0.3631]
            grid_size = new_grid.shape[0]
            return new_grid, grid_size
        else:
            return sobol_grid

    def update_complete(self, value, budgets, values, config, duration):

        # update the values of complete experiments
        self.values[self.job_ids] = values

        if value and budgets:
            new_grid = np.array(self.vmap.convert_to_unit(budgets))

            # prevent the same best value add to the grid multiple times
            if len(np.where((self.grid == new_grid).all(axis=1))[0]) == 0:
                print("---------------------Adding new grid---------------------")
                grid_id = self.add_to_grid(new_grid)
                self.values[grid_id] = value
                self.status[grid_id] = COMPLETE_STATE
                self.durations[grid_id] = duration

        # print out the best configuration
        print("Current best configuration: ", config)

    def update_cold_start(self, job_id, values, budgets, config):
        self.values[job_id] = values[0]
        values = values[1:]

        grids = np.reshape(budgets, (-1, self.grid.shape[1]))
        params = np.reshape(config, (len(values), -1))

        for i in range(grids.shape[0]):
            new_grid = self.vmap.convert_to_unit(grids[i, :])
            print("---------------------Adding new grid---------------------")
            grid_id = self.add_to_grid(new_grid)
            self.values[grid_id] = values[i]
            self.status[grid_id] = COMPLETE_STATE

        cur_best = np.min(values)
        cur_index = np.argmin(values)
        best_param = params[cur_index, :].tolist()
        self.best_params = [cur_best]
        self.best_params.extend(best_param)

    def get_results_by_jobids(self):
        values = self.values[self.job_ids]
        costs = self.costs[self.job_ids]
        params = self.get_all_params(self.grid[self.job_ids])
        return np.vstack((values, costs, params.transpose()))


class GridMap:

    def __init__(self, variables):
        self.variables = []
        self.cardinality = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += variable.size

            if variable.type == Experiment.ParameterSpec.INT:
                self.variables.append({'name': variable.name,
                                       'size': variable.size,
                                       'type': 'int',
                                       'min': int(variable.min),
                                       'max': int(variable.max)})

            elif variable.type == Experiment.ParameterSpec.FLOAT:
                self.variables.append({'name': variable.name,
                                       'size': variable.size,
                                       'type': 'float',
                                       'min': float(variable.min),
                                       'max': float(variable.max)})

            elif variable.type == Experiment.ParameterSpec.ENUM:
                self.variables.append({'name': variable.name,
                                       'size': variable.size,
                                       'type': 'enum',
                                       'options': list(variable.options)})
            else:
                raise Exception("Unknown parameter type.")
        log(f"Optimizing over {self.cardinality} dimensions\n")

    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = []
        index = 0

        for variable in self.variables:
            param = Parameter()
            param.name = variable['name']
            if variable['type'] == 'int':
                for _ in range(variable['size']):
                    param.int_val.append(
                        variable['min'] + self._index_map(u[index], variable['max'] - variable['min'] + 1))
                    index += 1

            elif variable['type'] == 'float':
                for _ in range(variable['size']):
                    param.dbl_val.append(variable['min'] + u[index] * (variable['max'] - variable['min']))
                    index += 1

            elif variable['type'] == 'enum':
                for _ in range(variable['size']):
                    tmp_index = self._index_map(u[index], len(variable['options']))
                    index += 1
                    param.str_val.append(variable['options'][tmp_index])
            else:
                raise Exception("Unknown parameter type.")

            params.append(param)
        return params

    def convert_to_unit(self, u):
        if len(u) != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = np.zeros(len(u))
        index = 0

        for variable in self.variables:
            if variable['type'] == 'int':
                for _ in range(variable['size']):
                    params[index] = (float(u[index]) - variable['min']) / (variable['max'] - variable['min'])
                index += 1
            elif variable['type'] == 'float':
                for _ in range(variable['size']):
                    params[index] = (float(u[index]) - variable['min']) / (variable['max']-variable['min'])
                index += 1
            else:
                raise Exception("Unknown parameter type.")

        print("convert_to_unit: ", params)
        return params

    # TODO: Miao get multiple parameters specified by index.
    def get_specific_params(self, candidate, index):
        idx = 0
        variable_min = 0
        variable_max = 0
        variable_type = 'int'
        variable_option = []

        for variable in self.variables:
            idx += variable['size']
            if idx > index:
                variable_min = variable['min']
                variable_max = variable['max']
                variable_type = variable['type']
                if variable_type == 'enum':
                    variable_option = variable['options']
                break

        if variable_type == 'int':
            param = np.int_(variable_min + self._index_map_array(candidate[:, index], variable_max - variable_min + 1))

        elif variable_type == 'float':
            param = variable_min + candidate[:, index] * (variable_max - variable_min)

        elif variable_type == 'enum':
            idx = self._index_map_array(candidate[:, index], len(variable_option))
            param = variable_option[idx]

        else:
            raise Exception("Unknown parameter type.")

        return param

    # get all parameters of multiple candidates.
    def get_all_params(self, candidate):
        params = np.zeros(candidate.shape)
        for ind in range(candidate.shape[1]):
            res = self.get_specific_params(candidate, ind)
            params[:, ind] = res
        return params

    def card(self):
        return self.cardinality

    def _index_map(self, u, items):
        u = np.max((u, 0.0))
        u = np.min((u, 1.0))
        return int(np.floor((1 - np.finfo(float).eps) * u * float(items)))

    # TODO: MIAO: add
    def _index_map_array(self, arr, items):
        arr[np.nonzero(arr > 1.0)] = 1.0
        arr[np.nonzero(arr < 0.0)] = 0.0
        return np.int_(np.floor((1 - np.finfo(float).eps) * arr * float(items)))
