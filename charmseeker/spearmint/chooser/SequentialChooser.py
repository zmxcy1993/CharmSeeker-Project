##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
# 
# This code is written for research and educational purposes only to 
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import util

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    if 'budget' in args.keys():
        args['budget'] = float(args['budget'])
    return SequentialChooser()

class SequentialChooser:

    def __init__(self, budget=300, noiseless=1):
        self.budget = np.log(budget)
        self.noiseless = noiseless

    def next(self, grid, values, costs,  candidates, pending, complete):
        vals = values[complete]
        cos = costs[complete]
        print("current costs: ", cos)
        print("self.budget: ", self.budget)

        idx = np.logical_and(cos <= self.budget, np.isfinite(vals))
        goodvals = np.nonzero(idx)[0]
        badvals = np.nonzero(np.logical_not(idx))[0]

        print('Found %d constraint violating jobs' % (badvals.shape[0]))
        print('Received %d valid results' % (goodvals.shape[0]))
        return int(candidates[0])

    def get_budget(self):
        return self.budget
