#import pdb
#pdb.set_trace()
import amici
import pypesto
import pypesto.logging
import petab
import pypesto.petab
import logging
from datetime import datetime

import matplotlib.pyplot as plt

import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)


import numpy as np
import pandas as pd

import faulthandler
faulthandler.enable()

def run_optimization(importer, optimizer, history_name, num_starts, min_gap):
    """Run optimization"""
    pypesto.logging.log_to_console(logging.INFO)
    
    objective = importer.create_objective(force_compile = True)
    
    problem = importer.create_problem(objective)

    option = pypesto.optimize.OptimizeOptions(allow_failed_starts=True)

    engine = pypesto.engine.SingleCoreEngine()

    print("Zavrsio sam ovo")
    history_options = pypesto.HistoryOptions(trace_record=True, storage_file=history_name)
    np.random.seed(num_starts)
    
    result = pypesto.optimize.minimize(problem,
                                       n_starts=num_starts,
                                       optimizer=optimizer,
                                       engine=engine,
                                       options=option,
                                       history_options=history_options)
    return result

def get_optimizer(optimizer_name):
    """Return pyPESTO optimizer"""
    opt_all = {'L-BFGS-B': pypesto.optimize.ScipyOptimizer(method='L-BFGS-B',
                                                           options={'disp': True, 'ftol': 1e-8, 'gtol': 1e-10}),
               'SLSQP': pypesto.optimize.ScipyOptimizer(method='SLSQP', options={'disp': True, 'ftol': 1e-8}),
               'ipopt': pypesto.optimize.IpoptOptimizer(
                   options={'disp': 5, 'maxiter': 200, 'accept_after_max_steps': 20}),
               'pyswarm': pypesto.optimize.PyswarmOptimizer(options={}),
               'powell': pypesto.optimize.ScipyOptimizer(method='Powell',
                                                         options={'disp': True, 'ftol': 1e-8})}
    return opt_all[optimizer_name]

def main():
    """Napisi opis..."""

    petab_problem = petab.Problem.from_yaml(
    '/home/zebo/Documents/Benchmark-Models-PEtab-master/Benchmark-Models/Boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml')

    # petab_problem = petab.Problem.from_yaml(
    # '/home/zebo/Documents/Benchmark-Models-PEtab-master/Benchmark-Models/Raia_CancerResearch2011/Raia_CancerResearch2011.yaml')
    
    # petab_problem = petab.Problem.from_yaml(
    # '/home/zebo/Documents/Benchmark-Models-PEtab-master/Benchmark-Models/Rahman_MBS2016/Rahman_MBS2016.yaml')

    
    importer = pypesto.petab.PetabImporter(petab_problem)
    optimizer = get_optimizer('L-BFGS-B')
    results = run_optimization(importer,
                               optimizer,
                               history_name =f'histories/Boehm_histories/' + f'new/test/history_Boehm_' + '_{id}.csv', 
                               num_starts=1,
                               min_gap=1e-16)


    pypesto.visualize.waterfall([results],
                                legends=['Just results'],
                                scale_y='log10', 
                                y_limits=2e-17, 
                                size=(15,6))
    plt.savefig("plots/Boehm_test_waterfall.png")

    pypesto.visualize.parameters([results],
                                 parameter_indices = [0,1,2,3,4,5],
                                 size=(15,12), 
                                 legends=['Quantitative'],
                                 balance_alpha=True)
    plt.savefig("plots/Boehm_test_parameters.png")

"""
#running two optimizations to compare waterfall plots
    number_of_starts=20

    petab_problem = petab.Problem.from_yaml(
    '/home/zebo/Documents/GitHub/examples/Raf_Mitra_NatCom2018OptimalScaling_3CatQual/Raf_Mitra_NatCom2018OptimalScaling_3CatQual.yaml')


    importer = pypesto.petab.PetabImporter(petab_problem)
    optimizer = get_optimizer('L-BFGS-B')
    results_Old = run_optimization(importer, 
                                   optimizer, 
                                   history_name =f'histories/Raf_histories/' + f'main_Old_plot/history_Raf_' + '_{id}.csv', 
                                   num_starts=number_of_starts, 
                                   min_gap=1e-16)

    petab_problem = petab.Problem.from_yaml(
    '/home/zebo/Documents/GitHub/examples/Raf_Mitra_NatCom2018OptimalScaling_3CatQual_hard_constraints/Raf_Mitra_NatCom2018OptimalScaling_3CatQual.yaml')

    importer = pypesto.petab.PetabImporter(petab_problem)
    optimizer = get_optimizer('L-BFGS-B')
    results_New = run_optimization(importer, 
                                   optimizer, 
                                   history_name =f'histories/Raf_histories/' + f'main_New_plot/history_Raf_' + '_{id}.csv', 
                                   num_starts=number_of_starts, 
                                   min_gap=1e-16)

    pypesto.visualize.waterfall([results_Old, results_New], 
                                legends=['Monotone categories', 'Hard constraints'],
                                scale_y='log10', 
                                y_limits=2e-17, 
                                size=(15,6))
    plt.savefig("plots/waterfall_hard_cons.png")

    pypesto.visualize.parameters([results_Old, results_New],
                                 parameter_indices = [2,3],
                                 size=(15,6), 
                                 legends=['Monotone categories', 'Hard constraints'],
                                 balance_alpha=True)
    plt.savefig("plots/parameters_hard_cons.png")

"""

if __name__ == "__main__":
    main()