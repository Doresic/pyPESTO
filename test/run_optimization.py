import pypesto
import pypesto.logging
import petab
import pypesto.petab
import logging

import matplotlib.pyplot as plt

import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

import numpy as np

from pypesto.hierarchical.spline_inner_solver import SplineInnerSolver

run_name = "linear"
starts = 1
spline_ratio = 1 / 2
model_name = "Boehm"
base_folder = "..."
base_folder = "/home/zebo/Documents/GitHub/examples/"


def run_optimization(importer, optimizer, history_name, num_starts, min_gap):
    """Run optimization"""
    pypesto.logging.log_to_console(logging.INFO)

    objective = importer.create_objective(qualitative=True, guess_steadystate=False)

    problem = importer.create_problem(objective)

    option = pypesto.optimize.OptimizeOptions(allow_failed_starts=True)

    engine = pypesto.engine.SingleCoreEngine()

    problem.objective.calculator.inner_solver = SplineInnerSolver(
        options={
            "spline_ratio": spline_ratio,
            "inner_optimizer": 'SLSQP',
            "minimal_difference": True,
        }
    )

    history_options = pypesto.HistoryOptions(
        trace_record=True, storage_file=history_name
    )
    np.random.seed(num_starts)

    result = pypesto.optimize.minimize(
        problem,
        n_starts=num_starts,
        optimizer=optimizer,
        engine=engine,
        options=option,
        history_options=history_options,
    )
    return result


def get_optimizer(optimizer_name):
    """Return pyPESTO optimizer"""
    opt_all = {
        "L-BFGS-B": pypesto.optimize.ScipyOptimizer(
            method="L-BFGS-B",
            options={"disp": True, "ftol": 2.220446049250313e-09, "gtol": 1e-5},
        ),
        "SLSQP": pypesto.optimize.ScipyOptimizer(
            method="SLSQP", options={"disp": True, "ftol": 1e-8}
        ),
        "ipopt": pypesto.optimize.IpoptOptimizer(
            options={"disp": 5, "maxiter": 200, "accept_after_max_steps": 20}
        ),
        "pyswarm": pypesto.optimize.PyswarmOptimizer(options={}),
        "powell": pypesto.optimize.ScipyOptimizer(
            method="Powell", options={"disp": True, "ftol": 1e-8}
        ),
    }
    return opt_all[optimizer_name]


def get_petab_problem(model_name):
    """Return petab problem yaml file path"""

    all_model_paths = {
        "Raf": "Raf_Mitra_NatCom2018OptimalScaling_3CatQual/Raf_Mitra_NatCom2018OptimalScaling_3CatQual.yaml",
        "tanh_Raf": "Raf_Mitra_NatCom2018OptimalScaling_3CatQual_tanh/Raf_Mitra_NatCom2018OptimalScaling_3CatQual.yaml",
        "Boehm": "Boehm_JProteomeRes2014OptimalScaling/Boehm_JProteomeRes2014OptimalScaling.yaml",
        "Benchmark_Boehm": "Boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml",
        "non_linear_Boehm": "Boehm_JProteomeRes2014OptimalScaling_diff_fct_2/Boehm_JProteomeRes2014OptimalScaling.yaml",
        "Raia": "Raia_CancerResearch2011OptimalScaling/Raia_CancerResearch2011OptimalScaling.yaml",
        "Rahman": "Rahman_MBS2016/Rahman_MBS2016.yaml",
        "Elowitz": "Elowitz_Nature2000/Elowitz_Nature2000.yaml",
    }
    petab_problem_yaml_path = base_folder + all_model_paths[model_name]

    return petab_problem_yaml_path


def get_estimated_parameters(model_name):
    all_estimated_parameters = {
        "Raf": [2, 3],
        "tanh_Raf": [2, 3],
        "Boehm": [0, 1, 2, 3, 4, 5],
        "Benchmark_Boehm": [0, 1, 2, 3, 4, 5],
        "non_linear_Boehm": [0, 1, 2, 3, 4, 5],
        "Raia": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "Rahman": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Elowitz": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    }

    return all_estimated_parameters[model_name]


def main():

    petab_problem_yaml_path = get_petab_problem(model_name)
    petab_problem = petab.Problem.from_yaml(petab_problem_yaml_path)

    importer = pypesto.petab.PetabImporter(petab_problem)
    optimizer = get_optimizer("L-BFGS-B")
    results = run_optimization(
        importer,
        optimizer,
        history_name=f"histories/"
        + model_name
        + "_histories/"
        + run_name
        + f"/history_"
        + model_name
        + "_"
        + "_{id}.csv",
        num_starts=starts,
        min_gap=1e-16,
    )

    pypesto.visualize.waterfall(
        [results],
        legends=["Just results"],
        scale_y="log10",
        y_limits=2e-17,
        size=(15, 6),
    )
    plt.savefig(base_folder + "plots/" + model_name + "_" + run_name + "_waterfall.png")

    pypesto.visualize.parameters(
        [results],
        parameter_indices=get_estimated_parameters(model_name),
        size=(15, 12),
        legends=["Numerical spline"],
        balance_alpha=True,
    )
    plt.savefig(base_folder + "plots/" + model_name + "_" + run_name + "_waterfall.png")


if __name__ == "__main__":
    main()
