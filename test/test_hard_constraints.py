#!/usr/bin/env python3
import pypesto
import pypesto.petab
import pypesto.optimize
import petab
import pypesto.logging
import logging
import argparse
from colorama import init as init_colorama
import numpy as np

from pypesto.hierarchical.optimal_scaling_solver import OptimalScalingInnerSolver
from pypesto.hierarchical.parameter import InnerParameter
from pypesto.hierarchical.problem import InnerProblem

def main():
    """NAPISI OPIS"""
    init_colorama(autoreset=True)

    #petab_problem = petab.Problem.from_yaml(f'{args.petab_dir}/{args.model_name}.yaml')
    petab_problem = petab.Problem.from_yaml('/home/zebo/Documents/GitHub/examples/Boehm_JProteomeRes2014OptimalScaling_HardConstraints/Boehm_JProteomeRes2014OptimalScaling_HardConstraints.yaml')

    importer= pypesto.petab.PetabImporter(petab_problem)
    model = importer.create_model()

    timepoints=[0,2.5,5,10,15,20,30,40,50,60,80,100,120,160,200,240]
    model.setTimepoints(timepoints)

    solver= model.getSolver()
    rdata = amici.runAmiciSimulation(model, solver)

    edatas_zebo = amici.ExpData(rdata,1.0, 0)
    edatas_list = [edatas_zebo]

    prob= InnerProblem()
    prob = prob.from_petab_amici(petab_problem, model, edatas_list)



if __name__ == "__main__":
    main()