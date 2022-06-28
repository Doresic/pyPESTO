import pypesto
import pypesto.logging
import pypesto.petab
import petab
import amici

import numpy as np
import pandas as pd

from pypesto.hierarchical.problem import InnerProblem
from pypesto.hierarchical.spline_inner_problem import SplineInnerProblem

def main():
    """Napisi opis..."""

    petab_problem = petab.Problem.from_yaml(
    '/home/zebo/Documents/GitHub/examples/Boehm_JProteomeRes2014OptimalScaling_HardConstraints/Boehm_JProteomeRes2014OptimalScaling_HardConstraints.yaml')

    importer = pypesto.petab.PetabImporter(petab_problem)
    model = importer.create_model()

    timepoints=[0,2.5,5,10,15,20,30,40,50,60,80,100,120,160,200,240]
    model.setTimepoints(timepoints)

    solver = model.getSolver()
    rdata=amici.runAmiciSimulation(model, solver)

    edatas_zebo=amici.ExpData(rdata, 1.0, 0)
    edatas_list = [edatas_zebo]

    lol=pd.DataFrame()

    prob = SplineInnerProblem.from_petab_amici(petab_problem, model, edatas_list)

    print(prob.get_hard_constraints_for_group(2.0))

if __name__ == "__main__":
    main()