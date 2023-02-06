from typing import List

import numpy as np
import pandas as pd
import slideflow as sf
from slideflow import errors
from slideflow.util import log


def generate(*args, method='auto', **kwargs):
    """Generates site preserved cross-folds, balanced on a given category.

    Preserved-site cross-validation is performed as described in the manuscript
    https://doi.org/10.1038/s41467-021-24698-1.

    Available solvers include Bonmin and CPLEX. The solver can be manually set
    with ``method``.  If not provided, the solver will default to CPLEX if
    available, and Bonmin as a fallback.

    CPLEX is properitary software by IBM.

    Bonmin can be installed with:

        .. code-block:: bash

            conda install -c conda-forge coinbonmin

    Args:
        data (pandas.DataFrame): Dataframe with slides that must be split into
            crossfolds.
        category (str): The column in data to stratify by.
        k (int): Number of crossfolds for splitting. Defaults to 3.
        target_column (str): Name for target column to contain the assigned
            crossfolds for each patient in the output dataframe.
        timelimit: maximum time to spend solving

    Returns:
        dataframe with a new column, 'CV3' that contains values 1 - 3,
        indicating the assigned crossfold
    """
    if method == 'auto':
        if not sf.util.CPLEX_AVAILABLE:
            log.info("CPLEX solver not found; falling back to pyomo/bonmin.")
            method = 'bonmin'
        else:
            method = 'cplex'
    if method == 'bonmin':
        return _generate_bonmin(*args, **kwargs)
    elif method == 'cplex':
        return _generate_cplex(*args, **kwargs)
    else:
        raise ValueError(f'Unrecognized solver {method}')


def _generate_bonmin(
    df: pd.DataFrame,
    category: str,
    k: int = 3,
    target_column: str = 'CV3',
    timelimit: int = 10
) -> pd.DataFrame:
    """Generates site preserved cross-folds, balanced on a given category,
    using the bonmin solver.

    Bonmin can be installed with:

        conda install -c conda-forge coinbonmin

    Args:
        data (pandas.DataFrame): Dataframe with slides that must be split into
            crossfolds.
        category (str): The column in data to stratify by.
        k (int): Number of crossfolds for splitting. Defaults to 3.
        target_column (str): Name for target column to contain the assigned
            crossfolds for each patient in the output dataframe.
        timelimit: maximum time to spend solving

    Returns:
        dataframe with a new column, 'CV3' that contains values 1 - 3,
        indicating the assigned crossfold

    .. _Preserved-site cross-validation:
        https://doi.org/10.1038/s41467-021-24698-1
    """
    if not sf.util.BONMIN_AVAILABLE:
        raise errors.SolverNotFoundError("Unable to find pyomo/bonmin solver.")

    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    unique_sites = df['site'].unique()
    unique_labels = df[category].unique()
    n_sites = len(unique_sites)
    n_labels = len(unique_labels)

    # Calculate number of labels seen at each site
    # Outer list is each unique label
    # Inner list is each unique site
    n_label_by_site = [
        [ len(df[((df['site'] == site) & (df[category] == label))])
          for site in unique_sites ]
        for label in unique_labels
    ]

    # Initialize model with pyomo
    model = pyo.ConcreteModel()
    model.n_sites = pyo.Param(initialize=n_sites)

    # Create boolean variables that signify whether
    # a site is included in each crossfold.
    for si in range(k):
        var = pyo.Var(pyo.RangeSet(model.n_sites), domain=pyo.Binary)
        setattr(model, f'cv{si}', var)

    def get_site_var(model, cv_idx):
        '''Function to return a cross-fold variable set from a model.'''
        return getattr(model, f'cv{cv_idx}')

    # Set constraints that sites should be assigned to exactly one crossfold.
    def get_constraint(site_index):
        def constraint_rule(m):
            return sum(get_site_var(m, cv)[site_index] for cv in range(k)) == 1
        return constraint_rule

    # Create objective rule for the optimization problem.
    def obj_rule(m):
        return sum(
            sum(
                (
                    sum(
                        get_site_var(m, cv)[si+1] * n_label_by_site[li][si] * k
                        for si in range(n_sites)
                    )
                    - sum(n_label_by_site[li])
                )**2
                for cv in range(k)
            ) for li in range(n_labels)
        )

    # Apply the objective.
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Apply constraints.
    for si in range(1, n_sites+1):
        setattr(model, f'const{si}', pyo.Constraint(rule=get_constraint(si)))

    # Solve the equation with bonmin.
    opt = SolverFactory('bonmin', validate=False)
    opt.options['bonmin.time_limit'] = timelimit
    results = opt.solve(model)
    model.solutions.store_to(results)

    # Convert solved variables into chosen sites.
    chosen_sites = [
        [ site for site_idx, site in enumerate(unique_sites)
          if getattr(model, f'cv{cv}')[site_idx+1].value > 0.5 ]
        for cv in range(k)
    ]

    # Print results and assign results to new DataFrame column.
    for i in range(k):
        log.info(f"Crossfold {i+1} Sites: {chosen_sites[i]}")

        # Assign site results.
        df.loc[df['site'].isin(chosen_sites[i]), target_column] = str(i+1)

    return df


def _generate_cplex(
    df: pd.DataFrame,
    category: str,
    k: int = 3,
    target_column: str = 'CV3',
    timelimit: int = 10
) -> pd.DataFrame:

    """Generates site preserved cross-folds, balanced on a given category,
    using the CPLEX solver.

    Args:
        data (pandas.DataFrame): Dataframe with slides that must be split into
            crossfolds.
        category (str): The column in data to stratify by.
        k (int): Number of crossfolds for splitting. Defaults to 3.
        target_column (str): Name for target column to contain the assigned
            crossfolds for each patient in the output dataframe.
        timelimit: maximum time to spend solving

    Returns:
        dataframe with a new column, 'CV3' that contains values 1 - 3,
        indicating the assigned crossfold

    .. _Preserved-site cross-validation:
        https://doi.org/10.1038/s41467-021-24698-1
    """
    if not sf.util.CPLEX_AVAILABLE:
        raise errors.SolverNotFoundError("CPLEX solver not found.")

    import cvxpy as cp

    unique_sites = df['site'].unique()
    unique_labels = df[category].unique()

    # Calculate number of labels seen at each site
    # Outer list is each unique label
    # Inner list is each unique site
    n_label_by_site = [
        [ len(df[((df['site'] == site) & (df[category] == label))])
          for site in unique_sites ]
        for label in unique_labels
    ]

    # Optimization variables.
    variables_by_cv = [
        cp.Variable(len(unique_sites), boolean=True)
        for _ in range(k)
    ]
    A = np.ones(len(unique_sites))
    constraints = [sum(variables_by_cv) == A]

    # Create and solve optimization problem.
    error = 0
    for li in range(len(unique_labels)):
        for cv in range(k):
            error += cp.square(
                cp.sum(
                    k * cp.multiply(variables_by_cv[cv],
                    n_label_by_site[li]))
                - sum(n_label_by_site[li])
            )
    prob = cp.Problem(cp.Minimize(error), constraints)
    prob.solve(solver='CPLEX', cplex_params={"timelimit": timelimit})

    # Convert solved variables into chosen sites.
    chosen_sites = [
        [ site for site_idx, site in enumerate(unique_sites)
          if variables_by_cv[cv].value[site_idx] > 0.5 ]
        for cv in range(k)
    ]

    # Print results and assign results to new DataFrame column.
    for i in range(k):
        log.info(f"Crossfold {i+1} Sites: {chosen_sites[i]}")

        # Assign site results.
        df.loc[df['site'].isin(chosen_sites[i]), target_column] = str(i+1)

    return df
