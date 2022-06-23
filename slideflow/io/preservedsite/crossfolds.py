import cvxpy as cp
import numpy as np
import pandas as pd

import slideflow as sf
from slideflow.util import log
from itertools import combinations
from typing import List


def flatten(arr):
    '''Flattens an array'''
    return [y for x in arr for y in x]


def generate_brute_force(
    data: pd.core.frame.DataFrame,
    category: str,
    values: List[str],
    crossfolds: int = 3,
    target_column: str = 'CV3',
    patient_column: str = 'patient',
    site_column: str = 'site',
) -> pd.core.frame.DataFrame:
    """Generate site-preserved cross-val splits through brute-force search.

    Args:
        data (pandas.DataFrame): Dataframe with slides that must be split into
            crossfolds.
        category (str): The column in data to stratify by.
        values (list(str)): A list of possible values within category to
            include for stratification
        crossfolds (int): Number of crossfolds for splitting. Defaults to 3.
        target_column (str): Name for target column to contain the assigned
            crossfolds for each patient in the output dataframe.
        patient_column (str): Column within dataframe indicating unique
            identifier for patient
        site_column (str): Column within dataframe indicating designated site
            for a patient

    Returns:
        dataframe with a new column, 'CV3' that contains values 1 - 3,
            indicating the assigned crossfold"""

    # Create new dataframe for data.
    submitters = data[patient_column].unique()
    newData = pd.merge(
        pd.DataFrame(submitters, columns=[patient_column]),
        data[[patient_column, category, site_column]],
        on=patient_column, how='left'
    )
    newData.drop_duplicates(inplace=True)
    unique_sites = data[site_column].unique()

    # Ensure that there enough sites to split across the number of crossfolds.
    if crossfolds > len(unique_sites):
        raise sf.errors.DatasetSplitError(
        "Insufficient number of sites ({}) for crossfolds ({})".format(
            len(unique_sites),
            crossfolds))

    # Create a list of all possible site combinations.
    max_sites_per_fold = 1 + len(unique_sites) - crossfolds
    all_site_combinations = [
        c for i in range(max_sites_per_fold)
        for c in combinations(unique_sites, i+1)
    ]
    # Create a list of all possible site-preserved cross-fold splits.
    all_splits = list(combinations(all_site_combinations, crossfolds))

    # Iterate through all possible cross-fold splits,
    # removing splits which are invalid.
    invalid = []
    for split in all_splits:
        # Ensure that all sites are used.
        n_sites = sum([len(i) for i in split])
        if n_sites != len(unique_sites):
            invalid.append(split)

        # Ensure that each site is only used once.
        sites_in_split = flatten(split)
        if len(sites_in_split) != len(set(sites_in_split)):
            invalid.append(split)

    # Remove invalid sites.
    for split in set(invalid):
        all_splits.remove(split)

    # Count the number of outcome values in each site.
    def n_patients_with_value_at_site(site, value):
        """Returns number of patients at a site who have an outcome value."""
        df_is_value = (newData[category] == value)
        df_is_site = (newData[site_column] == site)
        return (df_is_site & df_is_value).sum()

    n_values_by_site = {
        site: {
            value: n_patients_with_value_at_site(site, value)
            for value in values
        } for site in unique_sites
    }
    n_patients_by_site = {
        site: (newData[site_column] == site).sum()
        for site in unique_sites
    }

    # Calculate the error of each possible crossfold split.
    # Error is defined as XXX.
    crossfold_error = {}
    value_counts = {split: {} for split in all_splits}
    for split in all_splits:
        patients_in_split = sum(
            n_patients_by_site[site]
            for fold in split
            for site in fold
        )
        sum_of_squares = 0
        count = 0

        # Calculate error for each crossfold in a given split.
        for fold in split:
            patients_in_fold = sum(n_patients_by_site[site] for site in fold)
            value_counts[split][fold] = {value: 0 for value in values}

            # Iterate through each site and outcome value in the crossfold.
            for site in fold:
                for value in values:
                    value_counts[split][fold][value] += n_values_by_site[site][value]

            # Add error for value proportions in each crossfold split.
            # NOTE: 1./len(values) may not be the correct target.
            for value in values:
                value_proportion = value_counts[split][fold][value] / patients_in_fold
                sum_of_squares += (value_proportion - (1./len(values)))**2
                count += 1

            # Add error term for proportion of patients in a split.
            patient_proportion = patients_in_fold / patients_in_split
            sum_of_squares += (patient_proportion - (1./crossfolds))**2
            count += 1

        mean_square_error = sum_of_squares / count
        crossfold_error[split] = mean_square_error

    # Choose crossfold split with least error.
    best_combo = min(crossfold_error, key=crossfold_error.get)

    # Assign data by crossfold and site.
    list_for_best_combo = []
    for i in best_combo:
        sites_in_one_combo = []
        for s in i:
            sites_in_one_combo.append(s)
        list_for_best_combo.append(sites_in_one_combo)

    for i in range(crossfolds):
        data.loc[data[site_column].isin(list_for_best_combo[i]), target_column] = str(i+1)

    return data


def generate(data, category, values, crossfolds=3, target_column='CV3',
             patient_column='patient', site_column='site',
             timelimit=10):

    """Generates site preserved cross-folds, balanced on a given category.

    Args:
        data (pandas.DataFrame): Dataframe with slides that must be split into
            crossfolds.
        category (str): The column in data to stratify by.
        values (list(str)): A list of possible values within category to
            include for stratification
        crossfolds (int): Number of crossfolds for splitting. Defaults to 3.
        target_column (str): Name for target column to contain the assigned
            crossfolds for each patient in the output dataframe.
        patient_column (str): Column within dataframe indicating unique
            identifier for patient
        site_column (str): Column within dataframe indicating designated site
            for a patient
        timelimit: maximum time to spend solving

    Returns:
        dataframe with a new column, 'CV3' that contains values 1 - 3,
            indicating the assigned crossfold

    .. _Preserved-site cross-validation:
        https://doi.org/10.1038/s41467-021-24698-1
    """

    submitters = data[patient_column].unique()
    newData = pd.merge(
        pd.DataFrame(submitters, columns=[patient_column]),
        data[[patient_column, category, site_column]],
        on=patient_column, how='left'
    )
    newData.drop_duplicates(inplace=True)
    uniqueSites = data[site_column].unique()
    n = len(uniqueSites)
    listSet = []
    for v in values:
        listOrder = []
        for s in uniqueSites:
            listOrder += [len(newData[((newData[site_column] == s)
                                       & (newData[category] == v))])]
        listSet += [listOrder]
    gList = []
    for i in range(crossfolds):
        gList += [cp.Variable(n, boolean=True)]
    A = np.ones(n)
    constraints = [sum(gList) == A]
    error = 0
    for v in range(len(values)):
        for i in range(crossfolds):
            error += cp.square(
                cp.sum(crossfolds * cp.multiply(gList[i], listSet[v]))
                - sum(listSet[v])
            )
    prob = cp.Problem(cp.Minimize(error), constraints)
    prob.solve(solver='CPLEX', cplex_params={"timelimit": timelimit})
    gSites = []
    for i in range(crossfolds):
        gSites += [[]]
    for i in range(n):
        for j in range(crossfolds):
            if gList[j].value[i] > 0.5:
                gSites[j] += [uniqueSites[i]]
    for i in range(crossfolds):
        str1 = "Crossfold " + str(i+1) + " Sites: "
        j = 0
        str1 = str1 + str(gSites[i])
        log.info(str1)
    for i in range(crossfolds):
        data.loc[data[site_column].isin(gSites[i]), target_column] = str(i+1)
    return data