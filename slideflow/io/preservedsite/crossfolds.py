import cvxpy as cp
import numpy as np
import pandas as pd

import slideflow as sf
from slideflow.util import log
from itertools import combinations
import random


def generate_brute_force(data, category, values, crossfolds=3, target_column='CV3',
             patient_column='patient', site_column='site',
             timelimit=10):
    # New dataframe for data
    submitters = data[patient_column].unique()
    newData = pd.merge(
        pd.DataFrame(submitters, columns=[patient_column]),
        data[[patient_column, category, site_column]],
        on=patient_column, how='left'
    )
    newData.drop_duplicates(inplace=True)
    unique_sites = data[site_column].unique()

    # list of possible combinations of sites in folds; also built in check for use case when someone chooses the wrong number of folds
    if crossfolds > len(unique_sites):
        raise sf.errors.DatasetSplitError(
        "Insufficient number of sites ({}) for number of crossfolds ({})".format(
            len(unique_sites),
            crossfolds))

    most_possible_sites_in_one_fold = 1 + len(unique_sites) - crossfolds
    all_folds = [c for i in range(most_possible_sites_in_one_fold) for c in combinations(unique_sites, i+1)]
    list_of_possible_crossfolds = list(combinations(all_folds, crossfolds))
    removal_list = []
    for possible_crossfold in list_of_possible_crossfolds:
        item_length = sum([len(i) for i in possible_crossfold])
        sites_in_a_possible_crossfold = [site for site in possible_crossfold]
        if item_length < len(unique_sites) or item_length > len(unique_sites):
            removal_list.append(possible_crossfold)
        if len(sites_in_a_possible_crossfold) != len(set(sites_in_a_possible_crossfold)):
            removal_list.append(possible_crossfold)
    for possible_crossfold in set(removal_list):
        list_of_possible_crossfolds.remove(possible_crossfold)

    # split of values per site
    split_of_values_per_site = {}
    for site in unique_sites:
        dict_of_values = {}
        _sum = 0
        for value in values:
            dict_of_values[value] = len(newData[((newData[site_column] == site) & (newData[category] == value))])
            _sum += len(newData[((newData[site_column] == site) & (newData[category] == value))])
        dict_of_values['total'] = _sum
        split_of_values_per_site[site] = dict_of_values

    # error associated to each possible combo
    per_fold_size_target_ratio = 1./crossfolds
    per_site_target_ratio = 1./len(values)
    per_crossfold_combo_errors = {}
    dictionary_of_value_split_by_crossfold = {}
    for crossfold_possible in list_of_possible_crossfolds:
        dictionary_of_value_split_by_crossfold[crossfold_possible] = {}
        sum_of_squares = 0
        count = 0
        for fold in crossfold_possible:
            sum_of_total_data_per_fold = 0
            dictionary_of_value_split_by_crossfold[crossfold_possible][fold] = {value: 0 for value in values}
            dictionary_of_value_split_by_crossfold[crossfold_possible][fold]['total'] = 0
            for site in fold:
                for key in split_of_values_per_site[str(site)].keys():
                    dictionary_of_value_split_by_crossfold[crossfold_possible][fold][key] += split_of_values_per_site[site][key]
            sum_of_total_data_per_fold += dictionary_of_value_split_by_crossfold[crossfold_possible][fold]['total']
            for k in dictionary_of_value_split_by_crossfold[crossfold_possible][fold].keys():
                 dictionary_of_value_split_by_crossfold[crossfold_possible][fold][k] = dictionary_of_value_split_by_crossfold[crossfold_possible][fold][k]/float(dictionary_of_value_split_by_crossfold[crossfold_possible][fold]['total'])
                 fold_site_value = dictionary_of_value_split_by_crossfold[crossfold_possible][fold][k]
                 count += 1
                 sum_of_squares += (fold_site_value-per_site_target_ratio)**2
        for fold in crossfold_possible:
            fold_total = dictionary_of_value_split_by_crossfold[crossfold_possible][fold]['total']/(float(sum_of_total_data_per_fold))
            sum_of_squares += (fold_total - per_fold_size_target_ratio)**2
            count += 1
        mean_square_error = sum_of_squares/count
        per_crossfold_combo_errors[crossfold_possible] = mean_square_error

    # isolate best combo by error
    best_combo = min(per_crossfold_combo_errors, key=per_crossfold_combo_errors.get)

    # assign data by crossfold and site
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