import cvxpy as cp
import numpy as np
import pandas as pd

# import slideflow as sf
# from slideflow.util import log
from itertools import combinations
import random


def generate_test_data():

    k = 3
    patients = [f'pt{p}' for p in range(200)]
    sites = [f'site{s}' for s in range(5)]
    outcomes = list(range(4))
    category = 'outcome_label'
    df = pd.DataFrame({
        'patient': pd.Series([random.choice(patients) for _ in range(100)]),
        'site': pd.Series([random.choice(sites) for _ in range(100)]),
        'outcome_label': pd.Series([random.choice(outcomes) for _ in range(100)])
    })
    unique_labels = df['outcome_label'].unique()
    return [df, category, unique_labels, k]

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
        print("choose less number of crossfolds")
    else:
        most_possible_sites_in_one_fold = 1 + int(len(unique_sites)) - crossfolds
        list_of_all_combos = list()
        for i in range(most_possible_sites_in_one_fold):
            list_of_all_combos += list(combinations(unique_sites, i+1))
        list_of_proper_combos = list(combinations(list_of_all_combos, crossfolds))
        removal_list = list()
        for item in list_of_proper_combos:
            item_length = 0
            list_of_items_in_a_combo = list()
            for item2 in item:
                item_length += len(item2)
                list_of_items_in_a_combo.extend(item2)
            if item_length < len(unique_sites) or item_length > len(unique_sites):
                removal_list.append(item)
            if len(list_of_items_in_a_combo) != len(set(list_of_items_in_a_combo)):
                removal_list.append(item)
        for item in set(removal_list):
            list_of_proper_combos.remove(item)
    
    # split of values per site
    data_dict = dict()
    for site in unique_sites:
        dict_of_values = dict()
        sum = 0
        for value in values:
            dict_of_values[value] = len(newData[((newData[site_column] == site) & (newData[category] == value))])
            sum += len(newData[((newData[site_column] == site) & (newData[category] == value))])
        dict_of_values['total'] = sum
        data_dict[site] = dict_of_values

    # error associated to each possible combo
    per_fold_size_target_ratio = float(1)/crossfolds
    per_site_target_ratio = float(1)/len(values)
    per_combo_errors = dict()
    dictionary_of_split = dict()
    for combo in list_of_proper_combos:
        dictionary_of_split[combo] = dict()
        mean_square_error = 0
        count = 0
        for fold in combo:
            sum2 = 0
            dictionary_of_split[combo][fold] = {value: 0 for value in values}
            dictionary_of_split[combo][fold]['total'] = 0
            for site in fold:
                for key in data_dict[str(site)].keys():
                    dictionary_of_split[combo][fold][key] += data_dict[site][key]
            sum2 += dictionary_of_split[combo][fold]['total']
            for k in dictionary_of_split[combo][fold].keys():
                 dictionary_of_split[combo][fold][k] = dictionary_of_split[combo][fold][k]/float(dictionary_of_split[combo][fold]['total'])
                 fold_site_value = dictionary_of_split[combo][fold][k]
                 count += 1
                 mean_square_error += (fold_site_value-per_site_target_ratio)**2
        for fold in combo:
            fold_total = dictionary_of_split[combo][fold]['total']/(float(sum2))
            mean_square_error += (fold_total - per_fold_size_target_ratio)**2
            count += 1
        mean_square_error = mean_square_error/count
        per_combo_errors[combo] = mean_square_error
    
    # isolate best combo by error
    min = 100000000000000000000000000000000000000000000
    best_combo = None
    for key, value in per_combo_errors.items():
        if value < min:
            best_combo = key
            min = value
        else:
            pass

    # assign data by crossfold and site
    list_for_best_combo = list()
    for i in best_combo:
        sites_in_one_combo = list()
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
    # for i in range(crossfolds):
    #     str1 = "Crossfold " + str(i+1) + " Sites: "
    #     j = 0
    #     str1 = str1 + str(gSites[i])
    #     log.info(str1)
    for i in range(crossfolds):
        data.loc[data[site_column].isin(gSites[i]), target_column] = str(i+1)
    return data

# test using fake data generator
# data = generate_test_data()
# list_of_split = generate(data[0], data[1], data[2], data[3])
# print(list_of_split)
# dictionary = generate_brute_force(data[0], data[1], data[2], data[3])
# print(dictionary)
