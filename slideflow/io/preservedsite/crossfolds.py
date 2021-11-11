import pandas as pd
import numpy as np
import cvxpy as cp
import cplex
import slideflow as sf
from slideflow.util import log

def generate(data, category, values, crossfolds = 3, target_column = 'CV3', patient_column = sf.util.TCGA.patient, site_column = 'SITE', timelimit = 10):

    """Generates 3 site preserved cross folds with optimal stratification of category
    Input:
        data: dataframe with slides that must be split into crossfolds.
        category: the column in data to stratify by
        values: a list of possible values within category to include for stratification
        crossfolds: number of crossfolds to split data into
        target_column: name for target column to contain the assigned crossfolds for each patient in the output dataframe
        patient_column: column within dataframe indicating unique identifier for patient
        site_column: column within dataframe indicating designated site for a patient
        timelimit: maximum time to spend solving
    Output:
        dataframe with a new column, 'CV3' that contains values 1 - 3, indicating the assigned crossfold

    .. _Preserved-site cross-validation:
        https://doi.org/10.1038/s41467-021-24698-1
    """

    submitters = data[patient_column].unique()
    newData = pd.merge(pd.DataFrame(submitters, columns=[patient_column]), data[[patient_column, category, site_column]], on=patient_column, how='left')
    newData.drop_duplicates(inplace=True)
    uniqueSites = data[site_column].unique()
    n = len(uniqueSites)
    listSet = []
    for v in values:
        listOrder = []
        for s in uniqueSites:
            listOrder += [len(newData[(newData[site_column] == s) & (newData[category] == v)].index)]
        listSet += [listOrder]
    gList = []
    for i in range(crossfolds):
        gList += [cp.Variable(n, boolean=True)]
    A = np.ones(n)
    constraints = [sum(gList) == A]
    error = 0
    for v in range(len(values)):
        for i in range(crossfolds):
            error += cp.square(cp.sum(crossfolds * cp.multiply(gList[i], listSet[v])) - sum(listSet[v]))
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
        #for s in listSet:
        #    str1 = str1 + str(values[j]) + " - " + str(int(np.dot(gList[i].value, s))) + " "
        #    j = j + 1
        str1 = str1 + str(gSites[i])
        log.info(str1)
    bins = pd.DataFrame()
    for i in range(crossfolds):
        data.loc[data[site_column].isin(gSites[i]), target_column] = str(i+1)
    return data