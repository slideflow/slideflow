import random
import unittest

import pandas as pd

import crossfolds


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.k = 3
        cls.patients = [f'pt{p}' for p in range(200)]
        possible_sites = [f'site{s}' for s in range(5)]
        cls.patient_assignments = {p: {  # type: ignore
            'outcome': random.choice(range(5)),  # type: ignore
            'site': random.choice(possible_sites)} for p in cls.patients  # type: ignore
        }
        cls.category = 'outcome_label'
        cls.df = pd.DataFrame({
            'patient': pd.Series(cls.patients),
            'site': pd.Series([cls.patient_assignments[p]['site'] for p in cls.patients]),
            'outcome_label': pd.Series([cls.patient_assignments[p]['outcome'] for p in cls.patients])
        })
        cls.sites = cls.df['site'].unique()
        cls.unique_labels = cls.df['outcome_label'].unique()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        ...

    def _test_split(self, splits):
        '''From dataset_test.py'''
        split_patients = [p for split in splits for p in split]
        # Assert no split is empty
        self.assertTrue(all([len(split) for split in splits]))
        # Assert the patient list remains the same
        self.assertTrue(sorted(split_patients) == sorted(self.patients))

    def _test_site_split(self, splits):
        '''From dataset_test.py'''
        self._test_split(splits)
        split_sites = [
            list(set([
                self.patient_assignments[p]['site']
                for p in split
            ])) for split in splits
        ]
        flattened_sites = [s for split in split_sites for s in split]
        # Assert that sites are not shared between splits
        self.assertTrue(sorted(flattened_sites) == sorted(self.sites))

    def test_brute_force(self):
        df = crossfolds.generate_brute_force(
            self.df,
            self.category,
            self.unique_labels,
            self.k
        )
        print(df)
        splits = [
            df.loc[df['CV3'] == cv].patient.to_list()
            for cv in df['CV3'].unique()
        ]
        self._test_site_split(splits)

    def test_cplex(self):
        df = crossfolds.generate(
            self.df,
            self.category,
            self.unique_labels,
            self.k
        )
        print(df)
        splits = [
            df.loc[df['CV3'] == cv].patient.to_list()
            for cv in df['CV3'].unique()
        ]
        self._test_site_split(splits)

if __name__ == '__main__':
    unittest.main()
