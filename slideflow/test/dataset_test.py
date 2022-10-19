import logging
import random
import shutil
import unittest

import pandas as pd
import slideflow as sf
from slideflow.test.utils import TestConfig


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        cls.PROJECT = TestConfig().create_project(overwrite=True)  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore
        if cls.PROJECT is not None:  # type: ignore
            shutil.rmtree(cls.PROJECT.root)  # type: ignore

    def test_base_annotations(self):
        dataset = self.PROJECT.dataset()
        self.assertTrue(len(dataset.annotations) == 10)

    def test_load_annotations(self):
        dataset = self.PROJECT.dataset()
        ann_df = pd.DataFrame({
            'patient': pd.Series([f'pt{p}' for p in range(100)]),
            'slide': pd.Series([f'slide{s}' for s in range(100)]),
            'linear': pd.Series([random.random() for _ in range(100)])
        })
        dataset.load_annotations(ann_df)
        self.assertTrue(len(dataset.annotations) == 100)

    def test_load_faulty_annotations_with_duplicates(self):
        dataset = self.PROJECT.dataset()
        ann_df = pd.DataFrame({
            'patient': pd.Series([f'pt{p}' for p in range(100)]),
            'slide': pd.Series(['slide_test', 'slide_test'] + [f'slide{s}' for s in range(98)]),
            'linear': pd.Series([random.random() for _ in range(100)])
        })
        with self.assertRaises(sf.errors.DatasetError):
            dataset.load_annotations(ann_df)

    def test_load_faulty_annotations_without_patient(self):
        dataset = self.PROJECT.dataset()
        ann_df = pd.DataFrame({
            'slide': pd.Series([f'slide{s}' for s in range(100)]),
            'linear': pd.Series([random.random() for _ in range(100)])
        })
        self.assertRaises(sf.errors.AnnotationsError, dataset.load_annotations, ann_df)

    def test_load_faulty_annotations_without_slide(self):
        dataset = self.PROJECT.dataset()
        ann_df = pd.DataFrame({
            'patient': pd.Series([f'pt{p}' for p in range(100)]),
            'linear': pd.Series([random.random() for _ in range(100)])
        })
        self.assertRaises(sf.errors.AnnotationsError, dataset.load_annotations, ann_df)

    def test_properties(self):
        dataset = self.PROJECT.dataset()
        self.assertTrue(dataset.num_tiles == 0)
        self.assertFalse(dataset.filters)
        self.assertFalse(dataset.filter_blank)
        self.assertFalse(dataset.min_tiles)
        self.assertTrue(dataset.num_tiles == 0)
        self.assertTrue(dataset.num_tiles == 0)
        self.assertTrue(dataset.num_tiles == 0)

    def test_faulty_balance(self):
        dataset = self.PROJECT.dataset()
        self.assertRaises(sf.errors.DatasetBalanceError, dataset.balance, 'category1')

    def test_is_float(self):
        dataset = self.PROJECT.dataset()
        self.assertTrue(dataset.is_float('linear1'))
        self.assertTrue(dataset.is_float('linear2'))
        self.assertFalse(dataset.is_float('category1'))
        self.assertFalse(dataset.is_float('category2'))


class TestSplits(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        cls.patients = [f'pt{p}' for p in range(200)]  # type: ignore
        cls.sites = [f'site{s}' for s in range(5)]  # type: ignore
        cls.outcomes = list(range(4))  # type: ignore
        cls.patients_dict = {p: {  # type: ignore
            'outcome': random.choice(cls.outcomes),  # type: ignore
            'site': random.choice(cls.sites)} for p in cls.patients  # type: ignore
        }

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore

    def _test_split(self, splits):
        split_patients = [p for split in splits for p in split]
        # Assert no split is empty
        self.assertTrue(all([len(split) for split in splits]))
        # Assert the patient list remains the same
        self.assertTrue(sorted(split_patients) == sorted(self.patients))

    def _test_site_split(self, splits):
        self._test_split(splits)
        split_sites = [
            list(set([
                self.patients_dict[p]['site']
                for p in split
            ])) for split in splits
        ]
        flattened_sites = [s for split in split_sites for s in split]
        # Assert that sites are not shared between splits
        self.assertTrue(sorted(flattened_sites) == sorted(self.sites))

    @unittest.skipIf(not sf.util.CPLEX_AVAILABLE, "CPLEX not installed")
    def test_site_preserved_cplex_three_splits(self):
        splits = sf.dataset.split_patients_preserved_site(
            self.patients_dict, n=3, balance='outcome', method='cplex'
        )
        self._test_site_split(splits)

    @unittest.skipIf(not sf.util.CPLEX_AVAILABLE, "CPLEX not installed")
    def test_site_preserved_cplex_five_splits(self):
        splits = sf.dataset.split_patients_preserved_site(
            self.patients_dict, n=5, balance='outcome', method='cplex'
        )
        self._test_site_split(splits)

    @unittest.skipIf(not sf.util.BONMIN_AVAILABLE, "Pyomo/Bonmin not installed")
    def test_site_preserved_bonmin_three_splits(self):
        splits = sf.dataset.split_patients_preserved_site(
            self.patients_dict, n=3, balance='outcome', method='bonmin'
        )
        self._test_site_split(splits)

    @unittest.skipIf(not sf.util.BONMIN_AVAILABLE, "Pyomo/Bonmin not installed")
    def test_site_preserved_bonmin_five_splits(self):
        splits = sf.dataset.split_patients_preserved_site(
            self.patients_dict, n=5, balance='outcome', method='bonmin'
        )
        self._test_site_split(splits)

    def test_balanced_split_three_splits(self):
        splits = sf.dataset.split_patients_balanced(
            self.patients_dict, n=3, balance='outcome'
        )
        self._test_split(splits)

    def test_balanced_split_five_splits(self):
        splits = sf.dataset.split_patients_balanced(
            self.patients_dict, n=5, balance='outcome'
        )
        self._test_split(splits)

    def test_split_three_splits(self):
        splits = sf.dataset.split_patients(self.patients_dict, n=3)
        self._test_split(splits)

    def test_split_five_splits(self):
        splits = sf.dataset.split_patients(self.patients_dict, n=5)
        self._test_split(splits)


class TestLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        cls.PROJECT = TestConfig().create_project(overwrite=True)  # type: ignore
        cls.dataset = cls.PROJECT.dataset()  # type: ignore
        cls.num_slides = len(cls.dataset.slides())  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore
        if cls.PROJECT is not None:  # type: ignore
            shutil.rmtree(cls.PROJECT.root)  # type: ignore

    def _check_label_format(self, labels):
        self.assertIsInstance(labels, dict)
        self.assertTrue(len(labels) == self.num_slides)
        self.assertTrue(all([isinstance(s, str) for s in labels.keys()]))

    def test_categorical_labels_by_index(self):
        labels, unique = self.dataset.labels('category1', format='index')
        self._check_label_format(labels)
        self.assertTrue(all([isinstance(lbl, int) for lbl in labels.values()]))
        self.assertIsInstance(unique, list)
        self.assertTrue(all([isinstance(lbl, str) for lbl in unique]))

    def test_categorical_labels_by_name(self):
        labels, unique = self.dataset.labels('category1', format='name')
        self._check_label_format(labels)
        self.assertTrue(all([isinstance(lbl, str) for lbl in labels.values()]))
        self.assertIsInstance(unique, list)
        self.assertTrue(all([isinstance(lbl, str) for lbl in unique]))

    def _test_linear_labels(self, use_float):
        labels, unique = self.dataset.labels('linear1', use_float=use_float)
        self._check_label_format(labels)
        self.assertTrue(all([isinstance(lbl, list) for lbl in labels.values()]))
        self.assertTrue(all([len(lbl) == 1 for lbl in labels.values()]))
        self.assertTrue(all([isinstance(lbl[0], float) for lbl in labels.values()]))
        self.assertIsInstance(unique, list)
        self.assertFalse(len(unique))

    def test_linear_labels_with_manual_float(self):
        self._test_linear_labels(use_float=True)

    def test_linear_labels_with_auto_float(self):
        self._test_linear_labels(use_float='auto')

    def test_linear_labels_with_dict_float(self):
        self._test_linear_labels(use_float={'linear1': True})

    def _test_cat_and_linear_labels(self, labels, unique, cat_idx, linear_idx, cat_format):
        # Label format checks
        self._check_label_format(labels)
        self.assertTrue('category1' in unique and 'linear1' in unique)
        self.assertIsInstance(unique, dict)

        # Categorical label checks
        self.assertTrue(all([isinstance(lbl[cat_idx], cat_format) for lbl in labels.values()]))
        self.assertTrue(all([isinstance(lbl, str) for lbl in unique['category1']]))

        # Linear label checks
        self.assertTrue(all([isinstance(lbl[linear_idx], float) for lbl in labels.values()]))
        self.assertIsInstance(unique['linear1'], list)
        self.assertFalse(len(unique['linear1']))

    def test_categorical_and_linear_labels_by_index(self):
        labels, unique = self.dataset.labels(
            ['category1', 'linear1'],
            format='index',
            use_float={'linear1': True, 'category1': False}
        )
        self._test_cat_and_linear_labels(labels, unique, 0, 1, cat_format=int)

    def test_categorical_and_linear_labels_by_name(self):
        labels, unique = self.dataset.labels(
            ['category1', 'linear1'],
            format='name',
            use_float={'linear1': True, 'category1': False}
        )
        self._test_cat_and_linear_labels(labels, unique, 0, 1, cat_format=str)

    def test_linear_and_categorical_labels_by_index(self):
        labels, unique = self.dataset.labels(
            ['linear1', 'category1'],
            format='index',
            use_float={'linear1': True, 'category1': False}
        )
        self._test_cat_and_linear_labels(labels, unique, 1, 0, cat_format=int)

    def test_linear_and_categorical_labels_by_name(self):
        labels, unique = self.dataset.labels(
            ['linear1', 'category1'],
            format='name',
            use_float={'linear1': True, 'category1': False}
        )
        self._test_cat_and_linear_labels(labels, unique, 1, 0, cat_format=str)

    def test_multi_categorical_labels_by_index(self):
        labels, unique = self.dataset.labels(
            ['category1', 'category2'],
            format='index',
            use_float=False
        )
        self._check_label_format(labels)
        self.assertTrue('category1' in unique and 'category2' in unique)
        self.assertIsInstance(unique, dict)
        for cat_idx in range(2):
            self.assertTrue(all([isinstance(lbl[cat_idx], int) for lbl in labels.values()]))
            self.assertTrue(all([isinstance(lbl, str) for lbl in unique['category1']]))

    def test_multi_categorical_labels_by_name(self):
        labels, unique = self.dataset.labels(
            ['category1', 'category2'],
            format='name',
            use_float=False
        )
        self._check_label_format(labels)
        self.assertTrue('category1' in unique and 'category2' in unique)
        self.assertIsInstance(unique, dict)
        for cat_idx in range(2):
            self.assertTrue(all([isinstance(lbl[cat_idx], str) for lbl in labels.values()]))
            self.assertTrue(all([isinstance(lbl, str) for lbl in unique['category1']]))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
