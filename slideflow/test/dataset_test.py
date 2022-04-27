from typing import Optional, TYPE_CHECKING
import unittest

if TYPE_CHECKING:
    import slideflow as sf

class TestLabels(unittest.TestCase):

    PROJECT = None  # type: Optional["sf.Project"]

    @classmethod
    def setUpClass(cls):
        cls.dataset = cls.PROJECT.dataset(299, 302)
        cls.num_slides = len(cls.dataset.slides())

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