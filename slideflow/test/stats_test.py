import logging
import unittest
from types import SimpleNamespace

import numpy as np
import slideflow as sf


class TestSlideMap(unittest.TestCase):

    n_tiles = 10
    n_slides = 20

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        cls.slides = [f'slide{s}' for s in range(cls.n_slides)]  # type: ignore
        ftrs = sf.DatasetFeatures(None, None)
        ftrs.slides = cls.slides
        ftrs.predictions = {s: np.random.rand(cls.n_tiles, 2) for s in cls.slides}
        ftrs.activations = {s: np.random.rand(cls.n_tiles, 10) for s in cls.slides}
        ftrs.locations = {s: np.random.rand(cls.n_tiles, 2) for s in cls.slides}
        ftrs.uncertainty = {s: np.random.rand(cls.n_tiles, 2) for s in cls.slides}
        ftrs.num_features = 10
        ftrs.feature_generator = SimpleNamespace(uq=True)
        cls.DummyDatasetFeatures = ftrs
        cls.umap_kw = dict(n_neighbors=5)
        cls.slidemap = sf.SlideMap.from_features(
            cls.DummyDatasetFeatures,  # type:ignore
            **cls.umap_kw
        )

    @classmethod
    def tearDownClass(cls) -> None:
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore
        return super().tearDownClass()

    def test_init_from_features(self):
        self.assertEqual(len(self.slidemap.activations()), self.n_tiles * self.n_slides)

    def test_init_from_features_centroid(self):
        slidemap = sf.SlideMap.from_features(
            self.DummyDatasetFeatures,
            map_slide='centroid',
            **self.umap_kw
        )
        self.assertEqual(len(slidemap.activations()), self.n_slides)

    def test_init_from_features_average(self):
        slidemap = sf.SlideMap.from_features(
            self.DummyDatasetFeatures,
            map_slide='average',
            **self.umap_kw
        )
        self.assertEqual(len(slidemap.activations()), self.n_slides)

    def test_cluster(self):
        self.slidemap.cluster(5)
        self.assertEqual(len(self.slidemap.data.cluster.unique()), 5)

    def test_label_by_uncertainty(self):
        self.slidemap.label_by_uncertainty(0)

    def test_label_by_preds(self):
        self.slidemap.label_by_preds(0)

    def test_label_by_slide(self):
        dummy_labels = {s: np.random.choice(['test1', 'test2']) for s in self.slides}
        self.slidemap.label_by_slide(dummy_labels)
        self.assertTrue(sorted(list(self.slidemap.data.label.unique())) == ['test1', 'test2'])


class TestMetrics(unittest.TestCase):

    n_total = 1000
    n_patients = 20
    n_labels1 = 3
    n_labels2 = 2
    multi_slide_chance = 0.1

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)

        cls.patients_arr = np.array([f'patient{p}' for p in range(cls.n_patients)])  # type: ignore
        cls.is_multi_slide = np.random.random(cls.n_patients) < cls.multi_slide_chance  # type: ignore
        cls.patients = {}  # type: ignore
        cls.labels1_arr = np.arange(cls.n_labels1)  # type: ignore
        cls.labels2_arr = np.arange(cls.n_labels2)  # type: ignore
        cls.labels1 = {}  # type: ignore
        cls.labels2 = {}  # type: ignore

        def add_labels(labels_dict, label, p, i):
            labels_dict.update({
                f'{p}-slide0': label
            })
            cls.patients.update({
                f'{p}-slide0': p
            })
            if cls.is_multi_slide[i]:
                labels_dict.update({
                    f'{p}-slide1': label
                })
                cls.patients.update({
                    f'{p}-slide1': p
                })

        for i, p in enumerate(cls.patients_arr):  # type: ignore
            add_labels(cls.labels1, np.random.choice(cls.labels1_arr), p, i)  # type: ignore
            add_labels(cls.labels2, np.random.choice(cls.labels2_arr), p, i)  # type: ignore
        cls.slides = np.array(list(cls.patients.keys()))  # type: ignore
        cls.n_slides = len(list(set(cls.slides)))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore

    def _get_single_categorical_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.vectorize(self.labels1.get)(tile_to_slides)]
        y_pred = [np.random.random((self.n_total, self.n_labels1))]
        y_std = [np.random.random((self.n_total, self.n_labels1))]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_multi_categorical_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [
            np.vectorize(self.labels1.get)(tile_to_slides),
            np.vectorize(self.labels2.get)(tile_to_slides)
        ]
        y_pred = [
            np.random.random((self.n_total, self.n_labels1)),
            np.random.random((self.n_total, self.n_labels2))
        ]
        y_std = [
            np.random.random((self.n_total, self.n_labels1)),
            np.random.random((self.n_total, self.n_labels2))
        ]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_single_continuous_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.vectorize(self.labels1.get)(tile_to_slides)]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 1))]
        y_std = [np.random.random((self.n_total, 1))]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_multi_continuous_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.repeat(
            np.vectorize(self.labels1.get)(tile_to_slides)[:, np.newaxis],
            2,
            axis=-1
        )]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 2))]
        y_std = [np.random.random((self.n_total, 2))]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_survival_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.expand_dims(
            np.vectorize(self.labels1.get)(tile_to_slides),
            axis=-1
        )]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 2))]
        y_pred[0][:, 1] = np.random.choice([0, 1], size=(self.n_total)) # second dim is event
        y_std = [np.random.random((self.n_total, 2))]
        return y_true, y_pred, y_std, tile_to_slides

    def _group_reduce(self, df):
        dfs = sf.stats.group_reduce(df, patients=self.patients)
        self.assertIn('tile', dfs)
        self.assertIn('slide', dfs)
        self.assertIn('patient', dfs)
        self.assertEqual(len(dfs['tile']), self.n_total)
        self.assertEqual(len(dfs['slide']), self.n_slides)
        self.assertEqual(len(dfs['patient']), self.n_patients)
        self.assertIn('slide', dfs['tile'].columns)
        self.assertIn('patient', dfs['tile'].columns)
        self.assertIn('slide', dfs['slide'].columns)
        self.assertIn('patient', dfs['patient'].columns)
        return dfs

    def _assert_classification_metrics(self, metrics, outcomes, lengths):
        for outcome, length in zip(outcomes, lengths):
            self.assertTrue(metrics is not None)
            self.assertIsInstance(metrics, dict)
            self.assertIn('auc', metrics)
            self.assertIn('ap', metrics)
            self.assertIn(outcome, metrics['auc'])
            self.assertEqual(len(metrics['auc'][outcome]), length)
            self.assertEqual(len(metrics['ap'][outcome]), length)

    def _assert_regression_metrics(self, metrics, n_outcomes):
        self.assertTrue(metrics is not None)
        self.assertIsInstance(metrics, dict)
        self.assertIn('r_squared', metrics)
        self.assertEqual(n_outcomes, len(metrics['r_squared']))

    def _assert_survival_metrics(self, metrics):
        self.assertTrue(metrics is not None)
        self.assertIsInstance(metrics, dict)
        self.assertIn('c_index', metrics)
        self.assertIsInstance(metrics['c_index'], float)

    def test_single_categorical(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_single_categorical_data()
        )
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.classification_metrics(_df, level=level)
            self._assert_classification_metrics(metrics, ['out0'], [self.n_labels1])

    def test_single_categorical_named(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_single_categorical_data()
        )
        tile_df = sf.stats.name_columns(tile_df, 'classification', 'Named1')
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.classification_metrics(_df, level=level)
            self._assert_classification_metrics(metrics, ['Named1'], [self.n_labels1])

    def test_multi_categorical(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_multi_categorical_data()
        )
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.classification_metrics(_df, level=level)
            self._assert_classification_metrics(
                metrics,
                ['out0', 'out1'],
                [self.n_labels1, self.n_labels2]
            )

    def test_multi_categorical_named(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_multi_categorical_data()
        )
        tile_df = sf.stats.name_columns(
            tile_df,
            'classification',
            ['Named1', 'Named2']
        )
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.classification_metrics(_df, level=level)
            self._assert_classification_metrics(
                metrics,
                ['Named1', 'Named2'],
                [self.n_labels1, self.n_labels2]
            )

    def test_single_continuous(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_single_continuous_data()
        )
        tile_df = sf.stats.name_columns(tile_df, 'regression', ['NamedContinuous1'])
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.regression_metrics(_df, level=level)
            self._assert_regression_metrics(metrics, 1)

    def test_multi_continuous(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_multi_continuous_data()
        )
        tile_df = sf.stats.name_columns(
            tile_df,
            'regression',
            ['NamedContinuous1', 'NamedContinuous2']
        )
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.regression_metrics(_df, level=level)
            self._assert_regression_metrics(metrics, 2)

    def test_survival(self):
        tile_df = sf.stats.df_from_pred(
            *self._get_survival_data()
        )
        tile_df = sf.stats.name_columns(tile_df, 'survival')
        dfs = self._group_reduce(tile_df)
        for level, _df in dfs.items():
            metrics = sf.stats.metrics.survival_metrics(_df, level=level)
            self._assert_survival_metrics(metrics)

# -----------------------------------------------------------------------------# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
