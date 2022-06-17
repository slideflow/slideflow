import logging
import unittest

import numpy as np
import slideflow as sf


class TestMetrics(unittest.TestCase):

    n_total = 10000 #25000000
    n_patients = 200 #10000
    n_labels1 = 5 #30
    n_labels2 = 2
    multi_slide_chance = 0.1

    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_logging_level = logging.getLogger('slideflow').getEffectiveLevel()  # type: ignore
        logging.getLogger('slideflow').setLevel(40)

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

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        logging.getLogger('slideflow').setLevel(cls._orig_logging_level)  # type: ignore

    def _calc_metrics(self, func, model_type):
        return sf.stats.metrics_from_pred(
            sf.stats.df_from_pred(*func()),
            self.patients,
            model_type=model_type,
            save_predictions=False,
            plot=False
        )

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

    def _get_single_linear_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.vectorize(self.labels1.get)(tile_to_slides)]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 1))]
        y_std = [np.random.random((self.n_total, 1))]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_multi_linear_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.repeat(
            np.vectorize(self.labels1.get)(tile_to_slides)[:, np.newaxis],
            2,
            axis=-1
        )]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 2))]
        y_std = [np.random.random((self.n_total, 2))]
        return y_true, y_pred, y_std, tile_to_slides

    def _get_cph_data(self):
        tile_to_slides = np.random.choice(self.slides, size=self.n_total)
        y_true = [np.expand_dims(
            np.vectorize(self.labels1.get)(tile_to_slides),
            axis=-1
        )]
        y_pred = [np.random.choice(self.n_labels1, size=(self.n_total, 2))]
        y_pred[0][:, 1] = np.random.choice([0, 1], size=(self.n_total)) # second dim is event
        y_std = [np.random.random((self.n_total, 2))]
        return y_true, y_pred, y_std, tile_to_slides

    def test_single_categorical(self):
        results = self._calc_metrics(
            self._get_single_categorical_data,
            'categorical'
        )
        self.assertTrue(results is not None)

    def test_multi_categorical(self):
        results = self._calc_metrics(
            self._get_multi_categorical_data,
            'categorical'
        )
        self.assertTrue(results is not None)

    def test_single_linear(self):
        results = self._calc_metrics(
            self._get_single_linear_data,
            'linear'
        )
        self.assertTrue(results is not None)

    def test_multi_linear(self):
        results = self._calc_metrics(
            self._get_multi_linear_data,
            'linear'
        )
        self.assertTrue(results is not None)

    def test_cph(self):
        results = self._calc_metrics(
            self._get_cph_data,
            'cph'
        )
        self.assertTrue(results is not None)

if __name__ == '__main__':
    unittest.main()
