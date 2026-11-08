"""检查手写梯度、学习能力和固定种子的可复现性。"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from logistic_regression import LinearTextClassifier


class ModelTests(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(5)
        X = csr_matrix(rng.normal(size=(8, 4)))
        y = np.arange(8) % 3
        for kind in ('logistic',):
            model = LinearTextClassifier(kind, C=2).initialize(4, y)
            model.weights_[:] = rng.normal(scale=0.03, size=(4, 3))
            model.bias_[:] = rng.normal(scale=0.03, size=3)
            _, gw, gb = model.loss_gradient(X, y)
            for parameter, gradient in [(model.weights_, gw), (model.bias_, gb)]:
                for index in np.ndindex(parameter.shape):
                    old = parameter[index]
                    parameter[index] = old + 1e-6
                    plus = model.losses(X, y)[1]
                    parameter[index] = old - 1e-6
                    minus = model.losses(X, y)[1]
                    parameter[index] = old
                    self.assertAlmostEqual((plus-minus)/2e-6, gradient[index], places=6)

    def test_learning_and_reproducibility(self):
        y = np.tile(np.arange(3), 20)
        X = csr_matrix(np.eye(3)[y])
        for kind in ('logistic',):
            models = [LinearTextClassifier(kind, C=10, seed=3).initialize(3, y) for _ in range(2)]
            initial = models[0].losses(X, y)[1]
            for model in models:
                for _ in range(80):
                    model.train_epoch(X, y)
                self.assertLess(model.losses(X, y)[1], initial)
                np.testing.assert_array_equal(model.predict(X), y)
            np.testing.assert_array_equal(models[0].weights_, models[1].weights_)


if __name__ == '__main__':
    unittest.main()
