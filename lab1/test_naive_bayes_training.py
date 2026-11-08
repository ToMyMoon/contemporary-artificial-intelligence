import unittest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from naive_bayes_training import train_naive_bayes, cross_entropy


class IncrementalNBTests(unittest.TestCase):
    def test_counts_predictions_and_recorded_losses(self):
        rng = np.random.default_rng(42)
        X = csr_matrix(rng.random((35, 9)))
        y = np.arange(35) % 3
        model, history = train_naive_bayes(.1, X, y, X, y, batch_size=8)
        reference = MultinomialNB(alpha=.1).fit(X, y)
        np.testing.assert_allclose(model.feature_count_, reference.feature_count_)
        np.testing.assert_array_equal(model.class_count_, reference.class_count_)
        np.testing.assert_allclose(model.predict_log_proba(X), reference.predict_log_proba(X))
        self.assertEqual(len(history), 5)
        self.assertEqual(history.batch_size.sum(), 35)
        self.assertEqual(history.samples_seen.iloc[-1], 35)
        self.assertAlmostEqual(history.val_loss.iloc[-1], cross_entropy(reference, X, y))
        self.assertTrue(np.isfinite(history.to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
