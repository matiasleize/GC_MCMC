import unittest
import numpy as np
from fr_mcmc.utils.chi_square import chi2_supernovae

class TestChiSquare(unittest.TestCase):

    def setUp(self):
        """Set up example data for testing."""
        self.muth = np.array([0.1, 0.2, 0.3])
        self.muobs = np.array([0.1, 0.2, 0.3])
        self.Cinv = np.linalg.inv(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_chi2_supernovae_perfect_fit(self):
        """Test chi2_supernovae with perfect fit should result in chi2 = 0."""
        chi2 = chi2_supernovae(self.muth, self.muobs, self.Cinv)
        self.assertEqual(chi2, 0.0)

    def test_chi2_supernovae_non_zero(self):
        """Test chi2_supernovae with modified muobs to create a non-zero chi2."""
        muobs = np.array([0.1, 0.25, 0.35])
        chi2 = chi2_supernovae(self.muth, muobs, self.Cinv)
        self.assertGreater(chi2, 0.0)

    def test_chi2_supernovae_invalid_input_shapes(self):
        """Test chi2_supernovae with invalid input shapes should raise ValueError."""
        with self.assertRaises(ValueError):
            chi2_supernovae(self.muth, np.array([0.1, 0.2]), self.Cinv)

    def test_chi2_supernovae_non_identity_covariance(self):
        """Test chi2_supernovae with non-identity covariance matrix."""
        Cinv = np.linalg.inv(np.array([[2, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 2]]))
        muobs = np.array([0.1, 0.25, 0.35])
        chi2 = chi2_supernovae(self.muth, muobs, Cinv)
        self.assertGreater(chi2, 0.0)

    def test_chi2_supernovae_negative_values(self):
        """Test chi2_supernovae with negative values."""
        muth = np.array([-0.1, -0.2, -0.3])
        muobs = np.array([-0.1, -0.25, -0.35])
        chi2 = chi2_supernovae(muth, muobs, self.Cinv)
        self.assertGreater(chi2, 0.0)

    def test_chi2_supernovae_empty_arrays(self):
        """Test chi2_supernovae with empty arrays should raise ValueError."""
        with self.assertRaises(ValueError):
            chi2_supernovae(np.array([]), np.array([]), np.linalg.inv(np.array([[]])))

    def test_chi2_supernovae_single_element(self):
        """Test chi2_supernovae with single-element arrays."""
        muth = np.array([0.1])
        muobs = np.array([0.1])
        Cinv = np.linalg.inv(np.array([[1]]))
        chi2 = chi2_supernovae(muth, muobs, Cinv)
        self.assertEqual(chi2, 0.0)

if __name__ == '__main__':
    unittest.main()