import unittest
import numpy as np
import pandas as pd
from data import read_data_pantheon_plus_shoes, read_data_BAO_odintsov

class TestDataFunctions(unittest.TestCase):

    def setUp(self):
        # Setup code to create test data files
        self.test_file_pantheon_plus_shoes = 'test_pantheon_plus_shoes.csv'
        self.test_file_pantheon_plus_shoes_cov = 'test_pantheon_plus_shoes_cov.txt'
        self.test_file_BAO_odintsov = 'test_BAO_odintsov.txt'
        
        # Create a sample Pantheon+ shoes data file
        df = pd.DataFrame({
            'zHD': [0.1, 0.2, 0.3],
            'column_name': [1, 2, 3],
            'another_column': [4, 5, 6]
        })
        df.to_csv(self.test_file_pantheon_plus_shoes, index=False)
        
        # Create a sample covariance matrix file
        with open(self.test_file_pantheon_plus_shoes_cov, 'w') as f:
            f.write('3\n')
            f.write('0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n0.7\n0.8\n0.9\n')
        
        # Create a sample BAO Odintsov data file
        with open(self.test_file_BAO_odintsov, 'w') as f:
            f.write('0.1 70 1 147.78\n')
            f.write('0.2 68 2 147.78\n')
            f.write('0.3 67 1.5 147.78\n')

    def tearDown(self):
        # Cleanup code to remove test data files
        import os
        os.remove(self.test_file_pantheon_plus_shoes)
        os.remove(self.test_file_pantheon_plus_shoes_cov)
        os.remove(self.test_file_BAO_odintsov)

    def test_read_data_pantheon_plus_shoes(self):
        df = read_data_pantheon_plus_shoes(self.test_file_pantheon_plus_shoes,\
                                           self.test_file_pantheon_plus_shoes_cov)
        self.assertEqual(len(df), 3)
        self.assertIn('zHD', df.columns)
        self.assertIn('column_name', df.columns)
        self.assertIn('another_column', df.columns)

    def test_read_data_BAO_odintsov(self):
        z, h, dh, rd_fid = read_data_BAO_odintsov(self.test_file_BAO_odintsov)
        self.assertEqual(len(z), 3)
        self.assertEqual(len(h), 3)
        self.assertEqual(len(dh), 3)
        self.assertEqual(len(rd_fid), 3)
        np.testing.assert_array_equal(z, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(h, np.array([70, 68, 67]))
        np.testing.assert_array_equal(dh, np.array([1, 2, 1.5]))
        np.testing.assert_array_equal(rd_fid, np.array([147.78, 147.78, 147.78]))

if __name__ == '__main__':
    unittest.main()