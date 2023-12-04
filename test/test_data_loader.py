import unittest
from mlads_ds.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Assuming you have a test CSV file for this purpose
        loader = DataLoader("titanic.csv")
        data = loader.load_data()
        self.assertIsNotNone(data)  # Check if data is not None
        self.assertFalse(data.empty)  # Check if data frame is not empty


if __name__ == "__main__":
    unittest.main()
