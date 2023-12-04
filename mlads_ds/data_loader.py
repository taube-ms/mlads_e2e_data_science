import pandas as pd


class DataLoader:
    """Load data from a file path

    Args:
        file_path (str): Path to the data file

    Returns:
        pd.DataFrame: Dataframe containing the data

    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)
