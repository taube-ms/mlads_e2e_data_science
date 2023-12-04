import pandas as pd


class DataLoader:
    """A simple data loader class

    Attributes:
        filepath: The path to the data file

    Methods:
        load_data: Loads the data from the file

    """

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        return pd.read_csv(self.filepath)
