import pandas as pd
import os

class DataManager:
    """
    A simple class to handle loading and storing multiple dataframes.
    """
    def __init__(self, file_path=None):
        """
        Initializes the DataManager. Optionally loads a master dataframe if a file path is provided.

        Parameters:
        file_path (str, optional): Path to the Excel file containing the master dataset. Default is None.
        """
        # Initialize a dictionary to store dataframes by name
        self.sheet_dict = {}
        
        self.datafile_path = file_path

        # Load the master dataframe if file_path is provided
        if file_path:
            self.load_master_df(file_path)

    def load_master_df(self, file_path):
        """
        Loads the master datasheet from an Excel file,
        cleans unwanted columns, and fills in NaNs with 0.

        Parameters:
        file_path (str): Path to the Excel file containing the master dataset.

        Returns:
        pd.DataFrame: The cleaned master dataframe.
        """
        #Open the excel file
        xls = pd.ExcelFile(file_path)

        # Extract each sheet as a dictionary
        self.sheet_dict = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
        self.sheet_names = list(self.sheet_dict.keys())

    def test(self):
        #Run the test function
        print('This repository is properly loaded')
        print(self.datafile_path)
        print(self.sheet_names)