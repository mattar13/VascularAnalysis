import pandas as pd
import numpy as np
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
        #Store the dataframe in a location
        self.datafile_path = file_path

        # Initialize a dictionary to store dataframes by name
        self.id_sheet = pd.DataFrame() #Initialize an empty dataframe

        self.sheet_names = [] #Initialize an empty list of sheet names

        self.density_raster_dict = {} #This will be a dictionary of the density rasters

        # Load the master dataframe if file_path is provided
        if file_path:
            self.load_master_df(file_path)

    def load_master_df(self, file_path, id_sheet_name = 'Identification_Sheet'):
        """
        Loads the master datasheet from an Excel file,
        turn the identification sheet into a dataframe
        
        turn the raster sheets into a dictionary of arrays

        Parameters:
        file_path (str): Path to the Excel file containing the master dataset.

        Returns:
        pd.DataFrame: The cleaned master dataframe.
        """
        #Open the excel file
        xls = pd.ExcelFile(file_path)

        #Store the sheet names
        self.sheet_names = list(xls.sheet_names)

        #Set the identification sheet to the same name as the option
        self.id_sheet = xls.parse(id_sheet_name)

        # Extract each sheet as a dictionary
        for sheet_name in xls.sheet_names:
            if sheet_name != id_sheet_name: #If the sheet name is not the identification sheet
                sheet_data = xls.parse(sheet_name) #Parse the sheet
                self.density_raster_dict[sheet_name] = sheet_data #Store the sheet in the dictionary

    def construct_master_id_raster_df(self):
        print("Working")

    def construct_master_sheet_df(self, sheet_name):
        print(f"constructing master sheet df from {sheet_name}")
        
        #Open the master dataframe with densities
        master_df = pd.read_excel(sheet_name)
        
        #Fill all NaNs with 0
        master_df.fillna(0, inplace = True)
        # Remove columns with 'Unnamed' and specified variables
        master_df = master_df.loc[:, ~master_df.columns.str.contains('^Unnamed')]

        #Once we have the master dataframe, we can start constructing the raster dataframe
        length_columns = [col for col in master_df.columns if col.startswith('X') and col[1:].isdigit()]
        
        #This creates the knee identifier
        #print("Adding knee identifier")
        master_df = self.create_knee_identifier(master_df, length_columns)

        identifier_columns = [col for col in master_df.columns if not col.startswith('X')]

        #The end result of this function is to save each density raster to a seperate array
        self.density_raster_full = master_df[length_columns]
        self.id_sheet_full = master_df[identifier_columns]

    def create_knee_identifier(self, master_df, length_columns):
        # @title ### Create a knee identifier
        Knees = np.zeros(master_df.shape[0])
        Scores = np.zeros(master_df.shape[0])
        #Pick the threshold to order the data by
        threshold = 0.01 #Change this threshold if we want this to discover different points
        for col in range(1, master_df.shape[0]):
            score = 0 #This is a winning score
            nz_vals_below_thresh = np.where(master_df.iloc[col][length_columns]<threshold)[0]
            nz_vals_above_thresh = np.where(master_df.iloc[col][length_columns]>threshold)[0]
            fBT = fAT = 0 #Zero out all points after the analysis
            lBt = lAT = 0#len(length_columns) #Set the limits to infinity
            #find the First value below the thresh
            if nz_vals_below_thresh.size != 0:
                fBT = nz_vals_below_thresh[0]
                lBT = nz_vals_below_thresh[-1]
            #find the Last value above the thresh
            if nz_vals_above_thresh.size != 0:
                fAT = nz_vals_above_thresh[0]
                lAT = nz_vals_above_thresh[-1]

            Knees[col] = fBT #Set the knee to the first above threshold
            if fBT == 0: #the first above threshold should be 0, if not score is not great
                score += 1
                if fAT >= 10: #This will only fail if the entire array is zero.
                    score += 1
                else: #I think it is safe to now assign the knee
                    Knees[col] = lAT #Set the knee to the first above threshold

            Scores[col] = score

        master_df.insert(master_df.shape[1], "Knee", Knees)
        master_df.insert(master_df.shape[1], "Score", Scores)
        return master_df

    def seperate_by_layer(self, main_layer_sheet = "Superficial"):
        """
        Seperates the master dataframe by layer

        main_layer_sheet is referring to the sheet that contains the most
        
        """
        unique_layers = self.id_sheet_full['Layer'].unique()
        raster_dict = {}
        for layer in unique_layers:
            layer_idxs = self.id_sheet_full['Layer'] == layer
            #print(f"Layer: {layer}")
            #print(self.id_sheet_full[layer_idxs])
            raster_dict[layer] = self.id_sheet_full[layer_idxs]

        self.id_sheet = raster_dict[main_layer_sheet].reset_index(drop=True)
        #superficial_idxs = self.id_sheet_full['Layer'] == 'Superficial'
        #deep_idxs = self.id_sheet_full['Layer'] == 'Deep'
        #intermediate_idxs = self.id_sheet_full['Layer'] == 'Intermediate'
        #return self.id_sheet_full[superficial_idxs], self.id_sheet_full[deep_idxs], self.id_sheet_full[intermediate_idxs]

    def test(self):
        #Run the test function
        print('This repository is properly loaded')
        print(self.datafile_path)
        print(self.sheet_names)