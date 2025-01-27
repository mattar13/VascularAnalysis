import pandas as pd
import numpy as np
import os

class DataManager:
    """
    A simple class to handle loading and storing multiple dataframes.
    """
    def __init__(self, file_path=None, main_layer_sheet = "Superficial"):
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

        self.density_dict = {} #This will be a dictionary of the density rasters

        #Variables for the size of the raster
        self.raster_rows = 0
        self.raster_cols = 0
        self.raster_depth = 0

        self.main_layer_sheet = main_layer_sheet
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

        #Set the identification sheet to the same name as the option
        self.id_sheet = xls.parse(id_sheet_name)

        # Extract each sheet as a dictionary
        for sheet_name in xls.sheet_names:
            if sheet_name != id_sheet_name: #If the sheet name is not the identification sheet
                self.sheet_names.append(sheet_name)
                sheet_data = xls.parse(sheet_name) #Parse the sheet
                self.density_dict[sheet_name] = sheet_data #Store the sheet in the dictionary

    #These functions are for constructing the master sheet from a single sheet
    def construct_master_sheet_df(self, sheet_name, suffix = "Density"):
        #print(f"constructing master sheet df from {sheet_name}")
        
        #Open the master dataframe with densities
        master_df = pd.read_excel(sheet_name)
        
        #Fill all NaNs with 0
        master_df.fillna(0, inplace = True)
        # Remove columns with 'Unnamed' and specified variables
        master_df = master_df.loc[:, ~master_df.columns.str.contains('^Unnamed')]

        #Once we have the master dataframe, we can start constructing the raster dataframe
        length_columns = [col for col in master_df.columns if col.startswith('X') and col[1:].isdigit()]
        self.raster_cols = len(length_columns)

        #This creates the knee identifier
        #print("Adding knee identifier")
        master_df = self.create_knee_identifier(master_df, length_columns)

        identifier_columns = [col for col in master_df.columns if not col.startswith('X')]

        #The end result of this function is to save each density raster to a seperate array
        self.density_raster_full = master_df[length_columns]
        self.id_sheet_full = master_df[identifier_columns]
        
        #Seperate the data raster by layer
        self.seperate_by_layer(suffix)
               
        #Align the rasters
        self.align_rasters(suffix)

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

    def seperate_by_layer(self, suffix):
        """
        Seperates the master dataframe by layer

        main_layer_sheet is referring to the sheet that contains the most
        
        """
        #Pull out all the unique laters
        unique_layers = self.id_sheet_full['Layer'].unique()
        #Initialize a dictionary to store the layers
        self.raster_dict = {}
        self.id_dict = {}
        #Iterate through all unique layers
        for layer in unique_layers:
            #print(layer)
            #if a layer is "0" ignore. I hate non general solutions, but here we go
            if layer == 0:
                continue

            layer_idxs = self.id_sheet_full['Layer'] == layer
            #print(f"Layer: {layer}")
            #print(self.id_sheet_full[layer_idxs])
            self.raster_dict[layer+suffix] = self.density_raster_full[layer_idxs]
            self.id_dict[layer+suffix] = self.id_sheet_full[layer_idxs]
            self.sheet_names.append(layer+suffix)

        self.id_sheet = self.id_dict[self.main_layer_sheet+suffix].reset_index(drop=True)
        self.id_sheet.drop(columns = ['Knee', 'Layer', 'Score'], inplace = True)
        #Considering putting this in a try block
        self.id_sheet.drop(columns = ["ManualLengthAve", "ManualLengthNonZeroAve"], inplace = True)
        self.raster_rows = len(self.id_sheet)

    def align_rasters(self, suffix):
        for raster_name in self.sheet_names:
            #print(f"Aligning {raster_name}")
            if raster_name == self.main_layer_sheet+suffix:
                self.density_dict[raster_name] = self.raster_dict[raster_name].to_numpy()
            else: #We need to align the data if it is not the main layer
                #Make an empty density sheet
                empty_density_raster = np.zeros((self.raster_rows, self.raster_cols))

                id_sheet = self.id_dict[raster_name] #Pick the id_sheet
                density_raster = self.raster_dict[raster_name].to_numpy() #Pick the density raster
                for idx, row in self.id_sheet.iterrows():
                    exp_id = row['ExpNum'] #This
                    replicate = row['Replicate'] #This
                    ImageName = row['ImageName'] #and this are identifiers
                    raster_row = density_raster[(id_sheet['ExpNum'] == exp_id) & (id_sheet['Replicate'] == replicate) & (id_sheet['ImageName'] == ImageName)]
                    if raster_row.shape[0]!=0:
                        empty_density_raster[idx, :] = raster_row
                        self.density_dict[raster_name] = empty_density_raster
                #print(empty_density_raster.shape)
        #print("Getting it working")

    #These functions are for constructing the master sheet from a ID sheet and .tif files
    def construct_master_id_raster_df(self):
        print("Working")

    def test(self):
        #Run the test function
        print('This repository is properly loaded')
        print(self.datafile_path)
        print(self.sheet_names)