import pandas as pd
import numpy as np
import tifffile as tiff #Opening .tif files
import re
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

    def __getitem__(self, key):
        if key in self.sheet_names:
            return self.density_dict[key]

    #This function loads the master dataframe directly into the DataManager object
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
                self.density_dict[raster_name] = self.raster_dict[raster_name]
            else: #We need to align the data if it is not the main layer
                #Make an empty density sheet
                empty_density_raster = np.zeros((self.raster_rows, self.raster_cols))

                id_sheet = self.id_dict[raster_name] #Pick the id_sheet
                density_raster = self.raster_dict[raster_name] #Pick the density raster
                for idx, row in self.id_sheet.iterrows():
                    exp_id = row['ExpNum'] #This
                    replicate = row['Replicate'] #This
                    ImageName = row['ImageName'] #and this are identifiers
                    raster_row = density_raster[(id_sheet['ExpNum'] == exp_id) & (id_sheet['Replicate'] == replicate) & (id_sheet['ImageName'] == ImageName)]
                    if raster_row.shape[0]!=0:
                        empty_density_raster[idx, :] = raster_row
                        self.density_dict[raster_name] = pd.DataFrame(empty_density_raster)
                #print(empty_density_raster.shape)
        #print("Getting it working")

    #These functions are for constructing the master sheet from a ID sheet and .tif files
    def construct_master_id_tiff_df(self, id_fn, density_fn, suffix = "Diving"):
        #The caveat behind this is that an ID file needs to be opened already
        id_df = pd.read_csv(id_fn) #Read the csv id file for the density vec
        id_df['ImageName'] = id_df['File'].apply(lambda x: re.search(r'- (.*)\.tif', x).group(1) if pd.notna(x) else None)
        density_array = tiff.imread(density_fn)  # Read all z-stacks
        
        for raster_sheet in ["Superficial", "Intermediate"]:
            empty_raster_data = np.zeros((self.raster_rows, self.raster_cols))
            for idx, row in self.id_sheet.iterrows():
                ImageName = row['ImageName'] #and this are identifiers     
                matching_rows = id_df[id_df['ImageName'] == ImageName] #Pull out id row
                for _, MATCH in matching_rows.iterrows(): #Iterate through each match and pull out
                    rows = density_array[MATCH['Slice']-1, :, MATCH['Row']-1, :]
                    rows[np.isnan(rows)] = 0.0
                    if raster_sheet == "Superficial":
                        empty_raster_data[idx, :] = rows[0, :]
                    elif raster_sheet == "Intermediate":
                        empty_raster_data[idx, :] = rows[1, :]
            self.density_dict[raster_sheet+suffix] = pd.DataFrame(empty_raster_data)
            self.sheet_names.append(raster_sheet+suffix)

    #Functions meant for saving the data
    def save_data(self, output_path):
        #Save the data to an output path
        with pd.ExcelWriter(output_path) as writer:
            #First save the identification sheet
            self.id_sheet.to_excel(writer, sheet_name='Identification_Sheet', index=False)
            #Iterate through all other sheets 
            for sheet_name in self.sheet_names:
                #This funny thing happens where sometimes the data is a numpy array
                data_frame = pd.DataFrame(self.density_dict[sheet_name])
                #Finally save the data
                data_frame.to_excel(writer, sheet_name=sheet_name, index=False)

    def show_sheetnames(self):
        print(self.sheet_names)

    def show_column_titles(self):
        print(self.id_sheet.columns.tolist())

    #These methods are about retrival and how to get certain data
    def get_density_row(self, idx, sheetname = None):
        if sheetname is None:
            return [self.density_dict[sn].iloc[idx] for sn in self.sheet_names]
        
        elif sheetname in self.sheet_names:
            return self.density_dict[sheetname].iloc[idx]

    def get_id_row(self, idx):
        return self.id_sheet.iloc[idx]
    
    def show_id_row(self, idx):
        age = self.id_sheet.iloc[idx]['Age']
        genotype = self.id_sheet.iloc[idx]['Genotype']
        eye = self.id_sheet.iloc[idx]['Eye']
        quadrant = self.id_sheet.iloc[idx]['Quadrant']
        magnification = self.id_sheet.iloc[idx]['Magnification']
        mouse_id = self.id_sheet.iloc[idx]['Mouse ID']
        print(f"Properties of data entry {idx} \n\tAge: {age} \n\tGenotype: {genotype} \n\tEye: {eye} \n\tQuadrant: {quadrant} \n\tMagnification: {magnification} \n\tMouse ID: {mouse_id}")

    def get_category_df(self, expdate=None, expnum=None, 
            replicate=None, image_name=None, 
            age=None, genotype=None, eye=None, 
            quadrant=None, magnification=None, mouse_id=None
        ):
        """
        Filters the id_sheet dataframe based on optional category parameters.
        """
        # Start with the full dataframe
        filtered_df = self.id_sheet

        # Apply filters if parameters are provided
        if expdate is not None:
            filtered_df = filtered_df[filtered_df['ExpDate'] == expdate]
        if expnum is not None:
            filtered_df = filtered_df[filtered_df['ExpNum'] == expnum]
        if replicate is not None:
            filtered_df = filtered_df[filtered_df['Replicate'] == replicate]
        if image_name is not None:
            filtered_df = filtered_df[filtered_df['ImageName'] == image_name]
        if age is not None:
            filtered_df = filtered_df[filtered_df['Age'] == age]
        if genotype is not None:
            filtered_df = filtered_df[filtered_df['Genotype'] == genotype]
        if eye is not None:
            filtered_df = filtered_df[filtered_df['Eye'] == eye]
        if quadrant is not None:
            filtered_df = filtered_df[filtered_df['Quadrant'] == quadrant]
        if magnification is not None:
            filtered_df = filtered_df[filtered_df['Magnification'] == magnification]
        if mouse_id is not None:
            filtered_df = filtered_df[filtered_df['MouseId'] == mouse_id]

        if filtered_df.empty:
            print("No data found for the given parameters.")
            return None
        else:
            return filtered_df
    
    def get_category_index(self, **kwargs):
        """
        Returns the index of the first row that matches the given category parameters.
        """
        filtered_df = self.get_category_df(**kwargs)
        if filtered_df is not None:
            return filtered_df.index.values
        else:
            return None