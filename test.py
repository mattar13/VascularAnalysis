from make_datafile import DataManager

def test_loading():

    print("Testing loading functions") 
    # I) This is a test for loading data using the manager
    load_test = "test_files\\MasterSheet.xlsx"
    vasc_data = DataManager(load_test)

    return vasc_data#, data_constructed

def test_saving(vasc_data):
    #Save the data
    vasc_data.save_data("test_files\\test_output.xlsx")
    print("Data saved to test_files\\test_output.xlsx")

def test_retrival_methods(vasc_data):

    #Now we want to test the retrieval methods
    #get the first row of a single sheet
    vasc_data.get_density_with_index([0, 101], "SuperficialDensity")
    vasc_data.get_density_with_index(101)

    P10_df = vasc_data.get_ID_df_with_category(age = 10, genotype = "WT")

    P10_idx = vasc_data.get_index_with_category(age = 10, genotype = "WT")

    vasc_data.show_sheetnames()

    #Want to pull out all data that results in a certain averages not being met
    #Lets say we want to pull out a datasheet with all the P9 data

    sample_df = vasc_data.get_index_with_category(age = 9)
    #print(sample_df)
    sample_sheets = vasc_data.get_density_with_category(sheetname = "SuperficialDiving", age = 9)
    #print(sample_sheets[0])

def test_construction():
        #II) This is a test for constructing data using the manager
    data_constructed = DataManager() #Initialize the manager
    
    #Step 1: Load the density data
    print("Constructing from a singular master data sheet")
    construct_test = "test_files\\FullLeaf_LengthByDistance.xlsx" 
    data_constructed.construct_master_sheet_df(construct_test)
    
    #There are two different ways to load data:
    id_fn = "test_files\\P5 Length Density Vectors - file list.csv"
    density_fn = "test_files\\C3-Length Density Vectors-P5-vessel length by retina area.tif"   
    data_constructed.construct_master_id_tiff_df(id_fn, density_fn, suffix = "Density", raster_sheet_names = ["Superficial"])

    id_fn = "test_files\\Diving vessel density files.csv"
    density_fn = "test_files\\Diving vessel density vectors.tif"
    data_constructed.construct_master_id_tiff_df(id_fn, density_fn, suffix = "Diving", raster_sheet_names = ["Superficial", "Intermediate"])
    
    identification_sheet = data_constructed.id_sheet
    knee_data = identification_sheet[['KneeSuperficial', 'KneeDeep', 'KneeIntermediate']].to_numpy()
    #print(identification_sheet.loc[1,"KneeDeep"])
    
    return test_construction

def main():
    "Testing main functions"
    
    data_constructed = test_construction()

    test_saving(data_constructed)

if __name__ == "__main__":
    main()
