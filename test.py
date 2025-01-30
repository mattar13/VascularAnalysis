from make_datafile import DataManager

def test_loading():

    print("Testing loading functions") 
    # I) This is a test for loading data using the manager
    load_test = "test_files\\MasterSheet.xlsx"
    vasc_data = DataManager(load_test)

    #II) This is a test for constructing data using the manager
    #data_constructed = DataManager() #Initialize the manager
    #Step 1: Load the density data
    #construct_test = "test_files\\FullLeaf_LengthByDistance_REMOVE_BLUE.xlsx"
    #data_constructed.construct_master_sheet_df(construct_test)

    #Step 2: Load the data from tif files (with ID keys)
    #id_fn = "test_files\\Diving vessel density files.csv"
    #density_fn = "test_files\\Diving vessel density vectors.tif"
    #data_constructed.construct_master_id_tiff_df(id_fn, density_fn)
    #print("Test successful")
    #Save the data
    #data_constructed.save_data("test_files\\test_output.xlsx")
    #print("Data saved to test_files\\test_output.xlsx")
    return vasc_data#, data_constructed

def main():
    "Testing main functions"

    #There are two different ways to load data: 
    vasc_data = test_loading()
    
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
    sample_sheets = vasc_data.get_density_with_category(age = 9)
    print(sample_sheets["SuperficialDensity"])
    #print(sample_sheets[0])

if __name__ == "__main__":
    main()
