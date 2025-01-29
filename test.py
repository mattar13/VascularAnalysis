from make_datafile import DataManager

def test_loading():

    print("Testing loading functions") 
    # I) This is a test for loading data using the manager
    load_test = "test_files\\MasterSheet.xlsx"
    data_loaded = DataManager(load_test)

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
    return data_loaded#, data_constructed

def main():
    "Testing main functions"

    #There are two different ways to load data: 
    data_loaded = test_loading()
    
    #Now we want to test the retrieval methods
    #get the first row of a single sheet
    data_loaded.get_density_row([0, 101], "SuperficialDensity")
    data_loaded.get_density_row(101)

    P10_df = data_loaded.get_category_df(age = 10, genotype = "B2")

    P10_idx = data_loaded.get_category_index(age = 10, genotype = "B2")

    data_loaded.show_sheetnames()

    #Want to pull out all data that results in a certain averages not being met
    print(data_loaded["SuperficialDensity"])

if __name__ == "__main__":
    main()
