from make_datafile import DataManager

def main():
    "Testing main functions"

    #There are two different ways to load data: 
    # 
    # I) This is a test for loading data using the manager
    load_test = "test_files\\MasterSheet.xlsx"
    data_loaded = DataManager(load_test)
    print(data_loaded.sheet_names)

    #II) This is a test for constructing data using the manager
    construct_test = "test_files\\FullLeaf_LengthByDistance_REMOVE_BLUE.xlsx"
    data_constructed = DataManager() #Initialize the manager
    #Step 1: Load the density data
    data_constructed.construct_master_sheet_df(construct_test)

    #Step 2: Load the data from tif files (with ID keys)
    id_fn = "test_files\\Diving vessel density files.csv"
    density_fn = "test_files\\Diving vessel density vectors.tif"
    data_constructed.construct_master_id_tiff_df(id_fn, density_fn)
    print(data_constructed.sheet_names)

if __name__ == "__main__":
    main()
