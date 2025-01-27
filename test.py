from make_datafile import DataManager

def main():
    "Testing main functions"

    #This is a test for loading data using the manager
    #load_test = "test_files\\MasterSheet.xlsx"
    #data_loaded = DataManager(load_test)

    construct_test = "test_files\\FullLeaf_LengthByDistance_REMOVE_BLUE.xlsx"
    data_constructed = DataManager()
    data_constructed.construct_master_sheet_df(construct_test)
    print(data_constructed.id_sheet_full)

if __name__ == "__main__":
    main()
