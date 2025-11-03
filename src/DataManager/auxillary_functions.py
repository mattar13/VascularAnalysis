import numpy as np
import pandas as pd

def test_repo():
    print('This repository is properly loaded')

def insert_zero_rows_between_max_indices(data, identifier_info, n=2, category = 'Age'):
    max_indices = {} #This makes a dictionary for the max indices per category
    for k_id in identifier_info[category].unique():
        max_indices[k_id] = identifier_info[identifier_info[category] == k_id].index.max()

    sorted_ages = sorted(max_indices.keys()) #Sort the keys of the dictionary

    #Iterate through the ages and pull out the max index
    for i, age in enumerate(sorted_ages[:-1]):  # Iterate through all but the last age group
        insert_idx = max_indices[age] + 1 + i * n  # Calculate the index where rows should be inserted
        if data.ndim == 3:
            data = np.insert(data, insert_idx, np.zeros((n, data.shape[1], data.shape[2])), axis=0)  # Insert n rows of 0s
        elif data.ndim == 2:
            data = np.insert(data, insert_idx, np.zeros((n, data.shape[1])), axis=0)
        else:
            data = np.insert(data, insert_idx, np.zeros((n)), axis=0)
    return data

def apply_lut(image, lut):
    """Applies a lookup table to an RGB image."""
    result = np.zeros_like(image)
    for i in range(3):  # Loop over RGB channels
        result[:, :, i] = lut[image[:, :, i]]
    return result


def adjust_to_4d(array):
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)  # Add one dimension at the front
    elif array.ndim == 2:
        array = np.expand_dims(array, axis=(0, 1))  # Add two dimensions at the front
    elif array.ndim < 2:
        raise ValueError("Array must have at least 2 dimensions")
    return array

def pad_columns(array, pad_val = 34):
    if array.ndim != 4:
        raise ValueError("Input array must be 4-dimensional")
    
    current_columns = array.shape[3]
    if current_columns < pad_val:
        # Calculate the number of zero columns to add
        columns_to_add = pad_val - current_columns
        # Create an array of zeros with the same shape except for the fourth dimension
        zero_padding = np.zeros(array.shape[:3] + (columns_to_add,), dtype=array.dtype)
        # Concatenate the original array with the zero padding along the fourth dimension
        array = np.concatenate((array, zero_padding), axis=3)
    
    return array


    # Function to extract central (index 5) and peripheral (index 20) values from all layers
def extract_central_peripheral_all_layers(dm, genotype, sheet_names):
    """
    Extract central and peripheral vessel density values for a given genotype from all layers.
    
    Parameters:
    -----------
    dm : DataManager
        DataManager object with loaded data
    genotype : str
        Genotype to filter (e.g., 'WT' or 'B2')
    sheet_names : list
        List of sheet names to extract data from (e.g., ['SuperficialDensity', 'IntermediateDensity', 'DeepDensity'])
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Age, Genotype, Layer, Central, Peripheral
    """
    # Get filtered dataframe for the genotype
    filtered_df = dm.get_ID_df_with_category(genotype=genotype)
    
    if filtered_df is None or filtered_df.empty:
        print(f"No data found for genotype: {genotype}")
        return pd.DataFrame()
    
    # Extract data for each layer
    all_results = []
    
    for sheet_name in sheet_names:
        if sheet_name not in dm.density_dict:
            print(f"Warning: Sheet {sheet_name} not found, skipping...")
            continue
            
        # Extract layer name (remove 'Density' suffix)
        layer_name = sheet_name.replace('Density', '')
        
        # Get the raster data for this sheet
        raster_data = dm.density_dict[sheet_name]
        
        # Convert to numpy array for easier indexing
        raster_array = raster_data.to_numpy()
        
        # Extract data for each sample
        for idx in filtered_df.index:
            age = filtered_df.loc[idx, 'Age']
            genotype_val = filtered_df.loc[idx, 'Genotype']
            
            # Extract values at index 5 (central) and index 20 (peripheral)
            central = raster_array[idx, 5]
            peripheral = raster_array[idx, 20]
            
            all_results.append({
                'Age': age,
                'Genotype': genotype_val,
                'Layer': layer_name,
                'Central': central,
                'Peripheral': peripheral
            })
    
    return pd.DataFrame(all_results)