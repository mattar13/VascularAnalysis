import numpy as np

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