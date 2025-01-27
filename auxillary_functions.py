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