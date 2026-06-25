import os
import pandas as pd

    
def read_data(data_dir, image_dir, annotation_file, chunk_idx, total_chunks=8):
    data_path = os.path.join(data_dir, annotation_file)
    data = pd.read_csv(data_path, sep='\t')
    data['image_path'] = data['image_file'].apply(lambda p: os.path.join(image_dir, p))

    
    total_rows = len(data)
    if total_rows == 0:
        print("Warning: Dataset is empty!")
        return data

    
    actual_chunks = min(total_chunks, total_rows)

    if chunk_idx >= actual_chunks:
        print(f"Warning: chunk_idx {chunk_idx} is beyond available chunks ({actual_chunks}). Returning empty DataFrame.")
        return data.iloc[0:0]

    chunk_size = total_rows // actual_chunks
    remainder = total_rows % actual_chunks

    if chunk_idx < remainder:
        # First 'remainder' chunks get an extra row
        start_idx = chunk_idx * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        # Remaining chunks get the base chunk_size
        start_idx = remainder * (chunk_size + 1) + (chunk_idx - remainder) * chunk_size
        end_idx = start_idx + chunk_size

    data = data.iloc[start_idx:end_idx]
    print(f"Processing chunk {chunk_idx}/{actual_chunks}: rows {start_idx} to {end_idx} (total: {len(data)} rows)")
    return data