import os
import glob
import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe

def extract_info(root_path):
    """Extract dataset information from TRAIN and TEST .ts files."""
    # Find TRAIN.ts file
    train_files = glob.glob(os.path.join(root_path, '*_TRAIN.ts'))
    if not train_files:
        print(f"No TRAIN.ts file found in {root_path}")
        return None
    train_file = train_files[0]

    # Find TEST.ts file
    test_files = glob.glob(os.path.join(root_path, '*_TEST.ts'))
    test_file = test_files[0] if test_files else None

    # Load TRAIN data
    df_train, labels_train = load_from_tsfile_to_dataframe(
        train_file, return_separate_X_and_y=True, replace_missing_vals_with='NaN'
    )
    labels_train = pd.Series(labels_train, dtype="category")
    num_train = len(df_train)
    num_classes = len(labels_train.cat.categories)

    # Determine sequence length
    lengths = df_train.map(lambda x: len(x)).values
    if np.all(lengths == lengths[0, 0]):
        seq_len = lengths[0, 0]
    else:
        seq_len = int(np.max(lengths))

    # Number of variates (features/dimensions)
    num_variates = len(df_train.columns)

    # Load TEST data if exists
    num_test = 0
    if test_file:
        df_test, _ = load_from_tsfile_to_dataframe(
            test_file, return_separate_X_and_y=True, replace_missing_vals_with='NaN'
        )
        num_test = len(df_test)

    return {
        'dataset': os.path.basename(root_path),
        'seq_len': seq_len,
        'num_variates': num_variates,
        'num_classes': num_classes,
        'num_train': num_train,
        'num_test': num_test
    }

if __name__ == '__main__':
    dataset_dir = './dataset'
    results = []

    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder)
        if os.path.isdir(path) and not folder.startswith('.'):  # Skip hidden folders like .gitkeep
            print(f"Processing {folder}...")
            info = extract_info(path)
            if info:
                results.append(info)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('dataset_info.csv', index=False)
    print("Dataset information saved to dataset_info.csv")
    print(df)