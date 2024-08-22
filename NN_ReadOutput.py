import os
import sys
import pandas as pd

if __name__ == "__main__":
    # Validate command line arguments
    args = sys.argv[1:]
    if len(args) != 2:
        raise ValueError(
            "This script requires 2 positional arguments, the location of the mlp output files, and the location of the ensemble output files")
    mlp_path, ens_path = args[0], args[1]
    if not os.path.exists(mlp_path) or not os.path.exists(ens_path):
        raise ValueError("Provided path does not exist")

    mlp_dfs = []
    # Extract data from output files in mlp directory
    for i in range(40): # while True is better code, but this is easier to read. 40 is arbitrary number big enough
        filepath = mlp_path.strip('/') + '/' + f"env_score_{i}"
        if not os.path.exists(filepath):
            break
        mlp_dfs.append(pd.read_csv(filepath))
    mlp_dfs = pd.concat(mlp_dfs)

    ens_dfs = []
    # Extract data from output files in ens directory
    for i in range(40):
        filepath = ens_path.strip('/') + '/' + f"env_score_{i}"
        if not os.path.exists(filepath):
            break
        ens_dfs.append(pd.read_csv(filepath))
    ens_dfs = pd.concat(ens_dfs)

    if len(ens_dfs) != len(mlp_dfs):
        raise ValueError("MLP and Ensemble output files do not match")

    ens_dfs['mlp_score'] = mlp_dfs['mlp_score']

    ens_dfs.to_csv(ens_path.strip('/') + '/' + "combined_scores.csv")

    print("Combined dataframe: ")
    print(ens_dfs)
