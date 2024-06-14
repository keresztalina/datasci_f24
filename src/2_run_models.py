# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree as tr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# FUNCTIONS
def load_csvs_to_dfs(filenames):
    dataframes = []
    for name in filenames:
        file_path = f'data/{name}.csv'
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f'Loaded {file_path} into DataFrame: {name}')
        except FileNotFoundError:
            print(f'File {file_path} not found.')
    return dataframes

def plot_outcome(df: pd.DataFrame):
    # Extract the last column
    last_column_name = df.columns[-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[last_column_name], bins=50)
    plt.title(f'Histogram of {last_column_name}')
    plt.ylabel(last_column_name)
    plt.grid(True)

    plt.savefig(f'plots/hist_{last_column_name}.png')

def plot_correlations(df: pd.DataFrame):
    last_column_name = df.columns[-1]
    sns.clustermap(df.corr(), cmap='viridis')
    plt.savefig(f'plots/corr_{last_column_name}.png')

def main():
    csvs = [
        'cesd_total', 
        'gad_total',
        'inq_perceivedburden',
        'inq_thwartedbelong',
        'upps_total']
    dataframes = load_csvs_to_dfs(csvs)