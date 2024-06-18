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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree as tr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# FUNCTIONS
def load_csvs_to_dfs(filenames):
    ''' 
    Loads several .csv files into a list of dataframes.
    Args:
        filenames (list): list of filenames of the .csv files we want to load
    '''
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

def plot_outcome(df):
    ''' 
    Plots a histogram of the outcome variable in the dataframe, then saves 
    the plot. The outcome variable must be the last column of the dataframe, 
    and must be numeric.
    Args:
        df (pd.DataFrame): dataframe containing the predictors (X) and outcome (y)
    '''
    # Extract the last column
    last_column_name = df.columns[-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(
        df[last_column_name], 
        bins=50)
    plt.title(f'Histogram of {last_column_name}')
    plt.ylabel(last_column_name)
    plt.grid(True)

    plt.savefig(f'plots/hist_{last_column_name}.png')

def plot_correlations(df):
    ''' 
    Creates a correlation plot of all the predictors and the outcome variable,
    then saves the plot.
    Args:
        df (pd.DataFrame): dataframe containing the predictors (X) and outcome (y)
    '''
    last_column_name = df.columns[-1]
    sns.clustermap(
        df.corr(), 
        cmap='viridis')
    plt.savefig(f'plots/corr_{last_column_name}.png')

def create_splits(df):
    ''' 
    Turns the dataframe into a numpy array, then creates train, test and validation
    splits.
    Args:
        df (pd.DataFrame): dataframe containing the predictors (X) and outcome (y)
    '''
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.15, 
        random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train,
        test_size=X_test.shape[0] / X_train.shape[0],  
        random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def identify_variable_types(data):
    ''' 
    Automatically identifies continous and dummy variables in an array.
    Args:
        data (np.array): dataframe containing the predictors (X) and outcome (y)
    '''
    continuous_vars = []
    dummy_vars = []

    num_columns = data.shape[1]
    
    for i in range(num_columns):
        unique_values = np.unique(data[:, i])
        if len(unique_values) == 2 and np.array_equal(unique_values, [0, 1]):
            dummy_vars.append(i)
        else:
            continuous_vars.append(i)
    
    return continuous_vars, dummy_vars

def transform_X(X_train, X_val, X_test):
    ''' 
    Prepares the train, validation and test sets to be passed into the model. A 
    scaler is fit to the train set, and the train, validation and test sets are
    transformed accordingly. Then principal component analysis is performed on 
    the train split retaining components explaining 95% of the variance. The vali-
    dation and test sets are transformed accordingly.
    Args:
        X_train (np.array): training split of predictors (X)
        X_val (np.array): validation split of predictors (X)
        X_test (np.array): test split of predictors (X)
    '''

    continuous_cols, dummy_cols = identify_variable_types(X_train)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols), # scale
            ('dummy', 'passthrough', dummy_cols) # makes no sense to scale
        ]
    )
    X_train = preprocessor.fit_transform(X_train) # fit and transform
    X_val = preprocessor.transform(X_val) # only transform
    X_test = preprocessor.transform(X_test) # only transform

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    return X_train, X_val, X_test

def loop_through_dfs(list_of_dfs):
    ''' 
    First defines two functions necessary for running the models.
    Then loops through every dataframe in a list of dataframes and:
    - plots the outcome variable, 
    - plots correlations between the variables, 
    - creates train, validation and test splits, 
    - scales X and performs PCA,
    - runs a null model which predicts the mean of y,
    - runs a basic linear model,
    - runs several Ridge- and Lasso-regularized linear models,
    - runs Random Forest models,
    - runs XGBoost models,
    - compares the models based on RMSE.
    Args:
        list_of_dfs (list): list of dataframes on which to perform the above actions
    '''

    def run_on_splits(func):
        def _run_loop(*args, **kwargs):
            for x,y,nsplit in zip([X_train, X_val, X_test],
                                [y_train, y_val, y_test],
                                ['train', 'val', 'test']):
                func(*args, X=x, y=y, nsplit=nsplit, **kwargs)
        return _run_loop

    @run_on_splits
    def evaluate(model, X, y, nsplit, model_name, constant_value=None):
        ''' Evaluates the performance of a model 
        Args:
            model (sklearn.Estimator): fitted sklearn estimator
            X (np.array): predictors
            y (np.array): true outcome
            nsplit (str): name of the split
            model_name (str): string id of the model
            constant_value (int or None): relevant if the model predicts a constant
        '''
        if constant_value is not None:
            preds = np.array([constant_value] * y.shape[0])
        else:
            preds = model.predict(X)
        r2 = r2_score(y, preds)
        performance = np.sqrt(mean_squared_error(y, preds))
        model_performances.append({'model': model_name,
                            'split': nsplit,
                            'rmse': round(performance, 4),
                            'r2': round(r2, 4)})
    
    for idx, df in enumerate(list_of_dfs):
        
        print("Preparing for analysis...")
        # plot outcome variable
        plot_outcome(df)

        # plot correlations
        plot_correlations(df)

        # create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_splits(df)
        
        # transform X
        X_train, X_val, X_test = transform_X(X_train, X_val, X_test)
        print("...Preparations completed!")
        
        model_performances = []

        # run null model
        print("Running null model...")
        evaluate(
            model=None, 
            model_name='dummy', 
            constant_value=y_train.mean())
        print("...Ran null model!")

        # run plain linear regression
        print("Running linear regression...")
        reg = LinearRegression().fit(X_train, y_train)
        evaluate(
            model=reg, 
            model_name='linear')
        print("...Ran linear regression!")

        # run ridge- and lasso-regularized versions
        print("Running Ridge and Lasso regressions...")
        models = {} 
        models['linear-0.0'] = reg
        for alpha in [0.01, 0.1, 0.2, 0.5, 1.0, 20.0, 10.0, 100.0, 1000.0]:
            for est in [Lasso, Ridge]:
                if est == Lasso:
                    id = 'lasso'
                else:
                    id = 'ridge'
                reg = est(alpha=alpha).fit(X_train, y_train)
                models[f'{id}-{alpha}'] = reg
                evaluate(
                    model=reg, 
                    model_name=f'{id}-alpha-{alpha}')
        print("...Ran Lasso and Ridge!")

        # run random forest regression
        print("Running random forest, this will take a while...")
        rfreg = RandomForestRegressor(random_state=42)
        param_grid = { 
            'n_estimators': [10, 20, 100, 200, 500],
            'max_depth' : [2, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'max_features': [0.3, 0.6, 0.9], 
            'ccp_alpha': [0.01, 0.1, 1.0]
        }
        cv_rfr = RandomizedSearchCV(
            estimator=rfreg, 
            param_distributions=param_grid,
            scoring='neg_mean_squared_error', 
            n_iter=100, 
            cv=5)
        cv_rfr.fit(X_train, y_train)
        evaluate(
            model=cv_rfr.best_estimator_, 
            model_name=f'random-forest')
        print("...Ran Random Forest!")

        # run xgboost regression
        print("Running XGBoost, this will take a while...")
        xgbreg = XGBRegressor(random_state=42)

        param_grid = { 
            'n_estimators': [10, 20, 100, 200, 500],
            'max_depth' : [2, 3, 5, 10],
            'objective': ['reg:squarederror'],
            'colsample_bytree': [0.3, 0.6, 0.9],
            'learning_rate': [2e-5, 2e-4, 2e-3, 2e-2, 2e-1]
        }
        cv_xgb = RandomizedSearchCV(
            estimator=xgbreg, 
            param_distributions=param_grid,
            scoring='neg_mean_squared_error',
            n_iter=100, 
            cv=5)
        cv_xgb.fit(X_train, y_train)
        evaluate(
            model=cv_xgb.best_estimator_, 
            model_name=f'xgboost')
        print("...Ran XGBoost!")

        # check model performances
        print("Comparing models...")
        perf_df = pd.DataFrame(model_performances)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=perf_df.sort_values(
                by='rmse', 
                ascending=False), 
            y='model', 
            x='rmse', 
            marker='s', 
            hue='split', 
            palette=['darkorange', 'grey', 'darkred'])
        plt.savefig(f'plots/model_perfs_{idx}.png')
        print("...Model performances saved!")

def main():
    csvs = [
        'cesd_total', 
        'gad_total',
        'inq_perceivedburden',
        'inq_thwartedbelong',
        'upps_total']
    dataframes = load_csvs_to_dfs(csvs)
    loop_through_dfs(dataframes)

if __name__ == "__main__":
    main()