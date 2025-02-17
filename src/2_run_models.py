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
    # Loads files into a list of dataframes.

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
    # Plots a histogram of the outcome variable.

    # extract the last column
    last_column_name = df.columns[-1]
    
    # create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[last_column_name], bins=50)
    plt.title(f'Histogram of {last_column_name}')
    plt.ylabel(last_column_name)
    plt.grid(True)

    plt.savefig(f'plots/hist_{last_column_name}.png', bbox_inches='tight')

def plot_correlations(df: pd.DataFrame):
    # Creates a plot of the correlations between all the predictors. 

    last_column_name = df.columns[-1]
    sns.clustermap(df.corr(), cmap='viridis')
    plt.savefig(f'plots/corr_{last_column_name}.png', bbox_inches='tight')

def create_splits(df:pd.DataFrame):
    # Creates train, validation and test splits of size 70%-15%-15%.

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
    # Identifies variable types automatically for later standardization purposes. 

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
    # Standardizes continuous variables and passes through dummy ones.

    continuous_cols, dummy_cols = identify_variable_types(X_train)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols),
            ('dummy', 'passthrough', dummy_cols) 
        ]
    )
    X_train = preprocessor.fit_transform(X_train) # fit to train set and transform
    X_val = preprocessor.transform(X_val) # transform only
    X_test = preprocessor.transform(X_test) # transform only

    return X_train, X_val, X_test

def loop_through_dfs(list_of_dfs):
    # Loops through the dataframes and runs all the models

    def run_on_splits(func):
        # Function from class material. 
        def _run_loop(*args, **kwargs):
            for x,y,nsplit in zip([X_train, X_val, X_test],
                                [y_train, y_val, y_test],
                                ['train', 'val', 'test']):
                func(*args, X=x, y=y, nsplit=nsplit, **kwargs)
        return _run_loop

    @run_on_splits
    def evaluate(model, X, y, nsplit, model_name, constant_value=None):
        # Function from class material. 
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

        # perform PCA
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print("...Preparations completed!")
        
        model_performances = []

        # run null model
        print("Running null model...")
        evaluate(model=None, model_name='dummy', constant_value=y_train.mean())
        print("...Ran null model!")

        # run plain linear regression
        print("Running linear regression...")
        reg = LinearRegression().fit(X_train, y_train)
        evaluate(model=reg, model_name='linear')
        print("...Ran linear regression!")

        # run ridge and lasso versions
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
                evaluate(model=reg, model_name=f'{id}-alpha-{alpha}')
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
        cv_rfr = RandomizedSearchCV(estimator=rfreg, 
                                    param_distributions=param_grid,
                                    scoring='neg_mean_squared_error', 
                                    n_iter=100, 
                                    cv=5)
        cv_rfr.fit(X_train, y_train)
        evaluate(model=cv_rfr.best_estimator_, model_name=f'random-forest')
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
        cv_xgb = RandomizedSearchCV(estimator=xgbreg, 
                                    param_distributions=param_grid,
                                    scoring='neg_mean_squared_error',
                                    n_iter=100, 
                                    cv=5)
        cv_xgb.fit(X_train, y_train)
        evaluate(model=cv_xgb.best_estimator_, model_name=f'xgboost')
        print("...Ran XGBoost!")

        # check model performances
        print("Comparing models...")
        perf_df = pd.DataFrame(model_performances)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=perf_df.sort_values(by='rmse', ascending=False), 
                        y='model', 
                        x='rmse', 
                        marker='s', 
                        hue='split', palette=['darkorange', 'grey', 'darkred'])
        plt.savefig(f'plots/model_perfs_{idx}.png', bbox_inches='tight')
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