import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

def train_model(X_train, y_train):
    # Train a RandomForest model with hyperparameter tuning
    rf = RandomForestRegressor(random_state=1337)
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [2, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Evaluate model performance using cross-validation and test sets
    kf = KFold(n_splits=5, shuffle=True, random_state=1337)

    # Cross-validation metrics
    r2_scores_train = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    mae_scores_train = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    rmse_scores_train = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    # Cross-validation predictions for train
    model.fit(X_train, y_train)
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=kf)

    # Test set metrics
    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

    metrics = {
        'cv': {
            'R2': (np.mean(r2_scores_train), np.std(r2_scores_train)),
            'MAE': (np.mean(mae_scores_train), np.std(mae_scores_train)),
            'RMSE': (np.mean(rmse_scores_train), np.std(rmse_scores_train))
        },
        'test': {'R2': r2_test, 'MAE': mae_test, 'RMSE': rmse_test}
    }
    return y_train_pred, y_test_pred, metrics

def plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, metrics, title):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(9.9, 6.6))
    ax.scatter(y_train, y_train_pred, color='green', alpha=0.5, label=f'Train (n={len(y_train)})')
    ax.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label=f'Test (n={len(y_test)})')
    ax.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
            [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], '--r', linewidth=2)
    ax.set_xlim(1550, 3050)
    ax.set_ylim(1550, 3050)
    ax.set_xticks(np.arange(1600,3100,100),minor=True)
    ax.set_yticks(np.arange(1600,3100,100),minor=True)
    ax.set_xticks(np.arange(1600,3100,200),minor=False)
    ax.set_yticks(np.arange(1600,3100,200),minor=False)
    ax.set_xlabel('Actual capacities, mAh/g')
    ax.set_ylabel('Predicted capacities, mAh/g')
    ax.legend(loc='lower right')
    textstr = '\n'.join((
        'Model: Random Forest Regressor, 5-fold CV',
        f'Train: $R^2$ = {metrics["cv"]["R2"][0]:.3f} ± {metrics["cv"]["R2"][1]:.3f}, '
        f'MAE = {metrics["cv"]["MAE"][0]:.1f} ± {metrics["cv"]["MAE"][1]:.1f}, '
        f'RMSE = {metrics["cv"]["RMSE"][0]:.1f} ± {metrics["cv"]["RMSE"][1]:.1f}',
        f'Test: $R^2$ = {metrics["test"]["R2"]:.3f}, '
        f'MAE = {metrics["test"]["MAE"]:.1f}, '
        f'RMSE = {metrics["test"]["RMSE"]:.1f}'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='lightgrey')
    ax.text(1600, 3000, textstr, verticalalignment='top', horizontalalignment='left', bbox=props)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def plot_predicted_distribution(y_test, y_test_pred, bin_size, title):
    # Plot distribution of predicted and actual values.
    bins = np.arange(1600, 2900 + bin_size, bin_size)
    fig, ax = plt.subplots(figsize=(9.9, 6.6))
    sns.kdeplot(y_test, color='red', label='KDE Actual', shade=False)
    sns.histplot(y_test, color='red', label='Histogram Actual', kde=False, stat='density',
                bins=bins, alpha=0.5, cumulative=False, edgecolor=None)
    sns.kdeplot(y_test_pred, color='blue', label='KDE Predicted', shade=False)
    sns.histplot(y_test_pred, color='blue', label='Histogram Predicted', kde=False, stat='density', 
                bins=bins, alpha=0.5, cumulative=False, edgecolor=None)
    ax.set_xlabel('Specific capacities, mAh/g')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(1550, 3050)
    ax.set_xticks(np.arange(1600, 3100, 100), minor=True)
    ax.set_xticks(np.arange(1600, 3100, 200), minor=False)
    ax.set_box_aspect(1) 
    ax.set_title(title)
    plt.show()

def plot_feature_importances_rank(model, feature_names, top_feat=10, horizontal=False, title="Feature Importances"):
    # Plot feature importances with optional orientation
    feature_importances = model.feature_importances_
    feature_importances_df_ = pd.DataFrame({
        'Feature': [str(name) for name in feature_names],
        'Importance': feature_importances
    })
    feature_importances_df = feature_importances_df_.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['green' if 'cubic' in feature else 'purple' if 'D$_' in feature or 'A$_' in feature else 'red' if 'l,' in feature or 'd,' in feature or 'at. %' in feature else 'orange' if 'PC' in feature else 'blue' for feature in feature_importances_df['Feature'].head(top_feat)]    
    if horizontal:
        plt.barh(feature_importances_df['Feature'].head(top_feat)[::-1], 100 * feature_importances_df['Importance'].head(top_feat)[::-1], align='center', color=colors[::-1])
        plt.xlabel("Importance, %")
        plt.ylabel("Features")
    else:
        plt.bar(feature_importances_df['Feature'].head(top_feat), 100 * feature_importances_df['Importance'].head(top_feat), align='center', color=colors)
        plt.xlabel("Features")
        plt.ylabel("Importance, %")
    patches = []
    if 'green' in colors:
        green_patch = mpatches.Patch(color='green', label='XRD feature')
        patches.append(green_patch)
    if 'blue' in colors:
        blue_patch = mpatches.Patch(color='blue', label='Raman feature')
        patches.append(blue_patch)
    if 'purple' in colors:
        purple_patch = mpatches.Patch(color='purple', label='Wavelets feature')
        patches.append(purple_patch)
    if 'red' in colors:
        red_patch = mpatches.Patch(color='red', label='µ-XRF feature')
        patches.append(red_patch)
    if 'orange' in colors:
        orange_patch = mpatches.Patch(color='orange', label='PCA feature')
        patches.append(orange_patch)
    print(patches)
    ax.legend(handles=patches, loc='lower right')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names, domain):
    # Plot feature importances for a specified domain (Raman or XRD)
    feature_importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    fig, ax = plt.subplots(figsize=(9, 6))
    if domain == 'Raman':
        plt.bar(feature_importances_df['Feature'], 100 * feature_importances_df['Importance'], color='b', width=1.9)
        plt.xlabel('Raman shift, cm$^{-1}$', fontsize=15)
        plt.ylabel('Importance, %', fontsize=15)
        plt.title('Raman Importances on the Multimodal Regression Model (Random Forest)')
        ax.set_xlim(100, 700)
    elif domain == 'XRD':
        plt.bar(feature_importances_df['Feature'], 100 * feature_importances_df['Importance'], color='b', width=0.2)
        plt.xlabel('2Θ, °', fontsize=15)
        plt.ylabel('Importance, %', fontsize=15)
        plt.title('XRD Importances on the Multimodal Regression Model (Random Forest)')
        ax.set_xlim(14, 84)
        ax.set_ylim(0, 0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    plt.show()

def plot_shap_values_rank(model, X_train, top_feat=10):
    # Plot SHAP values for the model and training data
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_values_df = pd.DataFrame(shap_values, columns=X_train.columns)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(explainer(X_train), max_display=top_feat, show=False)
    plt.tight_layout()
    plt.show()

def plot_shap_values(model, X_train, domain):
    # Plot SHAP values for the model and training data
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_values_df = pd.DataFrame(shap_values, columns=X_train.columns)
    mean_shap_values = shap_values_df.mean(axis=0)
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    correlations = []
    for feature in X_train.columns:
        corr = np.corrcoef(X_train[feature], shap_values_df[feature])[0, 1]
        correlations.append(corr)
    correlations = [0 if np.isnan(corr) else corr for corr in correlations]
    signed_importance = mean_abs_shap_values * np.sign(correlations)

    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean_Abs_SHAP': mean_abs_shap_values,
        'Mean_SHAP': mean_shap_values,
        'Signed_importance': signed_importance
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = feature_importance_df['Signed_importance'].apply(lambda x: 'red' if x > 0 else 'blue')
    if domain == 'Raman':
        plt.bar(feature_importance_df['Feature'], abs(feature_importance_df['Signed_importance']), color=colors, width=2)
        ax.set_xlim(100, 700)
    elif domain == 'XRD':
        plt.bar(feature_importance_df['Feature'], abs(feature_importance_df['Signed_importance']), color=colors, width=0.2)
        ax.set_xlim(14, 84)
        ax.set_ylim(0, 3)
    plt.xlabel(f'{domain} Feature', fontsize=15)
    plt.ylabel('Mean (|SHAP value|)', fontsize=15)
    plt.title(f'SHAP Analysis for {domain} Features')
    plt.tight_layout()
    plt.show()


### data loading
Q_pred = pd.read_csv('data/pred_Q.csv', header=0, index_col=0, sep='\,')
df = pd.read_csv("data/df_109.csv", header=0, index_col=0, sep='\,') # xrf
data_raman = pd.read_csv("data/raman_data_prisma.csv", header=0, index_col=0, sep='\,') # raw data (bkg substracted)
data_xrd = pd.read_csv("data/xrd_data.csv", header=0, index_col=0, sep='\,') # raw data (bkg substracted)
pca = pd.read_csv("data/pcs_3_raman_xrd.csv", header=0, index_col=0, sep='\,') # pca data
fit_raman = pd.read_csv("data/raman_fit_prisma.csv", header=0, index_col=0, sep='\,') # fitted data
fit_xrd = pd.read_csv("data/xrd_fit.csv", header=0, index_col=0, sep='\,') # fitted data
wt = pd.read_csv("data/wavelets.csv", header=0, index_col=0, sep='\,') # wavelets

### data selection
C, d = df.iloc[:,2:5], df.iloc[:,-1] # composition and thickness

shifts = np.round(data_raman.iloc[:, 0].to_numpy()[::-1],1)
specs = data_raman.iloc[:, 1:][::-1]
specs.columns = specs.columns.astype(int)

theta = data_xrd['2th'].to_numpy()[1:-1]
ints = data_xrd.iloc[1:-1, 1:]
ints.columns = specs.columns.astype(int)

### ml data selection (uncomment and run the desired feature dataset)
y = Q_pred['Q']
### Raw data (Raman)
#X = specs.T
#X.columns = shifts
### Raw data (XRD)
#X = ints.T
#X.columns = theta
### XRF (composition and thickness)
#X = pd.concat([C, d], axis=1)
### Wavelets (Raman)
#X = wt
### Fitted data (Raman)
#X = fit_raman.iloc[:, 1:]
### Fitted data (XRD)
#X = fit_xrd
### PCA (Raman)
#X = pca.iloc[:, :3]
### PCA (XRD)
#X = pca.iloc[:, 3:]
### Raw data combined (Raman and XRD)
labels = np.concatenate((shifts, theta))
dfs = pd.concat([specs.T, ints.T], axis=1)
dfs.columns = labels
X = dfs
### Fitted data combined (Raman and XRD)
#X_raman = fit_raman.iloc[:, 1:]
#X_xrd = fit_xrd
#X = pd.concat([X_raman, X_xrd], axis=1)
### Wavelets and fitted XRD combined (Raman and XRD)
#X = pd.concat([wt, fit_xrd], axis=1)
### PCA combined (Raman and XRD)
#X = pca
### Combined Raman (raw), XRD (raw) and µ-XRF
#X = pd.concat([specs.T, ints.T, C, d], axis=1)

### train-test split
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13372)

### model training
best_model, best_params = train_model(X_train, y_train)
print(f'Best Parameters: {best_params}')

### model evaluation
y_train_pred, y_test_pred, metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
print(f"Cross-Validation Metrics: {metrics['cv']}")
print(f"Test Metrics: {metrics['test']}")

### plot actual vs predicted
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, metrics, 
                         title="Multidomain regression (PCA on Raman and XRD)")

### plot predicted distribution
plot_predicted_distribution(y_test, y_test_pred, bin_size=100,
                            title="Density of predicted vs actual capacities (Test dataset)")
 
### plot feature importances ranked
plot_feature_importances_rank(best_model, X.columns, top_feat=20, horizontal=True, title="Feature Importances")

### plot feature importances for a specific feature (raw data)
plot_feature_importances(best_model, X.columns, domain="Raman")
plot_feature_importances(best_model, X.columns, domain="XRD")

### plot SHAP values ranked
plot_shap_values_rank(best_model, X_train, top_feat=10)

### plot SHAP values for a specific feature (raw data)
plot_shap_values(best_model, X_train, domain="Raman")
plot_shap_values(best_model, X_train, domain="XRD")
