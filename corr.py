import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.patches as patches

### data loading
data_raman = pd.read_csv("data/raman_data_prisma.csv", header=0, index_col=0, sep='\,') #raw data (bkg substracted)
data_xrd = pd.read_csv("data/xrd_data.csv", header=0, index_col=0, sep='\,') #raw data (bkg substracted)

### data selection
shifts = np.round(data_raman.iloc[:, 0].to_numpy()[::-1],1)
specs = data_raman.iloc[:, 1:][::-1]
specs.columns = specs.columns.astype(int)

theta = data_xrd['2th'].to_numpy()[1:-1]
ints = data_xrd.iloc[1:-1, 1:]
ints.columns = specs.columns.astype(int)

### Raw data (Raman and XRD)
labels = np.concatenate((shifts, theta))
dfs = pd.concat([specs, ints], axis=0)
dfs.index = labels
X = dfs.T

### Correlation matrix
correlation_matrix = X.corr(method='pearson')

### Data selection for the heatmap
num_raman_features = specs.shape[0]
num_xrd_features = ints.shape[0]

num_300_raman = np.where(shifts == 300.8)[0][0]
num_24_theta = np.where(theta == 24)[0][0]
num_28_theta = np.where(theta == 28)[0][0]
num_32_theta = np.where(theta == 32)[0][0]
theta_cut = theta[num_24_theta:num_32_theta]

raman_xrd_correlation = correlation_matrix.iloc[num_raman_features:num_raman_features+num_xrd_features, :num_raman_features]
raman_xrd_correlation_cut = correlation_matrix.iloc[num_raman_features+num_24_theta:num_raman_features+num_32_theta, :num_raman_features]

# Define which ticks you want to show (e.g., every 5th feature)

rounded_raman_ticks = np.arange(100, 701, 50)
rounded_xrd_ticks = np.arange(15, 85, 5)
rounded_xrd_ticks_cut = np.arange(24, 32.1, 1)

# Define the positions of the ticks
xtick_positions = [np.argmin(np.abs(shifts - rt)) for rt in rounded_raman_ticks]
ytick_positions = [np.where(theta == rt)[0][0] for rt in rounded_xrd_ticks]
ytick_positions_cut = [np.where(theta == rt)[0][0] - 538 for rt in rounded_xrd_ticks_cut]

# Determine the absolute max value
vmax = max(raman_xrd_correlation_cut.max().max(), abs(raman_xrd_correlation_cut.min().min()))

# Plot heatmap and customize it
fig,ax = plt.subplots(figsize=(12, 10))
heatmap = sns.heatmap(raman_xrd_correlation, annot=False, cmap="coolwarm", fmt=".2f", 
                      xticklabels=rounded_raman_ticks, yticklabels=rounded_xrd_ticks, 
                      vmin=-vmax, vmax=vmax, cbar_kws={'pad': 0.025})
start, stop = 0., 1.0  
cmap = plt.get_cmap('coolwarm')  
cmap_trunc = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=start, b=stop), cmap(np.linspace(start, stop, 1000)))
cbar = heatmap.collections[0].colorbar
cbar.outline.set_edgecolor('black')
rect = patches.Rectangle((0,0), 1, 1, linewidth=2.5, edgecolor='black', facecolor='none', transform=ax.transAxes)
ax.add_patch(rect)
plt.title('Correlation Heatmap: Raman vs XRD') 
plt.xlabel(r'Raman Shifts, cm$^{-1}$')
plt.ylabel(r'2θ, °')
ax.set_xticks(ticks=xtick_positions, labels=rounded_raman_ticks, rotation=90)
ax.set_yticks(ticks=ytick_positions, labels=rounded_xrd_ticks, rotation=0)
plt.show()