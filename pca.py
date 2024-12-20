import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.decomposition import PCA

### data loading
Q_pred = pd.read_csv('data/pred_Q.csv', header=0, sep='\,')
data_raman = pd.read_csv("data/raman_data_prisma.csv", header=0, index_col=0, sep='\,')
data_xrd = pd.read_csv("data/xrd_data.csv", header=0, index_col=0, sep='\,')
df = pd.read_csv("data/df_109.csv", header=0, index_col=0, sep='\,')

### data selection
C = df.iloc[:,2:5]

shifts = np.round(data_raman.iloc[:, 0].to_numpy()[::-1],1)
specs = data_raman.iloc[:, 1:][::-1]

theta = data_xrd['2th'].to_numpy()[1:-1]
ints = data_xrd.iloc[1:-1, 1:] #/data_xrd.iloc[1:-1, 1:].max().max()

### data processing

specs_cent = specs.T - specs.T.mean(axis=0)
specs_norm_glob = (specs_cent - specs_cent.min().min())/(specs_cent.max().max() - specs_cent.min().min())
specs_norm_spec = specs_cent.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

specs_glob = (specs.T - specs.T.min().min())/(specs.T.max().max() - specs.T.min().min())

ints_cent = ints.T - ints.T.mean(axis=0)
ints_norm_glob = (ints_cent - ints_cent.min().min())/(ints_cent.max().max() - ints_cent.min().min())
ints_norm_spec = ints_cent.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

ints_glob = (ints.T - ints.T.min().min())/(ints.T.max().max() - ints.T.min().min())


labels = np.concatenate((shifts, theta))
df_norm_glob = pd.concat([specs_norm_glob, ints_norm_glob], axis=1)
df_norm_glob.columns = labels
df_norm_spec = pd.concat([specs_norm_spec, ints_norm_spec], axis=1)
df_norm_spec.columns = labels

### PCA Raman only glob
pca_raman_glob = PCA(n_components=3)
raman_pc_glob = pca_raman_glob.fit_transform(specs_norm_glob)
raman_df_glob = pd.DataFrame(data = raman_pc_glob, columns = ['PC 1', 'PC 2', 'PC 3'])
raman_ratio_glob = pca_raman_glob.explained_variance_ratio_
print(raman_ratio_glob, sum(raman_ratio_glob))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(raman_df_glob['PC 1'], raman_df_glob['PC 2'], raman_df_glob['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(raman_ratio_glob[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(raman_ratio_glob[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(raman_ratio_glob[2]*100, 1)} %)')
plt.title('PCA of Raman data')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=-145)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(raman_df_glob['PC 1'], raman_df_glob['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(raman_df_glob[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(raman_df_glob[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the Raman data', fontsize=18)
plt.show()

### PCA Raman only spec
pca_raman_spec = PCA(n_components=3)
raman_pc_spec = pca_raman_spec.fit_transform(specs_norm_spec)
raman_df_spec = pd.DataFrame(data = raman_pc_spec, columns = ['PC 1', 'PC 2', 'PC 3'])
raman_ratio_spec = pca_raman_spec.explained_variance_ratio_
print(raman_ratio_spec, sum(raman_ratio_spec))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(raman_df_spec['PC 1'], raman_df_spec['PC 2'], raman_df_spec['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(raman_ratio_spec[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(raman_ratio_spec[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(raman_ratio_spec[2]*100, 1)} %)')
plt.title('PCA of Raman data')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=50)
plt.show()

fig.savefig(os.path.join(base_path, "KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/3_pca_raman_scaled.svg"), format='svg', dpi=300, transparent=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(raman_df_spec['PC 1'], raman_df_spec['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(raman_ratio_spec[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(raman_ratio_spec[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the Raman data', fontsize=18)
plt.show()

### PC Raman spectra
pca_raman_spectra = pca_raman_spec.components_
pca_raman_spectra = pca_raman_glob.components_

fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
colors = ['r', 'g', 'b']
for i in range(pca_raman_spectra.shape[0]):
    plt.plot(shifts, pca_raman_spectra[i, :320], alpha=0.7, label=f'PC{i+1} ({round(raman_ratio_spec[i]*100, 1)} %)', color=colors[i], zorder=3-i)
    # plt.plot(shifts, pca_raman_spectra[i, :320] + specs_glob.mean(axis=0).to_numpy(), alpha=0.7, label=f'PC{i+1}', color=colors[i], zorder=3-i)
ax.tick_params(which='both', labelsize=16, width=1.5)
ax.set_xlim(100, 700)
ax.set_ylim(-0.25, 0.25)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_yticks(np.arange(-0.25,0.26,0.05),minor=True)
#ax.set_yticks([])
#ax.set_yticklabels([])
plt.xlabel('Raman shift, cm$^{-1}$', fontsize=20)
plt.ylabel('Intensity, a.u.', fontsize=20)
plt.legend()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
#plt.title('Raman Principal Components from multidomain PCA', fontsize=16)
plt.title('Raman Principal Components', fontsize=16)
plt.subplots_adjust(bottom=0.15)
plt.show()

fig.savefig(os.path.join(base_path, "KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/3_pcs_raman_scaled.svg"), format='svg', dpi=300, transparent=True)


### PCA XRD only glob
pca_xrd_glob = PCA(n_components=3)
xrd_pc_glob = pca_xrd_glob.fit_transform(ints_norm_glob)
xrd_df_glob = pd.DataFrame(data = xrd_pc_glob, columns = ['PC 1', 'PC 2', 'PC 3'])
xrd_ratio_glob = pca_xrd_glob.explained_variance_ratio_
print(xrd_ratio_glob, sum(xrd_ratio_glob))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xrd_df_glob['PC 1'], xrd_df_glob['PC 2'], xrd_df_glob['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(xrd_ratio_glob[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(xrd_ratio_glob[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(xrd_ratio_glob[2]*100, 1)} %)')
plt.title('PCA of XRD data')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=-145)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(xrd_df_glob['PC 1'], xrd_df_glob['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(xrd_df_glob[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(xrd_df_glob[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the XRD data', fontsize=18)
plt.show()

### PCA XRD only spec
pca_xrd_spec = PCA(n_components=3)
xrd_pc_spec = pca_xrd_spec.fit_transform(ints_norm_spec)
xrd_df_spec = pd.DataFrame(data = xrd_pc_spec, columns = ['PC 1', 'PC 2', 'PC 3'])
xrd_ratio_spec = pca_xrd_spec.explained_variance_ratio_
print(xrd_ratio_spec, sum(xrd_ratio_spec))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xrd_df_spec['PC 1'], xrd_df_spec['PC 2'], xrd_df_spec['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(xrd_ratio_spec[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(xrd_ratio_spec[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(xrd_ratio_spec[2]*100, 1)} %)')
plt.title('PCA of XRD data')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=-135)
plt.show()

fig.savefig(os.path.join(base_path, "KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/3_pca_xrd_scaled.svg"), format='svg', dpi=300, transparent=True)


fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(xrd_df_spec['PC 1'], xrd_df_spec['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(xrd_df_spec[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(xrd_df_spec[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the XRD data', fontsize=18)
plt.show()


### PC XRD spectra
pca_xrd_spectra = pca_xrd_spec.components_

fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
colors = ['r', 'g', 'b']
for i in range(pca_xrd_spectra.shape[0]):
    plt.plot(theta, pca_xrd_spectra[i, :], alpha=0.7, label=f'PC{i+1} ({round(xrd_ratio_spec[i]*100, 1)} %)', color=colors[i], zorder=3-i)
    #plt.plot(theta, pca_xrd_spectra[i, :]+ ints_glob.mean(axis=0).to_numpy(), alpha=0.7, label=f'PC{i+1} ({round(xrd_ratio_spec[i]*100, 1)} %)', color=colors[i], zorder=3-i)
ax.tick_params(which='both', labelsize=16, width=1.5)
ax.set_xlim(14, 84)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.arange(-0.1,0.11,0.01),minor=True)
ax.set_yticks(np.arange(-0.1,0.11,0.05),minor=False)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.set_yticks([])
#ax.set_yticklabels([])
plt.xlabel('2Θ, °', fontsize=20)
plt.ylabel('Intensity, a.u.', fontsize=20)
plt.legend()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('XRD Principal Components', fontsize=16)
plt.subplots_adjust(bottom=0.15)
plt.show()

fig.savefig(os.path.join(base_path, "KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/3_pcs_xrd_scaled.svg"), format='svg', dpi=300, transparent=True)



### Combined PCA glob
pca_glob = PCA(n_components=3)
pc_glob = pca_glob.fit_transform(df_norm_glob)
df_glob = pd.DataFrame(data = pc_glob, columns = ['PC 1', 'PC 2', 'PC 3'])
ratio_glob = pca_glob.explained_variance_ratio_
print(ratio_glob, sum(ratio_glob))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_glob['PC 1'], df_glob['PC 2'], df_glob['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(ratio_glob[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(ratio_glob[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(ratio_glob[2]*100, 1)} %)')
plt.title('PCA of multidomain data (Raman and XRD)')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=-145)
plt.show()

fig.savefig(os.path.join(base_path, "KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/3_pca_multidomain.svg"), format='svg', dpi=300, transparent=True)


fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_glob['PC 1'], df_glob['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(df_glob[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(df_glob[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the Raman data', fontsize=18)
plt.show()



### Raman PCA only spec
pca_spec = PCA(n_components=3)
pc_spec = pca_spec.fit_transform(df_norm_spec)
df_spec = pd.DataFrame(data = pc_spec, columns = ['PC 1', 'PC 2', 'PC 3'])
ratio_spec = pca_spec.explained_variance_ratio_
print(ratio_spec, sum(ratio_spec))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_spec['PC 1'], df_spec['PC 2'], df_spec['PC 3'], c=C, s=10)
ax.set_xlabel(f'PC 1 ({round(ratio_spec[0]*100, 1)} %)')
ax.set_ylabel(f'PC 2 ({round(ratio_spec[1]*100, 1)} %)')
ax.set_zlabel(f'PC 3 ({round(ratio_spec[2]*100, 1)} %)')
plt.title('PCA of XRD data')
#ax.view_init(elev=25, azim=-150)
ax.view_init(elev=30, azim=-145)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_spec['PC 1'], df_spec['PC 2'], c=C, s=20, alpha=0.7)
ax.set_xlabel(f'PC 1 ({round(df_spec[0]*100, 1)} %)', fontsize=20)
ax.set_ylabel(f'PC 2 ({round(df_spec[1]*100, 1)} %)', fontsize=20)
ax.tick_params(which='both', labelsize=16, width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.title('PCA of the Raman data', fontsize=18)
plt.show()

### Data PCs save

df_pca = pd.concat([raman_df_spec['PC 1'], raman_df_spec['PC 2'], xrd_df_spec['PC 1'], xrd_df_spec['PC 2']], axis=1)
df_pca.columns = ['Raman PC 1', 'Raman PC 2', 'XRD PC 1', 'XRD PC 2']
df_pca.to_csv(os.path.join(base_path, r"KIT-HIU/Science/Experiments/Correlation analysis/Raman vs XRD/pcs_2_raman_xrd.csv"), index=False)
 
