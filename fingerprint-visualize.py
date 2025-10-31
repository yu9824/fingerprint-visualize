# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3.8.13 ('fp-visualize')
#     language: python
#     name: python3
# ---

# %%
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib import request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import rdkit
import seaborn as sns
import shap
from boruta import BorutaPy
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import (
    GetHashedMorganFingerprint,
    GetMorganFingerprintAsBitVect,
)
from rdkit.Chem.PandasTools import LoadSDF
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from tqdm.notebook import tqdm, trange

mpl.__version__


# %%
SEED = 334


# %%
rdkit.rdBase.rdkitVersion


# %%
url_sdf = "http://datachemeng.wp.xdomain.jp/wp-content/uploads/2017/04/logSdataset1290_2d.sdf"
# download and create cache
dirpath_cache = os.path.abspath("./cache")
if not os.path.isdir(dirpath_cache):
    os.mkdir(dirpath_cache)
fpath_sdf_cached = os.path.join(dirpath_cache, os.path.basename(url_sdf))
if not os.path.isfile(fpath_sdf_cached):
    with (
        request.urlopen(url_sdf) as response,
        open(fpath_sdf_cached, mode="w", encoding="utf-8") as out_file,
    ):
        data = response.read().decode("utf-8")
        out_file.write(data)


# %%
df: pd.DataFrame = LoadSDF(fpath_sdf_cached)
df.head()


# %%
target_col = "logS"
col_smiles = "SMILES"


# %% [markdown]
# SMILESに変換して重複を削除してキーにする。

# %%
df[col_smiles] = df["ROMol"].apply(Chem.MolToSmiles)
df[target_col] = df[target_col].astype(float)

df_extracted = pd.concat(
    [
        df[[col_smiles, target_col]].groupby(col_smiles).mean(),
        df[[col_smiles, "ROMol"]].groupby(col_smiles).first(),
    ],
    axis=1,
)
df_extracted.head()


# %%
bitinfos: List[Dict[int, Tuple]] = []
fps: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect] = []
for _mol in df_extracted["ROMol"]:
    bitinfo: Dict[int, Tuple] = {}
    fp = np.array(
        GetMorganFingerprintAsBitVect(
            _mol, radius=2, nBits=1024, bitInfo=bitinfo
        ),
        dtype=int,
    )
    bitinfos.append(bitinfo)
    fps.append(fp)
X: pd.DataFrame = pd.DataFrame(np.vstack(fps), index=df_extracted.index)
X.shape


# %%
y = df_extracted[target_col]
y.shape


# %%
vselector = VarianceThreshold(threshold=0.0)
vselector.fit(X, y)
X_vselected: pd.DataFrame = X.iloc[:, vselector.get_support()]
X_vselected.shape


# %%
feature_selector = BorutaPy(
    RandomForestRegressor(n_jobs=-1),
    n_estimators="auto",
    verbose=0,
    perc=80,
    random_state=SEED,
    max_iter=100,
)
feature_selector.fit(X_vselected.values, y)
X_selected: pd.DataFrame = X_vselected.iloc[:, feature_selector.support_]
X_selected.shape


# %%
estimator = RandomForestRegressor(random_state=334, n_jobs=-1)
estimator.fit(X_selected, y)
y_oof = cross_val_predict(
    estimator,
    X_selected,
    y,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=-1,
)


# %%
# true vs pred
def plot_true_vs_pred(
    y: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> None:
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(facecolor="w")

    _tmp = np.hstack([np.array(y).ravel(), np.array(y_pred).ravel()])
    _range = (min(_tmp), max(_tmp))
    alpha = 0.05
    offset = (max(_range) - min(_range)) * alpha
    _plot_range = (min(_tmp) - offset, max(_tmp) + offset)

    ax.plot(*[_plot_range] * 2, color="gray", zorder=1)
    ax.scatter(y, y_pred, marker="o", s=10, alpha=0.5)

    ax.set_xlabel("$y_{true}$")
    ax.set_ylabel("$y_{pred}$")

    ax.set_xlim(_plot_range)
    ax.set_ylim(_plot_range)

    ax.text(
        min(_range),
        max(_range),
        f"$R^2={r2_score(y, y_pred):.2g}$\nRMSE$={mean_squared_error(y, y_pred, squared=False):.2f}$",
        ha="left",
        va="top",
    )

    ax.set_aspect("equal")
    fig.tight_layout()



# %%
plot_true_vs_pred(y, y_oof)


# %%
estimator = clone(estimator).fit(X_selected, y)


# %%
y_pred = estimator.predict(X_selected)
plot_true_vs_pred(y, y_pred)


# %%
# explainer = shap.LinearExplainer(model, X_selected)
explainer = shap.TreeExplainer(estimator, X_selected)
shap_values = pd.DataFrame(
    explainer.shap_values(X_selected),
    index=X_selected.index,
    columns=X_selected.columns,
)


# %%
plt.rcParams.update(plt.rcParamsDefault)
shap.summary_plot(shap_values.values, X_selected, plot_type="dot")


# %%
# shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values[0], matplotlib=True)


# %%
shap_values.shape


# %% [markdown]
#
# $$
# A_i = \sum_{n=1}^{N} \left( C_n \times \frac{1}{f_n} \times \frac{1}{x_n} \right)
# $$
#
# * $C_n$: 各フィンガープリントの寄与
# * $f_n$: 分子中に含まれる各部分構造の数 ($n = 1, 2, \ldots, N$)
# * $x_n$: 各部分構造に含まれる原子数

# %%
selected_bit_num = X_selected.columns.tolist()

# i = 1
# for i in trange(X_selected.shape[0]):    # samples
# for i in trange(10):    # samples
rng_ = np.random.RandomState(SEED)
for i in tqdm(rng_.randint(0, X_selected.shape[0], 5)):
    bitinfo = bitinfos[i]
    # ratio_contribution = model.coef_[i] # C_n
    ratio_contributions: pd.Series = shap_values.iloc[i]  # C_n
    print(ratio_contributions)
    mol = df_extracted["ROMol"][i]

    bit_list = list(set(bitinfo.keys()) & set(selected_bit_num))

    importance_atoms = np.zeros(mol.GetNumAtoms(), dtype=float)
    for _bit in bit_list:
        n_substructure = len(bitinfo[_bit])
        ratio_contribution = ratio_contributions[_bit]
        for i_atom, radius in bitinfo[_bit]:
            if radius == 0:
                n_atom_in_substructure = 1
                importance_atoms[i_atom] += (
                    ratio_contribution
                    / n_atom_in_substructure
                    / n_substructure
                )
            else:
                atom_map = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol, radius=radius, rootedAtAtom=i_atom
                )
                submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)

                n_atom_in_substructure = len(submol.GetAtoms())
                for j_atom in atom_map:
                    importance_atoms[j_atom] += (
                        ratio_contribution / n_atom_in_substructure
                    )

    # scaling
    importance_atoms_scaled = (
        importance_atoms / abs(importance_atoms).max() * 0.5
    )
    importance_atoms_scaled

    atom_colors = {
        i: (1, 1 - importance_atoms_scaled[i], 1 - importance_atoms_scaled[i])
        if importance_atoms_scaled[i] > 0
        else (
            1 + importance_atoms_scaled[i],
            1 + importance_atoms_scaled[i],
            1,
        )
        for i in range(len(importance_atoms_scaled))
    }

    view = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    tm = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    view.DrawMolecule(
        tm,
        highlightAtoms=atom_colors.keys(),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
        legend=f"y_true: {y[i]:.2f}  y_pred: {y_pred[i]:.2f}",
    )
    view.FinishDrawing()
    svg = view.GetDrawingText()
    with open("highlighted_sample.svg", "w") as f:
        f.write(svg)
    display(SVG(svg))


# %%
selected_bit_num = X_selected.columns.tolist()

# i = 1
# for i in trange(X_selected.shape[0]):    # samples
# for i in trange(10):    # samples
rng_ = np.random.RandomState(SEED)
for i in tqdm(rng_.randint(0, X_selected.shape[0], 5)):
    bitinfo = bitinfos[i]
    # ratio_contribution = model.coef_[i] # C_n
    ratio_contributions: pd.Series = shap_values.iloc[i]  # C_n
    mol = df_extracted["ROMol"][i]


def visualize_importance_atoms():
    bit_list = list(set(bitinfo.keys()) & set(selected_bit_num))

    importance_atoms = np.zeros(mol.GetNumAtoms(), dtype=float)
    for _bit in bit_list:
        n_substructure = len(bitinfo[_bit])
        ratio_contribution = ratio_contributions[_bit]
        for i_atom, radius in bitinfo[_bit]:
            if radius == 0:
                n_atom_in_substructure = 1
                importance_atoms[i_atom] += (
                    ratio_contribution
                    / n_atom_in_substructure
                    / n_substructure
                )
            else:
                atom_map = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol, radius=radius, rootedAtAtom=i_atom
                )
                submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)

                n_atom_in_substructure = len(submol.GetAtoms())
                for j_atom in atom_map:
                    importance_atoms[j_atom] += (
                        ratio_contribution / n_atom_in_substructure
                    )

    # scaling
    importance_atoms_scaled = (
        importance_atoms / abs(importance_atoms).max() * 0.5
    )
    importance_atoms_scaled

    atom_colors = {
        i: (1, 1 - importance_atoms_scaled[i], 1 - importance_atoms_scaled[i])
        if importance_atoms_scaled[i] > 0
        else (
            1 + importance_atoms_scaled[i],
            1 + importance_atoms_scaled[i],
            1,
        )
        for i in range(len(importance_atoms_scaled))
    }

    view = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    tm = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    view.DrawMolecule(
        tm,
        highlightAtoms=atom_colors.keys(),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
        legend=f"y_true: {y[i]:.2f}  y_pred: {y_pred[i]:.2f}",
    )
    view.FinishDrawing()
    svg = view.GetDrawingText()
    with open("highlighted_sample.svg", "w") as f:
        f.write(svg)
    display(SVG(svg))

