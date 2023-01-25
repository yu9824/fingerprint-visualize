from typing import Union, Dict, Optional, Tuple, List
from typing_extensions import Self
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from joblib import Parallel, delayed


__all__ = [
    "Chemical",
]

SMILES = str


class Chemical(dict):
    def __init__(
        self,
        chemical: Union[SMILES, Chem.rdchem.Mol, Self],
        canonicalize: bool = True,
    ):
        """chemical is a SMILES string, an RDKit Mol object, or another

        Parameters
        ----------
        chemical : Union[SMILES, Chem.rdchem.Mol, Self]
            SMILES string, RDKit Mol object, or another Chemical object
        canonicalize : bool, optional
            if SMILES is assigned, canonicalize SMILES or not, by default True

        Raises
        ------
        TypeError
            chemical must be of type SMILES, Chem.rdchem.Mol, or Chemical
        """
        self.chemical = chemical
        self.canonicalize = canonicalize
        super().__init__()

        # SMILES
        if type(chemical) == str:
            self.__mol = Chem.MolFromSmiles(chemical)
            if self.__mol is None:
                raise ValueError(f"Invalid SMILES: {chemical}")

            if self.canonicalize:
                self.__smiles = Chem.MolToSmiles(self.__mol, canonical=True)
            else:
                self.__smiles = chemical
        elif type(chemical) == Chem.rdchem.Mol:
            self.__mol = chemical
            self.__smiles = Chem.MolToSmiles(self.__mol, canonical=True)
        elif type(chemical) == type(self):
            self.__mol = chemical.mol
            self.__smiles = chemical.smiles
        else:
            raise TypeError(
                f"Chemical must be of type {SMILES}, {Chem.rdchem.Mol}, "
                + f"or {type(self)}"
            )

        # typing
        self.__mol: Chem.rdchem.Mol
        self.__smiles: SMILES

        self["smiles"] = self.__smiles
        self["mol"] = self.__mol

    @property
    def mol(self) -> Chem.rdchem.Mol:
        return self.__mol

    @mol.setter
    def mol(self, value: Chem.rdchem.Mol):
        raise AttributeError("mol is read-only")

    @property
    def smiles(self) -> SMILES:
        return self.__smiles

    @smiles.setter
    def smiles(self, value: SMILES):
        raise AttributeError("smiles is read-only")


class VisualizeImportanceAtoms:
    def __init__(
        self,
        bitinfos: List[Dict[int, Tuple[Tuple[int, int]]]],
        ratio_contributions: List[Union[Dict[int, float], pd.Series]],
        mols: List[Chem.rdchem.Mol],
        legends: Optional[List[str]] = None,
        count_fp: bool = False,
    ):
        self.bitinfos = bitinfos
        self.ratio_contributions = ratio_contributions
        self.mols = mols
        self.legends = legends
        self.count_fp = count_fp

        if legends is None:
            self.legends = [""] * len(mols)

        if (
            len(self.bitinfos)
            == len(self.ratio_contributions)
            == len(self.mols)
            == len(self.legends)
        ):
            self.n_samples = len(self.bitinfos)
        else:
            raise ValueError(
                "bitinfos, ratio_contributions, mols, legends"
                " must have same length"
            )

    def __call__(
        self,
        indices: Optional[List[int]] = None,
        n_jobs: Optional[int] = None,
        scaling: bool = True,
        show_on_ipynb: bool = False,
        save_dir: Optional[str] = None,
        filename: str = "sample{}.svg",
    ) -> None:

        if indices is None:
            indices = range(self.n_samples)

        importances_atoms: List[np.ndarray] = Parallel(n_jobs=n_jobs)(
            delayed(self._calc_importance_atoms)(
                mol=self.mols[i],
                bitinfo=self.bitinfos[i],
                ratio_contribution=self.ratio_contributions[i],
                count_fp=self.count_fp,
            )
            for i in indices
        )
        # importances_atoms: List[np.ndarray] = [
        #     self._calc_importance_atoms(
        #         mol=self.mols[i],
        #         bitinfo=self.bitinfos[i],
        #         ratio_contribution=self.ratio_contributions[i],
        #         count_fp=self.count_fp,
        #     )
        #     for i in indices
        # ]

        assert len(importances_atoms) == len(self.mols)

        self.scales_ = [
            np.max(abs(importance_atoms))
            for importance_atoms in importances_atoms
        ]
        if scaling:
            self.scales_ = [
                max(self.scales_) for _ in range(len(self.scales_))
            ]

        svgs = Parallel(n_jobs=n_jobs)(
            delayed(self._draw)(
                mol=self.mols[i],
                importance_atoms=importances_atoms[i],
                scale=self.scales_[i],
                legend=self.legends[i],
            )
            for i in indices
        )

        assert len(svgs) == len(indices)
        for i, svg in zip(indices, svgs):
            if os.path.isdir(save_dir):
                if filename.endswith(".svg") and "{}" in filename:
                    with open(
                        os.path.join(save_dir, filename.format(i)),
                        mode="w",
                        encoding="utf-8",
                    ) as f:
                        f.write(svg)
                else:
                    raise ValueError(f"{filename} is not svg file")
            else:
                raise ValueError(f"{save_dir} is not directory")

            if show_on_ipynb:
                from IPython.display import SVG, display

                display(SVG(svg))

        return svgs

    def _calc_importance_atoms(
        self,
        mol: Chem.rdchem.Mol,
        bitinfo: Dict[int, Tuple[Tuple[int, int]]],
        ratio_contribution: Union[Dict[int, float], pd.Series],
        count_fp: bool = False,
    ) -> np.ndarray:
        if type(ratio_contribution) == pd.Series:
            ratio_contribution = ratio_contribution.to_dict()

        bit_list = list(set(bitinfo.keys()) & set(ratio_contribution.keys()))

        importance_atoms = np.zeros(mol.GetNumAtoms(), dtype=float)
        for _bit in bit_list:
            n_substructure = 1 if count_fp else len(bitinfo[_bit])
            contribution: float = ratio_contribution[_bit]
            for i_atom, radius in bitinfo[_bit]:
                if radius == 0:
                    n_atom_in_substructure = 1
                    importance_atoms[i_atom] += (
                        contribution / n_atom_in_substructure / n_substructure
                    )
                else:
                    atom_map = {}
                    env = Chem.FindAtomEnvironmentOfRadiusN(
                        mol, radius=radius, rootedAtAtom=i_atom
                    )
                    submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)

                    n_atom_in_substructure = submol.GetNumAtoms()
                    for j_atom in atom_map:
                        importance_atoms[j_atom] += (
                            contribution
                            / n_atom_in_substructure
                            / n_substructure
                        )

        return importance_atoms

    def _draw(
        self,
        mol: Chem.rdchem.Mol,
        importance_atoms: np.ndarray,
        scale: float,
        legend: Optional[str] = None,
    ) -> str:

        if legend is None:
            legend = ""

        # scaling
        importance_atoms_scaling = importance_atoms / scale * 0.5

        atom_colors: Dict[int, Tuple[float, float, float]] = {
            j: (
                1.0,
                1 - importance_atoms_scaling[j],
                1 - importance_atoms_scaling[j],
            )
            if importance_atoms_scaling[j] > 0
            else (
                1 + importance_atoms_scaling[j],
                1 + importance_atoms_scaling[j],
                1.0,
            )
            for j in range(len(importance_atoms_scaling))
        }
        view = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        tm = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
        view.DrawMolecule(
            tm,
            highlightAtoms=atom_colors.keys(),
            highlightAtomColors=atom_colors,
            highlightBonds=[],
            highlightBondColors={},
            legend=legend,
        )
        view.FinishDrawing()
        svg = view.GetDrawingText()
        return svg
