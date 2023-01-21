from typing import Union, Dict, Optional, Tuple
from typing_extensions import Self

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


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
