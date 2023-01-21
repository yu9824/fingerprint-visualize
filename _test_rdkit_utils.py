from rdkit import Chem

from rdkit_utils import Chemical


def test_chemical():
    chemical = Chemical("CCO")
    assert chemical.smiles == "CCO"
    assert chemical.mol is not None
    assert chemical["smiles"] == "CCO"
    assert chemical["mol"] is not None

    chemical = Chemical(chemical=Chem.MolFromSmiles("CCO"))
    assert chemical.smiles == "CCO"
    assert chemical.mol is not None
    assert chemical["smiles"] == "CCO"
    assert chemical["mol"] is not None

    chemical = Chemical(chemical)
    assert chemical.smiles == "CCO"
    assert chemical.mol is not None
    assert chemical["smiles"] == "CCO"
    assert chemical["mol"] is not None

    print(chemical)


if __name__ == "__main__":
    test_chemical()
