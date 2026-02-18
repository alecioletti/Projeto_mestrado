"""

@author: alessandra cioletti
"""


import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors



# This file contains functions to preprocessing the sdf and create the datasert for descriptors

## Auxiliary Functions ##


def countTerminalAtoms(mol):
    """
    Description:
        Counts terminal N, O, and S atoms in the molecule.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        tuple: Count of terminal atoms (N, O, S).
    """

    n_terminal = 0
    o_terminal = 0
    s_terminal = 0

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()

        if atomic_num == 7 and degree == 1:  # Nitrogênio terminal
            n_terminal += 1
        elif atomic_num == 8 and degree == 1:  # Oxigênio terminal
            o_terminal += 1
        elif atomic_num == 16 and degree == 1:  # Enxofre terminal
            s_terminal += 1

    return n_terminal, o_terminal, s_terminal


def calculateBenzeneRingRatio(mol):
    """
    Description:
        Calculates the ratio of aromatic rings to total rings.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        float: Ratio of aromatic rings.
    """
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
    total_rings = num_aromatic_rings + num_aliphatic_rings
    ring_ratio = num_aromatic_rings / total_rings if total_rings > 0 else 0.0

    return ring_ratio

def calculateSPHybridization(mol):
    """
    Description:
        Calculates the number of carbons with SP2 and SP3 hybridization.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        tuple: (num_sp2, num_sp3).
    """
    num_sp2 = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C' and str(atom.GetHybridization()) == 'SP2')
    num_sp3 = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C' and str(atom.GetHybridization()) == 'SP3')
    
    return num_sp2, num_sp3

def calculateMolecularRigidity(mol):
    """
    Description:
        Calculates molecular rigidity.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        float: Molecular rigidity value.
    """
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
    mol_rigidity = num_rotatable_bonds / num_heavy_atoms if num_heavy_atoms > 0 else 0.0
    
    return mol_rigidity

def calculateChemicalFlexibilityIndex(mol):
    """
    Description:
        Calculates the chemical flexibility index.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        int: Flexibility index.
    """
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    num_rings = Descriptors.RingCount(mol)
    flexibility_index = num_rotatable_bonds - num_rings
    
    return flexibility_index

def calculateNumHalogenBonds(mol):

    """
    Description:
        Calculates the number of bonds involving halogens.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        int: Number of halogen bonds.
    """
    halogen_symbols = ['F', 'Cl', 'Br', 'I']
    halogen_bonds = sum(1 for bond in mol.GetBonds() if (bond.GetBeginAtom().GetSymbol() in halogen_symbols) or (bond.GetEndAtom().GetSymbol() in halogen_symbols))

    return halogen_bonds

def calculateBranchingIndex(mol):
    """
    Description:
        Calculates the branching index (maximum degree).

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        int: Branching index.
    """

    branching_index = max(atom.GetDegree() for atom in mol.GetAtoms()) if mol.GetNumAtoms() > 0 else 0

    return branching_index

def calculateNumConjugatedBonds(mol):
    """
    Description:
        Calculates the number of conjugated bonds.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        int: Number of conjugated bonds.
    """
    num_conjugated_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsConjugated())
    
    return num_conjugated_bonds
    

def calculateSaturationRatio(mol):
    """
    Description:
        Calculates the saturation ratio of the molecule.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        float: Saturation ratio.
    """
    num_single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
    total_bonds = mol.GetNumBonds()
    saturation_ratio = num_single_bonds / total_bonds if total_bonds > 0 else 0.0 

    return saturation_ratio


def calculateMolComplexity(mol):
    """
    Description:
        Calculates molecular complexity.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        int: Molecular complexity.
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_rings = Descriptors.RingCount(mol)
    mol_complexity = num_atoms + num_bonds - num_rings
    
    return mol_complexity




def calculateShannonEntropy(mol):
    """
    Description:
        Calculates the Shannon entropy of the molecule.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        float: Shannon entropy.
    """

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    counts = Counter(symbols)
    total_atoms = len(symbols)
    entropy = 0.0
    for count in counts.values():
        probability = count / total_atoms
        entropy -= probability * np.log2(probability)
    
    return entropy


## Main Function ##

def addNewDescriptorsToDF(df, smiles_column='smiles'):
    """
    Description:
        Adds new descriptor columns to the existing DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the SMILES column.
        smiles_column (str): The name of the column in the DataFrame containing SMILES (default is 'SMILES').

    Returns:
        pd.DataFrame: The original DataFrame with the new descriptor columns added.
    """
    if smiles_column not in df.columns:
        print(f"Erro: A coluna '{smiles_column}' não foi encontrada no DataFrame.")
        return df

    benzene_ratios = []
    num_sp2_atoms = []
    num_sp3_atoms = []
    N_terminal = []
    O_terminal= []
    S_terminal = []
    molecular_rigidities = []
    chemical_flexibility_indices = []
    num_halogen_bonds_list = []
    branching_indices = []
    num_conjugated_bonds_list = []
    saturation_ratios = []
    mol_complexities = []
    shannon_entropies = []

    for index, row in df.iterrows():
        smiles = row[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            benzene_ratios.append(calculateBenzeneRingRatio(mol))
            sp2, sp3 = calculateSPHybridization(mol)
            num_sp2_atoms.append(sp2)
            num_sp3_atoms.append(sp3)
            n_terminal, o_terminal, s_terminal = countTerminalAtoms(mol)
            N_terminal.append(n_terminal)
            O_terminal.append(o_terminal)
            S_terminal.append(s_terminal)
            molecular_rigidities.append(calculateMolecularRigidity(mol))
            chemical_flexibility_indices.append(calculateChemicalFlexibilityIndex(mol))
            num_halogen_bonds_list.append(calculateNumHalogenBonds(mol))
            branching_indices.append(calculateBranchingIndex(mol))
            num_conjugated_bonds_list.append(calculateNumConjugatedBonds(mol))
            saturation_ratios.append(calculateSaturationRatio(mol))
            mol_complexities.append(calculateMolComplexity(mol))
            shannon_entropies.append(calculateShannonEntropy(mol))
        else:
            benzene_ratios.append(None)
            num_sp2_atoms.append(None)
            num_sp3_atoms.append(None)
            N_terminal.append(None)
            O_terminal.append(None)
            S_terminal.append(None)
            molecular_rigidities.append(None)
            chemical_flexibility_indices.append(None)
            num_halogen_bonds_list.append(None)
            branching_indices.append(None)
            num_conjugated_bonds_list.append(None)
            saturation_ratios.append(None)
            mol_complexities.append(None)
            shannon_entropies.append(None)

    df['BenzeneRingRatio'] = benzene_ratios
    df['NumSP2Atoms'] = num_sp2_atoms
    df['NumSP3Atoms'] = num_sp3_atoms
    df['NumTerminalN'] = N_terminal
    df['NumTerminalO'] = O_terminal
    df['NumTerminalS'] = S_terminal
    df['MolecularRigidity'] = molecular_rigidities
    df['ChemicalFlexibilityIndex'] = chemical_flexibility_indices
    df['NumHalogenBonds'] = num_halogen_bonds_list
    df['BranchingIndex'] = branching_indices
    df['NumConjugatedBonds'] = num_conjugated_bonds_list
    df['SaturationRatio'] = saturation_ratios
    df['MolComplexity'] = mol_complexities
    df['ShannonEntropy'] = shannon_entropies

    return df
