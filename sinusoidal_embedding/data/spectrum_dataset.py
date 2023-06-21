import os
import random
import pickle

import numpy as np

from torch.utils.data import Dataset

class SpectrumDataset(Dataset):
    def __init__(self, spectra, transform=None, keep_idx=None, filter=None):
        if (keep_idx is not None) and (filter is not None):
            raise ValueError('can only specify either keep_idx or filter but not both')

        self._spectra = spectra

        if filter is not None:
            # only filter specified
            self._keep_idx = [
                idx for idx, spectrum in enumerate(self._spectra) if filter(spectrum)]
        elif keep_idx is not None:
            # only keep_idx specified
            self._keep_idx = sorted(keep_idx)
            if self._keep_idx[0] < 0 or self._keep_idx[-1] >= len(self._spectra):
                raise ValueError(
                    'keep_idx includes index values exceeding specified number of provided spectra')
        else:
            # neither filter nor keep_idx specified
            self._keep_idx = list(range(len(self._spectra)))

        self._num_spectra = len(self._keep_idx)
        self._exclude_idx = set(range(len(self._spectra))) - set(self._keep_idx)

        self._transform = transform

    def __len__(self):
        return self._num_spectra

    def __getitem__(self, idx):
        idx = self._keep_idx[idx]
        spectrum = self._spectra[idx].clone()

        if self._transform is not None:
            spectrum = self._transform(spectrum)

        return spectrum

    @property
    def spectra(self):
        return [self._spectra[idx] for idx in self._keep_idx]

    @classmethod
    def load(cls, data_path, transform=None, keep_idx=None, filter=None):
        with open(os.path.join(data_path, 'spectra.pkl'), 'rb') as fid:
            spectra = pickle.load(fid)

        return cls(spectra, transform, keep_idx, filter)

class SpectrumPairDataset(Dataset):
    def __init__(
            self, spectra, smiles2spectrum, spectrum2smiles, similarity_matrix, sim_bins=None,
            transform=None, keep_idx=None, filter=None):
        if (keep_idx is not None) and (filter is not None):
            raise ValueError('can only specify either keep_idx or filter but not both')

        self._spectra = spectra
        self._smiles2spectrum = smiles2spectrum
        self._spectrum2smiles = spectrum2smiles
        self._similarity_matrix = similarity_matrix

        if filter is not None:
            # only filter specified
            self._keep_idx = [
                idx for idx, spectrum in enumerate(self._spectra) if filter(spectrum)]
        elif keep_idx is not None:
            # only keep_idx specified
            self._keep_idx = sorted(keep_idx)
            if self._keep_idx[0] < 0 or self._keep_idx[-1] >= len(self._spectra):
                raise ValueError(
                    'keep_idx includes index values exceeding specified number of provided spectra')
        else:
            # neither filter nor keep_idx specified
            self._keep_idx = list(range(len(self._spectra)))

        self._num_spectra = len(self._keep_idx)
        self._exclude_idx = set(range(len(self._spectra))) - set(self._keep_idx)

        self._smiles2spectrum = [
            [spectrum for spectrum in spectra_list if spectrum not in self._exclude_idx]
            for spectra_list in self._smiles2spectrum]

        self._sim_bins = (list(zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]))
            if sim_bins is None else sim_bins)
        self._transform = transform

    def __len__(self):
        return self._num_spectra

    def __getitem__(self, idx):
        idx = self._keep_idx[idx]
        spectrum_0 = self._spectra[idx].clone()

        sim_window = random.choice(self._sim_bins)
        sim_window_diff = sim_window[1] - sim_window[0]
        found_spectra = list()
        while len(found_spectra) == 0:
            found_smiles = np.where(
                (self._similarity_matrix[self._spectrum2smiles[idx], :] >= sim_window[0]) &
                (self._similarity_matrix[self._spectrum2smiles[idx], :] <= sim_window[1]))[0]
            found_spectra = [
                spectrum for smiles in found_smiles
                    for spectrum in self._smiles2spectrum[smiles]
                    if spectrum != idx]
            sim_window = (sim_window[0] - sim_window_diff, sim_window[1] + sim_window_diff)

        pair_idx = random.choice(found_spectra)

        spectrum_1 = self._spectra[pair_idx].clone()
        similarity = self._similarity_matrix[
            self._spectrum2smiles[idx], self._spectrum2smiles[pair_idx]]

        if self._transform is not None:
            spectrum_0 = self._transform(spectrum_0)
            spectrum_1 = self._transform(spectrum_1)

        return spectrum_0, spectrum_1, similarity

    @classmethod
    def load(cls, data_path, sim_bins=None, transform=None, keep_idx=None, filter=None):
        with open(os.path.join(data_path, 'spectra.pkl'), 'rb') as fid:
            spectra = pickle.load(fid)
        with open(os.path.join(data_path, 'smiles2spectrum.pkl'), 'rb') as fid:
            smiles2spectrum = pickle.load(fid)
        with open(os.path.join(data_path, 'spectrum2smiles.pkl'), 'rb') as fid:
            spectrum2smiles = pickle.load(fid)

        sim_matrix_dir = os.path.join(data_path, 'tanimoto_similarity_matrix_f16.npy')
        part_file_names = sorted([
            file_name for file_name in os.listdir(sim_matrix_dir) if '.npy' in file_name])
        similarity_matrix = np.concatenate([
            np.load(os.path.join(sim_matrix_dir, part_file_name))
            for part_file_name in part_file_names])

        return cls(
            spectra, smiles2spectrum, spectrum2smiles, similarity_matrix, sim_bins, transform,
            keep_idx, filter)
