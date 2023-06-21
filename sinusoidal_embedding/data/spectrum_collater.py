from abc import ABCMeta, abstractmethod

import numpy as np
import torch

class SpectrumCollater(metaclass=ABCMeta):
    def __init__(self, mz_precision = None):
        self._mz_precision = mz_precision

    def __call__(self, spectra):
        return self._process_spectra(spectra)

    @abstractmethod
    def _process_spectra(self, spectra):
        pass


class SpectrumMZIntensityCollater(SpectrumCollater):
    def _process_spectra(self, spectra):
        spectra_mz, spectra_intensity = zip(*spectra)

        mz_tensor = self._process_mz(spectra_mz)
        intensity_tensor = self._process_intensity(spectra_intensity)

        return mz_tensor, intensity_tensor

    @abstractmethod
    def _process_mz(self, spectra_mz):
        pass

    def _process_intensity(self, spectra_intensity):
        intensity_tensor = np.concatenate([
            intensity.reshape(1, -1, 1) for intensity in spectra_intensity], axis=0)
        intensity_tensor = torch.tensor(intensity_tensor, dtype=torch.float)

        return intensity_tensor

class SpectrumNumericalMZCollater(SpectrumMZIntensityCollater):
    def _process_mz(self, spectra_mz):
        mz_tensor = np.concatenate([
            mz.reshape(1, mz.shape[0], mz.shape[1]) for mz in spectra_mz], axis=0)

        if self._mz_precision is None:
            mz_tensor = torch.tensor(mz_tensor, dtype=torch.float)
        elif self._mz_precision == 16:
            mz_tensor = torch.tensor(mz_tensor, dtype=torch.float16)
        elif self._mz_precision == 32:
            mz_tensor = torch.tensor(mz_tensor, dtype=torch.float32)
        elif self._mz_precision == 64:
            mz_tensor = torch.tensor(mz_tensor, dtype=torch.float64)
        else:
            raise ValueError()

        return mz_tensor

class TanimotoPairsSpectrumCollater(SpectrumCollater):
    def __call__(self, data):
        spectra_0, spectra_1, tanimoto = zip(*data)

        spectra_0 = self._process_spectra(spectra_0)
        spectra_1 = self._process_spectra(spectra_1)

        tanimoto = torch.tensor(tanimoto, dtype=torch.float)

        return spectra_0, spectra_1, tanimoto

class TanimotoPairsSpectrumNumericalMZCollater(
        TanimotoPairsSpectrumCollater, SpectrumNumericalMZCollater):
    pass
