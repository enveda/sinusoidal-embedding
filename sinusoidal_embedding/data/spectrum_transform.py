from matchms import Spectrum
from matchms import Spikes
from matchms.filtering import normalize_intensities

class SpectrumNumericalMZTransform(object):
    def __init__(
            self, n_max=512, peak_limits=(0, 1000), include_fractional_mz=False,
            normalize_peak_intensities=False):
        self._n_max = n_max
        self._peak_limits = peak_limits
        self._normalize_peak_intensities = normalize_peak_intensities
        self._include_fractional_mz = include_fractional_mz

    def __call__(self, spectrum):
        spectrum = spectrum.clone()

        if self._normalize_peak_intensities:
            spectrum = normalize_intensities(spectrum)

        mz_arr = np.insert(spectrum.peaks.mz, 0, spectrum.metadata['precursor_mz'])
        intensity_arr = np.insert(spectrum.peaks.intensities, 0, 2.0)

        try:
            mz_arr, intensity_arr = zip(*sorted([(mz, intensity)
                    for mz, intensity in zip(mz_arr, intensity_arr)
                    if self._peak_limits[0] <= mz <= self._peak_limits[1]],
                key=itemgetter(1), reverse=True)[:self._n_max])
        except ValueError:
            mz_arr, intensity_arr = np.array([]), np.array([])

        mz_arr = np.pad(mz_arr, (0, self._n_max - len(mz_arr)), constant_values=0.0)
        mz_arr = mz_arr.reshape((-1, 1))
        if self._include_fractional_mz:
            mz_arr = np.hstack((mz_arr, np.floor(mz_arr), mz_arr - np.floor(mz_arr)))

        intensity_arr = np.array(intensity_arr)
        intensity_arr = np.pad(
            intensity_arr, (0, self._n_max - len(intensity_arr)), constant_values=0.0)

        return mz_arr, intensity_arr