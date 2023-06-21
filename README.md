# sinusoidal-embedding
code to generate sinusoidal embeddings

## Installation instructions
This package is poetry managed can be installed via

```
poetry add git+https://github.com/enveda/sinusoidal-embedding.git
```

## Usage example

The example below is a brief example of how to instantiate and use sinusoidal peak embeddings. The output of the `peak_embedding` is a tensor with dimensions `[batch, peak sequence, embedding]` and can be passed to any torch sequence model. 

```
import numpy as np

from torch.utils.data import DataLoader

from matchms import Spectrum

from sinusoidal_embedding.peak_embedding import PeakEmbedding
from sinusoidal_embedding.peak_embedding import SinusoidalMZEmbedding
from sinusoidal_embedding.data.spectrum_dataset import SpectrumDataset
from sinusoidal_embedding.data.spectrum_transform import SpectrumNumericalMZTransform
from sinusoidal_embedding.data.spectrum_collater import SpectrumNumericalMZCollater

# Generate some random spectra for testing
spectra = [
    Spectrum(
        np.sort(np.random.rand(n)*1000), 
        np.random.rand(n), 
        metadata={'precursor_mz': np.random.rand() * 1000}
    ) 
    for n in np.random.randint(0,1000, 200)
]

# Instantiate data loader
spectrum_transform = SpectrumNumericalMZTransform()
spectrum_collater = SpectrumNumericalMZCollater()
spectrum_dataset = SpectrumDataset(spectra, transform=spectrum_transform)

data_loader = DataLoader(
    spectrum_dataset,
    batch_size=8,
    collate_fn=spectrum_collater,
)

# Instantiate Sinusoidal Peak Embedding
peak_embedding = PeakEmbedding(SinusoidalMZEmbedding(embd_dim=64))

# Generate peak embeddings
data = next(iter(data_loader))
peak_embedding(data)
```