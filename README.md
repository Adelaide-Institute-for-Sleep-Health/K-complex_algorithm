# Probabilistic classification of K-complexes

K-complexes are prominent features of sleep electroencephalography and could be a useful bio-marker of a wide range of clinical conditions. However, manual scoring of K-complexes is impractical, time-consuming and thus costly and currently not well-standardized. This repositery contains our K-complex detection algorithm - with its weights - published in SLEEP. The algorithm attributes a given waveform a probability (from 0 to 100%) of being a K-complex, thus making the scoring intuitive.

## Getting Started

### Prerequisites
The algorithm was developed in Python 3.7 and requires the following dependencies (Version):

- MNE (0.20)
- Pywavelets (1.0.3)
- Pytorch (torch = 1.1.0, torchvision = 0.2.2)
- Gpytorch (0.3.4)
- scikit-learn (0.22.2)
- Matplotlib (3.2.1) only for plotting functions

Pytorch and Gpytorch needs to be installed with the mentionned version (newer version will not work).

### First steps
After installing the dependencies, download the code and unzip it. 

Tutorial1.py: run K-complex detection on an EDF file from PhysioNet
Data is contained within the [MNE-python](https://mne.tools/stable/index.html) package.

Tutorial2.py: run K-complex detection on learn_nsrr01.edf from the [National Sleep Researsh Ressources](https://sleepdata.org/datasets/learn/files/polysomnography)
Download the .edf and .xml file and placed them in a folder named "data" before executing Tutorial2.py

### Citation

If you use this code, please consider citing:

Bastien Lechat, Kristy Hansen, Peter Catcheside, Branko Zajamsek, Beyond K-complex binary scoring during sleep: Probabilistic classification using deep learning, Sleep, https://doi.org/10.1093/sleep/zsaa077

### License
  Copyright (C) 2020  Bastien Lechat

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
