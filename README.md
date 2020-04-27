# Probabilistic classification of K-complexes

K-complexes are prominent features of sleep electroencephalography and could be a useful bio-marker of a wide range of clinical conditions. However, manual scoring of K-complexes is impractical, time-consuming and thus costly and currently not well-standardized. This repositery contains our K-complex detection algorithm - with its weights - published in SLEEP. The algorithm attributes a given waveform a probability (from 0 to 100%) of being a K-complex, thus making the scoring intuitive.

Code will be made available soon
## Getting Started

### Prerequisites
The algorithm was developed in python 3.7 and requires the following dependencies (Version):

- MNE (0.20)
- Pywavelets (1.0.3)
- Pytorch (torch = 1.1.0, torchvision = 0.2.2)
- Gpytorch (0.3.4)
- scikit-learn (0.22.2)
- Matplotlib (3.2.1) only for plotting functions

Pytorch and Gpytorch needs to be installed with the mentionned version (newer version will not work).

### First steps


### Citation

If you use this code, please consider citing:

Bastien Lechat, Kristy Hansen, Peter Catcheside, Branko Zajamsek, Beyond K-complex binary scoring during sleep: Probabilistic classification using deep learning, Sleep, https://doi.org/10.1093/sleep/zsaa077


