#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.



import numpy as np

import joblib
import os
from sleepAnalysis.utils import EpochData
from sleepAnalysis.utils import findN2peaks
import warnings
warnings.filterwarnings('ignore')

import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy
import sklearn
import torch


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim,drop_out =0.5):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000, bias=False))
        self.add_module('bn1', torch.nn.BatchNorm1d(1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('dropout1', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear2', torch.nn.Linear(1000, 1000,bias=False))
        self.add_module('bn2', torch.nn.BatchNorm1d(1000))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear3', torch.nn.Linear(1000, 500,bias=False))
        self.add_module('bn3', torch.nn.BatchNorm1d(500))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('dropout3', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear4', torch.nn.Linear(500, 256,bias=False))
        self.add_module('bn4', torch.nn.BatchNorm1d(256))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('dropout4', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear6', torch.nn.Linear(256, output_dim,bias=False))

class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution,
                                                           learn_inducing_locations=True)
        super(GaussianProcessLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKLModel(gpytorch.Module):
    def __init__(self, inducing_points, feature_extractor, num_features):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(inducing_points)
        self.num_features = num_features

    def forward(self, x):
        #print(x.type())
        projected_x = self.feature_extractor(x.float())

        res = self.gp_layer(projected_x)
        return res



def predict(model, likelihood, X):
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad():
        output = likelihood(model(X))  #

        pred_labels = output.mean.ge(0.5).float().cpu().numpy()

        probas = output.mean.cpu().numpy()
    return probas, pred_labels


def get_model():
    dirname = os.path.dirname(__file__)

    inducing_filename = os.path.join(dirname, 'WeightsKCalgo/inducing_points_A2.npy')
    model_file = os.path.join(dirname, 'WeightsKCalgo/finaldkl_final_model_epoch50.dat')
    data_dim = 128
    num_features = 16
    drop_out_rate = 0.8

    feature_extractor = LargeFeatureExtractor(input_dim=data_dim,
                                              output_dim=num_features,
                                              drop_out=drop_out_rate)

    X_induced = torch.from_numpy(np.load(inducing_filename))


    model = DKLModel(inducing_points=X_induced, feature_extractor=feature_extractor,
                     num_features=num_features)

    # Bernouilli likelihood because only 2 classes
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu'))['model'])
    likelihood.load_state_dict(torch.load(model_file,map_location=torch.device('cpu'))['likelihood'])

    return model, likelihood


def score_KCs(C3, Fs, Stages,sleep_stages = [2,3]):
    dirname = os.path.dirname(__file__)
    scaler_filename = os.path.join(dirname,'WeightsKCalgo/scaler_final_A2.save')
    scaler = joblib.load(scaler_filename)


    model, likelihood = get_model()


    amp_thres = 20 * 10 ** -6  # 20 micro volt
    dist = 2  # [s]
    post_peak = 3  # [s]
    pre_peak = 3  # [s]
    length_of_stages = Stages['dur'].values[0]
    peaks,stage_peaks = findN2peaks(Stages=Stages.loc[Stages['label'].isin(sleep_stages),:],
                        data=C3, Fs=Fs, min=amp_thres, distance=dist,
                        criteria=length_of_stages)

    assert len(peaks)==len(stage_peaks)
    d = EpochData(peaks, C3, post_peak, pre_peak, Fs)

    from sleepAnalysis.Wavelet import wave

    s = [0, 1, 2, 3, 4]
    wavelet = 'sym3'
    wa = wave(Fs, wavelet, d)

   # wa.showdwtfreqs()

    coefsnoden = wa.dwtdec().get_coef(wanted_scale=s)

    X = coefsnoden

    data_scaled = scaler.transform(X)

    probas, _ = predict(model, likelihood, torch.from_numpy(data_scaled))

    return peaks,stage_peaks,d,probas
