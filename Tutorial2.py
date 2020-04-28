import mne
import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as ET


from KC_algorithm.model import score_KCs
from KC_algorithm.plotting import KC_from_probas
################################################################
# Healper function to load the hypnogram from the NSRR .xml file


def import_event_and_stages_CFS(xml_file):
    """
    Helper function to import stages and events of the Cleveland family study
    :param xml_file: xml file from NSRR CFS files
    :param reject_epochs_with_overlapping_events: bool, if True, will reject epochs with any overlapping events
    :return:
    - events: a dataframe with key 'onset', 'dur', 'label', covering each events, their onset, duration and its labels
    - events: same as events but with stages
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    events = pd.DataFrame()
    stages = pd.DataFrame()

    events['label'] = [child.text for child in root.findall("./ScoredEvents/ScoredEvent/Name")]
    events['onset'] = np.asarray([child.text for child in root.findall("./ScoredEvents/ScoredEvent/Start")],
                                 dtype='float')
    events['dur'] = np.asarray([child.text for child in root.findall("./ScoredEvents/ScoredEvent/Duration")],
                               dtype='float')


    stages['label'] = np.asarray([child.text for child in root.findall("./SleepStages/SleepStage")], dtype='int')
    stages['dur'] = np.ones_like(stages['label']) * np.asarray([child.text for child in root.findall("./EpochLength")],
                                                               dtype='int')
    stages['onset'] = np.cumsum(stages['dur']) - np.asarray([child.text for child in root.findall("./EpochLength")],
                                                               dtype='int')

    return events, stages

def main():
    #################################################################
    dirname = os.path.dirname(__file__)

    edf_filename = os.path.join(dirname, 'data/learn-nsrr01.edf')
    annot_filename = os.path.join(dirname, 'data/learn-nsrr01-profusion.xml')

    ######### Pre-processing ###########
    raw = mne.io.read_raw_edf(edf_filename, preload=True)
    raw, _ = mne.set_eeg_reference(raw, [], verbose='warning') # data is already referenced in this NSRR file
    raw.resample(128)
    raw = raw.filter(0.3, None)


    ## Import hypnogram ##
    events, hypno = import_event_and_stages_CFS(xml_file=annot_filename)


    ## Parameters for K-complex scoring##
    wanted_channel = 'EEG' # EEG = C3 in NSRR files
    wanted_sleep_stages = [2,3]
    # we multiply the channel by -1 since the algorithm was trained on negative EEG polarity
    C3 = np.asarray(
        [raw[count, :][0] for count, k in enumerate(raw.info['ch_names']) if
         k == wanted_channel]).ravel()*-1
    Fs = raw.info['sfreq']



    peaks, stage_peaks, d, probas = score_KCs(C3, Fs, hypno,sleep_stages=[2,3])


    #######################################################################
    probability_threshold = 0.5 # include only waveform scored with a probability of at least 50%

    labels = np.where(probas>0.5,1,0)
    onsets = peaks[probas>0.5]
    probas = probas[probas>0.5]

    print('{} K-complexes were detected'.format(np.sum(labels)))
    #

    ########################################################################
    ####                        VIZUALISATION                           ####

    ##----- Average K-complexes for different probability threshold ------##
    KC_from_probas(C3*-1,onsets,probas,Fs)

    ##------------------------Plotting with mne --------------------------##

    tmin = -2
    tmax= 2

    events_mne = np.vstack([onsets, np.zeros_like(onsets), np.ones_like(onsets)]).T


    ep = mne.Epochs(raw, events_mne, picks=['EEG'],baseline=None,tmin=tmin,tmax=tmax)

    # Plot each individuals K-complexes
    ep.plot(block=True)

    # Plot a similar Figure 4 in our manuscript
    ep.plot_image(group_by=None, picks=['EEG'], vmin=-50, vmax=50)

if __name__ == "__main__":
    main()
