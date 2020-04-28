import mne
from mne.datasets.sleep_physionet.age import fetch_data
import numpy as np
import pandas as pd
from KC_algorithm.model import score_KCs
from KC_algorithm.plotting import KC_from_probas


def main():
    [edf_file] = fetch_data(subjects=[0], recording=[1])

    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}


    #### Load edf file##
    raw = mne.io.read_raw_edf(edf_file[0], preload=True)
    raw, _ = mne.set_eeg_reference(raw, [], verbose='warning')
    raw.resample(128)
    raw = raw.filter(0.3, None)
    Fs = raw.info['sfreq']

    ### Load and transform hypnogram ####
    annot = mne.read_annotations(edf_file[1])

    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}

    events_train, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    hypno = pd.DataFrame([])
    hypno['onset'] = events_train[:,0]/Fs
    hypno['dur'] = np.ones_like(events_train[:,0])*30
    hypno['label'] = events_train[:,2]


    ## Parameters for K-complex scoring##

    # The algorithm was trained on C3, as you will see the algorithm
    # does not perform well on Fpz-Cz
    wanted_channel = 'EEG Fpz-Cz' #
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
