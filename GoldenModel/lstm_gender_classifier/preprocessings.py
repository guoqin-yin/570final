import glob
import pickle

import librosa
import numpy as np
import os
import random

from constants import DATA_DIR, DATASET_STR, SPEAKER_FILE, \
    CHAPTER_IDX, SPEAKER_IDX, FILENAME_IDX, GENDER_CLASSES, DURATION, \
    NUM_MFCC, NUM_FRAMES, PICKLE_FILE_PREFIX, PROJECT_ROOT, CLASSES, MAX_CLASSES


def init_reader_gender_map():
    reader_gender_map = {}
    with open(SPEAKER_FILE) as f:
        content = f.readlines()

        for line in content:
            if DATASET_STR in line:
                temp = line.split('|')
                speaker_id = temp[0].strip()
                reader_gender = temp[1].strip()
                reader_gender_map[speaker_id] = reader_gender
    return reader_gender_map


# rg_map = init_reader_gender_map()
# print(len(reader_gender_map.keys()))


def get_data(class_type):
    rg_map = {}
    if class_type == 'gender':
        rg_map = init_reader_gender_map()
    file_list = glob.glob(DATA_DIR+'*/*/*.flac')
    print("Loading {:d} files from: {:s}".format(len(file_list), DATA_DIR))
    all_data = []
    for f in file_list:
        fsplit = f.split('/')
        speaker_id = fsplit[SPEAKER_IDX]
        # chapter_id = fsplit[CHAPTER_IDX]
        # filename = fsplit[FILENAME_IDX]

        all_data.append({
            'speaker_id': speaker_id,
            'filename': f
        })

    random.shuffle(all_data)
    print("dataset size: {}".format(len(all_data)))
    
    X = []
    y = []
    TEMP_CLASS_IDNEX = []
    for pair in all_data:
        if class_type == 'speaker':
            if len(CLASSES) > 0:
                if pair['speaker_id'] in CLASSES:
                    X.append(pair['filename'])
                    y.append(CLASSES.index(pair['speaker_id']))
            else:
                if len(TEMP_CLASS_IDNEX) >= MAX_CLASSES and pair['speaker_id'] not in TEMP_CLASS_IDNEX:
                    continue

                X.append(pair['filename'])
                if pair['speaker_id'] in TEMP_CLASS_IDNEX:
                    y.append(TEMP_CLASS_IDNEX.index(pair['speaker_id']))
                else:
                    TEMP_CLASS_IDNEX.append(pair['speaker_id'])
                    y.append(TEMP_CLASS_IDNEX.index(pair['speaker_id']))
        else:
            X.append(pair['filename'])
            y.append(GENDER_CLASSES.index(rg_map[pair['speaker_id']]))

    return X, y


# data_X, data_y = get_data("speaker")
# print(len(data_X))
# print(len(data_y))


def load_flac(filename):
    audio, sr = librosa.load(filename, duration=DURATION)
    return audio, sr


# print(len(load_flac(data_X[0])[0]))
# print(load_flac(data_X[0])[1])


def add_zero_paddings(audio, sr):
    signal_length = DURATION * sr
    audio_length = len(audio)
    padding_length = signal_length - audio_length
    if padding_length > 0:
        paddings = np.zeros(padding_length)
        padded_signal = np.hstack((audio, paddings))
        return padded_signal
    return audio


def get_mfcc(filename):
    audio, sr = load_flac(filename)
    signal = add_zero_paddings(audio, sr)
    return librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)


# mfcc_data = get_mfcc(data_X[0])
# print(len(mfcc_data[0]))

def get_mfccs(file_list=False, pickle_file=False):
    if pickle_file:
        x_audio = load_from_pickle(pickle_file)
        return x_audio
    else:
        x_audio = []
        for i in range(len(file_list)):
            if i%100 == 0:
                print("{:.2f} loaded".format(i/len(file_list)))
            x_audio.append(np.reshape(get_mfcc(file_list[i]), [NUM_MFCC, NUM_FRAMES]))
        return x_audio


def get_datases(class_type):
    x, y = get_data(class_type)
    split_tupe_x = np.split(np.array(x), [int(0.9 * len(x))])
    x_train = split_tupe_x[0].tolist()
    x_test = split_tupe_x[1].tolist()

    split_tupe_y = np.split(np.array(y), [int(0.9 * len(y))])
    y_train = split_tupe_y[0].tolist()
    y_test = split_tupe_y[1].tolist()

    return (x_train, y_train), (x_test, y_test)


def save_to_pickle(data, filename):
    filename = PROJECT_ROOT + 'pkl_data/' + PICKLE_FILE_PREFIX + filename
    print("storing {} data into file {}".format(len(data), filename))
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()


def load_from_pickle(filename):
    filename = PROJECT_ROOT + 'pkl_data/' + PICKLE_FILE_PREFIX + filename
    print("loading from file: {}".format(filename))
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

# (x_train, y_train), (x_test, y_test) = get_datases()
# print(len(x_train))
# print(len(y_train))

def regularize_data(x_data_mfccs):
    x_data_mfccs = x_data_mfccs.astype(int)

    Nx_train = x_data_mfccs.shape[0]
    Ny_train = x_data_mfccs.shape[1]
    Nz_train = x_data_mfccs.shape[2]
    for i in range (0,Nx_train):
        for j in range (0,Ny_train):
            for k in range (0,Nz_train):
                if (x_data_mfccs[i,j,k] > 127):
                    x_data_mfccs[i,j,k] = 127
                elif (x_data_mfccs[i,j,k] < -128):
                    x_data_mfccs[i,j,k] = -128


    return x_data_mfccs

