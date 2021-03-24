import socket

host = socket.gethostname()
CLASSES = []
MAX_CLASSES = 10
NUM_MFCC = 16
NUM_FRAMES = 40
DURATION = 0.91  # in seconds
GENDER_CLASSES = ['M', 'F']
PICKLE_FILE_PREFIX = 'LibriSpeech-mfcc-'



PROJECT_ROOT = '/afs/umich.edu/class/eecs627/w20/group5/GoldenModel/lstm_gender_classifier/'
DATASET_STR = 'dev-clean'
DATA_ROOT = '/afs/umich.edu/class/eecs627/w20/group5/GoldenModel/lstm_gender_classifier/LibriSpeech/'
DATA_DIR = DATA_ROOT + DATASET_STR + '/'
SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
SPEAKER_IDX = 11
CHAPTER_IDX = 12
FILENAME_IDX = 13
NUM_CLASSES = 40
