#this script should be executed in pytorch1.11 env
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
 
data_dir = "/home/zacharyyeh/Datasets/VOX"
 
class VOXDataset(Dataset):
  def __init__(self, data_dir, segment_len=128):
    self.data_dir = data_dir
    #segment_len is the lengh of vectors accepted by transformer layer
    #the input data has to be reshape to fit segment_len
    self.segment_len = segment_len
 
    #Load the mapping from speaker neme to their corresponding id. 
    #this mapping contain speaker id  correspond to speaker number
    #Is important to decode the speaker id, so that it become an interger that can be proccessed
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    self.speaker2id = mapping["speaker2id"]
 
    #Load metadata of training data.
    #Metadata contains the speaker id correspond to its feature path
    #Normally a speaker is related to several features
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.load(open(metadata_path))["speakers"]
 
    #Get the total number of speaker.
    self.speaker_num = len(metadata.keys())
    self.data = []
    #iterate through metadata
    for speaker in metadata.keys():
      #for each speaker, find all data's path that is spoken by the speaker, append to data
      for utterances in metadata[speaker]:
        #append(feature_paths, speaker num(interger))
        self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
  def __len__(self):
    return len(self.data)
 
  def __getitem__(self, index):
    feat_path, speaker = self.data[index]
    # Load preprocessed mel-spectrogram.
    mel = torch.load(os.path.join(self.data_dir, feat_path))
 
    # Segmemt mel-spectrogram into "segment_len" frames.
    if len(mel) > self.segment_len:
      # If the lengh is larger than segment_len
      # Randomly get the starting point of the segment.
      start = random.randint(0, len(mel) - self.segment_len)
      # Get a segment with "segment_len" frames.
      mel = torch.FloatTensor(mel[start:start+self.segment_len])
    else:
      mel = torch.FloatTensor(mel)
    # Turn the speaker id into long for computing loss later.
    speaker = torch.FloatTensor([speaker]).long()

    #mel shape(128, 40)
    #40 is the size of a single frame features
    #there all total 128 frame per mel spectrum
    return mel, speaker
 
  def get_speaker_number(self):
    return self.speaker_num
