from keras.models import Sequential, Model, model_from_json
import keras
import pickle
import pandas as pd
import numpy as np
import sys
import warnings
import IPython.display as ipd
import pyaudio
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
if not sys.warnoptions:
    warnings.simplefilter("ignore")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECOND = 5
WAVE_OUTPUT_FILENAME = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\Test_Audio\\Happy_Audio\\testing.wav"
p = pyaudio.PyAudio()
stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = CHUNK)

print("* recording")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECOND)):
               data = stream.read(CHUNK)
               frames.append(data)
print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))

data, sampling_rate = librosa.load("C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\Test_Audio\\Happy_Audio\\Tannmay_happy.wav")
ipd.Audio("C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\Test_Audio\\Happy_Audio\\Tannmay_happy.wav")
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
json_file = open("C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\code\\final_audio_analysis\\saved_models\\model_json_aug.json",'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weight into new model
loaded_model.load_weights("C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\code\\final_audio_analysis\\saved_models\\Emotion_Model_aug.h5")
print("loaded model from disk")
# the optimiser
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Lets transform the dataset so we can apply the predictions
X, sample_rate = librosa.load('C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\Test_Audio\\Happy_Audio\\Tannmay_happy.wav'
                              ,res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )

sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
newdf = pd.DataFrame(data=mfccs).T
print(newdf)
# Apply predictions
newdf= np.expand_dims(newdf, axis=2)
newpred = loaded_model.predict(newdf, 
                         batch_size=16, 
                         verbose=1)

print(newpred)
filename = 'C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\code\\final_audio_analysis\\labels'

infile = open(filename,'rb')
lb = pickle.load(infile)
infile.close()

# Get the final predicted label
final = newpred.argmax(axis=1)
final = final.astype(int).flatten()
final = (lb.inverse_transform((final)))
print(final) #emo(final) #gender(final) 

