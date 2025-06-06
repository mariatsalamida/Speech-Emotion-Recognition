
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch

# List all available devices and print them with their indices
print("Available audio devices:")
devices = sd.query_devices()
for index, device in enumerate(devices):
    print(f"Index {index}: {device['name']}")

# initialization of the model
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
emotion_label = '---'
# EXERCISE: Thread-check variable
# You will need to define a boolean variable that controls whether the
# emotion recognition thread is running. The role of this variable is to identify
# whether the emotion recognition process is already running. Later on, this variable
# will be checked any time emotion recognition needs to start in the while loop of
# the program. If the variable indicates that the emotion recogntion thread is already
# running, nothing should happen. If the thread has finished running, the thread should
# .join() and become None, so that it will start the next time in the while loop.
# ENTER VARIABLE HERE
# Thread-check variable
is_predicting = False

plt.ion()

# set blocksize for the audio callback
blocksize = 512
# set desired sample rate
sr = 16000

# initialize a buffer for processing / viewing
buffer_size_secs = 2
buffer = np.zeros(int(buffer_size_secs * sr))

# get list of all input / output devices
print(sd.query_devices())

# make sure that we get the intended device, e.g., laptop microphone
macbook_mic_device_id = -1
macbook_speakers_device_id = -1
for i in range(len(sd.query_devices())):
    tmp_dev = sd.query_devices(i)
    if 'macbook' in tmp_dev['name'].lower() and 'microphone' in tmp_dev['name'].lower():
        macbook_mic_device_id = i
    if 'macbook' in tmp_dev['name'].lower() and 'speakers' in tmp_dev['name'].lower():
        macbook_speakers_device_id = i
print('macbook_mic_device_id: ', macbook_mic_device_id)
print('macbook_speakers_device_id: ', macbook_speakers_device_id)

# get list of all input / output devices
print(sd.query_devices(macbook_mic_device_id))


def callback(indata, frames, time, status):
    amp = np.abs(indata).max()
    print(int(amp * 100) * '|' + str(frames) + str(indata.shape))
    # get global buffer
    global buffer
    # roll it and append block incoming from the mic
    buffer = np.roll(buffer, -indata.shape[0])
    buffer[-512:] = indata[:, 0]


# function to plot buffer
# EXERCISE: modify the plot_buffer function so that it takes as
# argument the predicted emotion to print as title
# BEWARE: if ploting may start before the model has taken its first decision,
# so you will need to have a "default" value printed if the decision has not
# been made yet
def plot_buffer(emotion):
    plt.clf()
    plt.plot(buffer)
    plt.axis([0, buffer.shape[0], -1, 1])
    # EXERCISE: add a plot title based on the input argument of the plot_buffer function
    plt.title(f'Emotion: {emotion}')
    plt.show()
    plt.pause(0.01)


# function to wait user termination on a separate thread
def user_input_function():
    input()
    global playing_sound
    playing_sound = False


# function to predict emotion
def predict_emotion():
    # EXERCISE: insert the thread-check variable as global
    global emotion_label, is_predicting
    is_predicting = True
    # EXERCISE: the thread-check variable should indicate that processing has started
    inputs = feature_extractor(buffer, sampling_rate=16000, padding=True, return_tensors="pt")
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    emotion_label = labels[0]
    # EXERCISE: the thread-check variable should indicate that processing has ended
    is_predicting = False


with sd.InputStream(device=macbook_mic_device_id,
                    samplerate=sr, callback=callback,
                    blocksize=blocksize):
    print('#' * 10)
    print('press Return to quit')
    print('#' * 10)
    playing_sound = True
    threaded_input = Thread(target=user_input_function)
    threaded_input.start()
    while playing_sound:
        # EXERCISE: modify the plot_buffer function so that it takes as
        # argument the predicted emotion to print as title
        plot_buffer(emotion_label)

        # EXERCISE: if the emotion prediction process is not running,
        # create and start the thread to run emotion recognition.
        if not is_predicting:
            emotion_thread = Thread(target=predict_emotion)
            emotion_thread.start()

        # If the process is already running, do nothing - wait for it to finish.
    threaded_input.join()
    if emotion_thread.is_alive():
        emotion_thread.join()