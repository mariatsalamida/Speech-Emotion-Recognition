# Speech Emotion Recognition
Real‐time speech emotion recognition using a pretrained Wav2Vec2 model with live waveform plotting
**Note:** This repository contains a Python script that captures live microphone audio, continuously predicts the speaker’s emotion using a pretrained Wav2Vec2 model, and plots the audio waveform in real time with the current emotion as the title. The code was originally written as part of a master’s‐level exercise.
## Description

`speech_emotion_recognition.py` records audio from your microphone in 2‐second sliding windows, runs an emotion classifier (`superb/wav2vec2-base-superb-er`) on each buffer, and displays a live matplotlib plot of the waveform with the predicted emotion shown as the window title. Press **Enter** at any time to stop recording and exit gracefully.

Key behaviors (inferred from the source code):

- Lists all available audio devices at startup (so you can verify your microphone and speaker indices).  
- By default, it picks the first input device it finds. If that’s not your microphone, you can manually set the device index.  
- Maintains a 2‐second rolling buffer of audio samples at 16 kHz.  
- Launches a separate thread to run emotion prediction on each buffer, ensuring the plotting thread never blocks.  
- Continuously updates a matplotlib figure, plotting the raw waveform and showing the current predicted emotion (or a placeholder if no prediction is ready).  
- Exits when you press **Enter** (it uses a secondary thread to listen for the Return key).

## Folder Structure

- `speech_emotion_recognition.py`  
  Main Python script that captures live audio and predicts emotion.  
- `requirements.txt`  
  All third‐party Python dependencies needed to run the script.  
- `.gitignore`  
  Common files/folders to ignore (e.g. `__pycache__`, virtual‐env folders, etc.).  
- `README.md`  
  This file.

  ## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/speech-emotion-recognition.git
   cd speech-emotion-recognition

2. **(Recommended) Create a virtual environment**
  python3 -m venv .venv
  .venv\Scripts\activate    # On Windows
  #OR 
  source .venv/bin/activate  # On Linux/macOS
 
3. **Install dependencies**
   pip install -r requirements.txt
   **Note:** If you run into issues installing torch or transformers, refer to their   official installation guides. For a CPU‐only install, for example:
  pip install torch==2.0.0+cpu torchvision==0.15.0+cpu -f       https://download.pytorch.org/whl/torch_stable.html
  pip install transformers sounddevice numpy matplotlib
4. **Usage**
   Run the script python3 speech_emotion_recognition.py
   Select audio device: mic_device_id = 0  # change to your device’s index
   View output:
     A Matplotlib window displays a live 2-second waveform.
     The window title shows “Emotion: <LABEL>” (e.g., “happy,” “sad”).
     Press Enter in the terminal to quit.

5. **Dependencies**
Listed in requirements.txt:
sounddevice>=0.4.8
numpy>=1.24.0
matplotlib>=3.5.0
torch>=2.0.0
transformers>=4.30.0

License
MIT License © 2025 MARIA TSALAMIDA
 
