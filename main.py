import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio
y, sr = librosa.load("samples/Unravel.wav")
print("Sample rate:", sr)
print("Audio length (in seconds):", len(y)/sr)

#Plot the waveform
plt.figure(figsize=(14,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Compute the Short-Time Fourier Transform
S = np.abs(librosa.stft(y))

# Pitch and magnitude extraction
pitches, magnitudes = librosa.piptrack(S=S, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), threshold=0.1)

# Loop through each frame
for i in range(0, pitches.shape[1], 10):    #loop
    index = magnitudes[:, i].argmax()   #selects max magnitude of freq bin in i time frame
    pitch = pitches[index, i]   #converts to pitch
    if pitch > 0:
        print(f"Time step {i}, Pitch: {pitch:.2f} Hz")      #displays pitch for time frame rounded to two dec places
