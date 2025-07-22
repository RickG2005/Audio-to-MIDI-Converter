import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Load the audio
y, sr = librosa.load("samples/Unravel.wav", duration=10)
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
times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
detected_notes = []
for i in range(0, pitches.shape[1], 10):    #loop
    index = magnitudes[:, i].argmax()   #selects max magnitude of freq bin in i time frame
    pitch = pitches[index, i]   #converts to pitch
    if pitch > 0:
        midi = librosa.hz_to_midi(pitch)
        note_name = librosa.midi_to_note(midi)
        print(f"Time {i}, Pitch: {pitch:.2f} Hz, Note: {note_name}")
        detected_notes.append((times[i], int(round(midi)), note_name))

# Plotting the spectrogram
S_db = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(14,6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# Plot midi notes against time
times, midi_notes, note_name = zip(*detected_notes)
plt.figure(figsize=(12,5))
plt.scatter(times, midi_notes, color = "blue", s = 8)
for t,m,n in zip(times, midi_notes, note_name):
    plt.text(t, m+0.5, n)
plt.title("Detected MIDI notes over time")
plt.xlabel("Time(s)")
plt.ylabel("MIDI note")
plt.grid(True)
plt.show()
