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
times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
detected_notes = []
for i in range(0, pitches.shape[1], 50):    #loop
    index = magnitudes[:, i].argmax()   #selects max magnitude of freq bin in i time frame
    pitch = pitches[index, i]   #converts to pitch
    if pitch > 0:
        midi = librosa.hz_to_midi(pitch)
        note_name = librosa.midi_to_note(midi)
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

# Groups time intervals together for the same note
grouped_notes = []
prev_note = None
start_time = None
for i, (times, midi_notes, note_name) in enumerate(detected_notes):
    if note_name != prev_note:
        if prev_note is not None:
            grouped_notes.append((prev_note, start_time, times))
        start_time = times
        prev_note = note_name

if prev_note is not None:
    end_time = detected_notes[-1][0] + (detected_notes[-1][0]-detected_notes[-2][0])
    grouped_notes.append((prev_note, start_time, end_time))

for note, start_time, end_time in grouped_notes:
    duration = end_time - start_time
    print(f"{note}: {start_time:.2f} (duration = {duration:.2f})")

# Plot midi notes against time
plt.figure(figsize=(12,5))
for prev_note, start_time, end_time in grouped_notes:
    midi = librosa.note_to_midi(prev_note)
    plt.hlines(midi, start_time, end_time, colors="blue", linewidth = 4) 
    plt.text((start_time + end_time)/2, midi + 0.5, prev_note, ha = "center", va = "bottom", fontsize = 8)
plt.title("MIDI notes over time")
plt.xlabel("Time(s)")
plt.ylabel("MIDI note")
plt.grid(True)
plt.show()
