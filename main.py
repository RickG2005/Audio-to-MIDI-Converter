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
hop_length = 256
S = np.abs(librosa.stft(y, hop_length=hop_length))

# Pitch and magnitude extraction
pitches, magnitudes = librosa.piptrack(S=S, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), threshold=0.1)

# Loop through each frame
times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr,hop_length=hop_length)
detected_notes = []
for i in range(pitches.shape[1]):    #loop
    index = magnitudes[:, i].argmax()   #selects max magnitude of freq bin in i time frame
    pitch = pitches[index, i]   #converts to pitch
    if pitch > 0 and magnitudes[index, i] > 0.1:
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
tolerance = 0.5
for i, (times, midi_notes, note_name) in enumerate(detected_notes):
    if note_name != prev_note:
        if prev_note is not None:
            grouped_notes.append((prev_note, start_time, times))
        start_time = times
        prev_note = note_name
    else:   #allows short breaks in notes
        if i > 0 and times - detected_notes[i-1][0] > tolerance:
            grouped_notes.append((prev_note, start_time, detected_notes[i-1][0]))
            start_time = times

if prev_note is not None:
    end_time = detected_notes[-1][0] + (detected_notes[-1][0]-detected_notes[-2][0])
    grouped_notes.append((prev_note, start_time, end_time))

for note, start_time, end_time in grouped_notes:
    duration = end_time - start_time
    print(f"{note}: {start_time:.2f} (duration = {duration:.2f})")

# Plot piano roll of midi notes against time
plt.figure(figsize=(14, 6))  # Slightly wider figure for clarity

for note_name, start_time, end_time in grouped_notes:
    midi = librosa.note_to_midi(note_name)  # Convert note to MIDI number
    duration = end_time - start_time        # Calculate how long the note lasts

    # Draw a horizontal bar like a piano roll
    plt.broken_barh([(start_time, duration)], (midi - 0.4, 0.8), facecolors='skyblue')

# Set y-ticks to note names (C4, D#4, etc.)
midi_values = [librosa.note_to_midi(n) for n, _, _ in grouped_notes]
min_midi = min(midi_values) - 2
max_midi = max(midi_values) + 2
yticks = list(range(min_midi, max_midi + 1))
yticklabels = [librosa.midi_to_note(m) for m in yticks]
plt.yticks(yticks, yticklabels)

plt.xlabel("Time (s)")
plt.ylabel("Note")
plt.title("Piano Roll View")
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
