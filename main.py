import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import mido
import time

# Load the audio
y, sr = librosa.load("samples/Unravel.wav")
print("Sample rate:", sr)
print("Audio length (in seconds):", len(y) / sr)

# Plot the waveform
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Compute the Short-Time Fourier Transform
hop_length = 256
S = np.abs(librosa.stft(y, hop_length=hop_length))

# Pitch and magnitude extraction for polyphonic detection
pitches, magnitudes = librosa.piptrack(S=S, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Loop through each frame to detect all pitches above a certain magnitude
times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
detected_notes = []
polyphonic_threshold = 0.5 
for i in range(pitches.shape[1]):
    valid_pitches_indices = np.where(magnitudes[:, i] > polyphonic_threshold)
    
    for index in valid_pitches_indices[0]:
        pitch = pitches[index, i]
        if pitch > 0:
            midi = librosa.hz_to_midi(pitch)
            note_name = librosa.midi_to_note(midi)
            detected_notes.append((times[i], int(round(midi)), note_name))

# --- New Harmonic Filtering Logic ---
def filter_harmonics(detected_notes):
    filtered_notes = []
    notes_by_time = {}
    for time, midi, name in detected_notes:
        time_rounded = round(time, 2)
        if time_rounded not in notes_by_time:
            notes_by_time[time_rounded] = {'notes': [], 'start': time}
        notes_by_time[time_rounded]['notes'].append((midi, name))

    for time_rounded, data in notes_by_time.items():
        notes = data['notes']
        notes.sort(key=lambda x: x[0])  # Sort by MIDI value (pitch)
        
        is_fundamental = [True] * len(notes)
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                midi_fundamental = notes[i][0]
                midi_harmonic = notes[j][0]
                
                if (midi_harmonic - midi_fundamental) % 12 == 0:  # Octave
                    is_fundamental[j] = False
                elif abs(librosa.midi_to_hz(midi_harmonic) / librosa.midi_to_hz(midi_fundamental) - 1.5) < 0.05:  # Perfect fifth
                    is_fundamental[j] = False
                elif abs(librosa.midi_to_hz(midi_harmonic) / librosa.midi_to_hz(midi_fundamental) - 1.25) < 0.05: # Major third
                    is_fundamental[j] = False

        for i, (midi, name) in enumerate(notes):
            if is_fundamental[i]:
                filtered_notes.append((data['start'], midi, name))
                
    return filtered_notes

filtered_notes = filter_harmonics(detected_notes)

# Plotting the spectrogram
S_db = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# --- Note Grouping Logic ---
filtered_notes.sort(key=lambda x: x[0]) 
grouped_notes = [] 
if filtered_notes: 
    notes_at_time = {} 
    for time, midi, name in filtered_notes: 
        time_rounded = round(time, 2) 
        if time_rounded not in notes_at_time: 
            notes_at_time[time_rounded] = {'notes': [], 'start': time} 
        if (midi, name) not in notes_at_time[time_rounded]['notes']: 
            notes_at_time[time_rounded]['notes'].append((midi, name)) 

    previous_notes = None 
    start_time = None 
    sorted_times = sorted(notes_at_time.keys()) 
    
    for i, time_rounded in enumerate(sorted_times): 
        current_notes = tuple(sorted(notes_at_time[time_rounded]['notes'])) 
        
        if previous_notes is None: 
            start_time = notes_at_time[time_rounded]['start'] 
            previous_notes = current_notes 
            continue 
            
        if current_notes != previous_notes or (time_rounded - sorted_times[i-1] > 0.1): 
            end_time = sorted_times[i-1] 
            grouped_notes.append((previous_notes, start_time, end_time)) 
            
            start_time = notes_at_time[time_rounded]['start'] 
            previous_notes = current_notes 
            
    end_time = sorted_times[-1] 
    grouped_notes.append((previous_notes, start_time, end_time)) 

# --- New Duration Filtering Logic --- 
final_notes = [] 
min_duration = 0.1 # Minimum duration in seconds 
for notes, start_time, end_time in grouped_notes: 
    duration = end_time - start_time 
    if duration > min_duration: 
        final_notes.append((notes, start_time, end_time)) 

for notes, start_time, end_time in final_notes: 
    duration = end_time - start_time 
    note_names = ", ".join([note for _, note in notes]) 
    print(f"Notes: {note_names} -> Start: {start_time:.2f}s, Duration: {duration:.2f}s") 

# Plot piano roll of midi notes against time 
plt.figure(figsize=(14, 6)) 

for notes, start_time, end_time in final_notes: 
    duration = end_time - start_time 
    for midi, note_name in notes: 
        plt.broken_barh([(start_time, duration)], (midi - 0.4, 0.8), facecolors='skyblue') 

# Set y-ticks to note names 
all_midi = sorted(list(set(midi for note_group, _, _ in final_notes for midi, _ in note_group))) 
yticks = all_midi 
yticklabels = [librosa.midi_to_note(m) for m in yticks] 
plt.yticks(yticks, yticklabels) 

plt.xlabel("Time (s)") 
plt.ylabel("Note") 
plt.title("Piano Roll View (Polyphonic)") 
plt.grid(True, axis='x', linestyle='--', alpha=0.6) 
plt.tight_layout() 
plt.show()

from mido import Message, MidiFile, MidiTrack, bpm2tempo

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

tempo = bpm2tempo(120)  # 120 BPM
track.append(mido.MetaMessage('set_tempo', tempo=tempo))
ppq = mid.ticks_per_beat

def seconds_to_ticks(seconds, tempo, ticks_per_beat):
    beats = seconds / (tempo / 1_000_000)  # seconds → beats
    return int(round(beats * ticks_per_beat))

# Sort notes by start time
final_notes_sorted = sorted(final_notes, key=lambda x: x[1])

current_tick = 0  # where we are in MIDI time

for notes, start_time, end_time in final_notes_sorted:
    start_ticks = seconds_to_ticks(start_time, tempo, ppq)
    end_ticks = seconds_to_ticks(end_time, tempo, ppq)

    # Δ time between current and new note start
    delta_start = max(0, start_ticks - current_tick)

    # --- NOTE ONs ---
    for i, (midi, _) in enumerate(notes):
        track.append(
            Message('note_on', note=midi, velocity=64, time=delta_start if i == 0 else 0)
        )

    # --- NOTE OFFs ---
    duration_ticks = max(1, end_ticks - start_ticks)
    for i, (midi, _) in enumerate(notes):
        track.append(
            Message('note_off', note=midi, velocity=64, time=duration_ticks if i == 0 else 0)
        )

    # Update pointer
    current_tick = end_ticks

# Save MIDI file
mid.save("midi_output/output.mid")
print("✅ MIDI file saved as output.mid")
