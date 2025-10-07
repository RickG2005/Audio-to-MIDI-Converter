import mido
import os

def create_midi_file(final_notes, audio_file, output_dir="midi_output"):
    """
    Convert detected notes into a MIDI file and save it.

    Parameters:
    -----------
    final_notes : list
        List of tuples (notes, start_time, end_time), where
        notes = [(midi_number, note_name), ...]
    audio_file : str
        Path to the original audio file (used for naming MIDI file).
    output_dir : str
        Directory to save the MIDI file (default = 'output').
    """

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Create MIDI filename
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    midi_filename = os.path.join(output_dir, base_name + ".mid")

    # Create MIDI file
    mid = mido.MidiFile(ticks_per_beat=480)  # standard resolution
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Assume 120 BPM for timing
    tempo = mido.bpm2tempo(120)
    ticks_per_second = mid.ticks_per_beat * 2  # 120 BPM ≈ 2 beats per second

    # Sort notes by start time
    final_notes = sorted(final_notes, key=lambda x: x[1])

    current_tick = 0

    for notes, start_time, end_time in final_notes:
        duration = end_time - start_time
        start_tick = int(start_time * ticks_per_second)
        duration_ticks = max(1, int(duration * ticks_per_second))

        # Delta time = difference from last event
        delta_time = start_tick - current_tick
        if delta_time < 0:
            delta_time = 0
        current_tick = start_tick

        # Add note-on events
        for midi, _ in notes:
            track.append(mido.Message('note_on', note=midi, velocity=80, time=delta_time))
            delta_time = 0  # only first message gets the delta time

        # Add note-off events after duration
        delta_time = duration_ticks
        for midi, _ in notes:
            track.append(mido.Message('note_off', note=midi, velocity=80, time=delta_time))
            delta_time = 0

    # Save the MIDI file
    mid.save(midi_filename)
    print(f"[✔] MIDI file saved at: {midi_filename}")
    return midi_filename
