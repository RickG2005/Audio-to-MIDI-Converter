import mido
import os

def create_midi_file(final_notes, audio_file, output_dir="midi_output"):
    """
    Convert detected notes into a MIDI file and save it.

    Parameters
    ----------
    final_notes : list
        List of tuples (notes, start_time, end_time), where
        notes = [(midi_number, note_name), ...].
        Each tuple represents one musical event or chord.

    audio_file : str
        Path to the original audio file (used for naming the output MIDI).

    output_dir : str
        Directory to save the generated MIDI file (default = 'midi_output').

    Returns
    -------
    str
        The path of the saved MIDI file.
    """

    # Ensure the output folder exists before writing
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base name of the input file for naming the MIDI
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    midi_filename = os.path.join(output_dir, base_name + ".mid")

    # Create a new MIDI file with standard resolution (480 ticks per beat)
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (default = 120 BPM)
    tempo = mido.bpm2tempo(120)

    # Calculate ticks per second based on tempo (120 BPM ≈ 2 beats per second)
    ticks_per_second = mid.ticks_per_beat * 2

    # Sort all detected notes in ascending order of start time
    final_notes = sorted(final_notes, key=lambda x: x[1])

    # Initialize reference for event timing in ticks
    current_tick = 0

    # Iterate through each detected note/chord group
    for notes, start_time, end_time in final_notes:
        duration = end_time - start_time  # Duration in seconds
        start_tick = int(start_time * ticks_per_second)
        duration_ticks = max(1, int(duration * ticks_per_second))  # Prevent zero duration

        # Delta time is the difference between this note's start and the previous event
        delta_time = start_tick - current_tick
        if delta_time < 0:
            delta_time = 0  # Avoid negative timing errors

        # Update the current tick to this event's start
        current_tick = start_tick

        # -----------------------------------------
        # NOTE-ON EVENTS
        # -----------------------------------------
        # Send a Note On message for each note in the group
        # The first message includes the computed delta time
        for midi, _ in notes:
            track.append(mido.Message('note_on', note=midi, velocity=80, time=delta_time))
            delta_time = 0  # Reset delta_time for subsequent simultaneous notes

        # -----------------------------------------
        # NOTE-OFF EVENTS
        # -----------------------------------------
        # After note duration, send a Note Off message for each note
        delta_time = duration_ticks
        for midi, _ in notes:
            track.append(mido.Message('note_off', note=midi, velocity=80, time=delta_time))
            delta_time = 0  # Only first note uses delta_time

    # Save the MIDI file to disk
    mid.save(midi_filename)

    # Confirmation message for user
    print(f"[✔] MIDI file saved at: {midi_filename}")

    return midi_filename
