# Audio to MIDI Converter (Polyphonic Pitch Detection using Librosa)

This project converts **polyphonic audio** (like piano or instrumental tracks) into **MIDI files** by analyzing frequencies, extracting pitches, and grouping detected notes with refined filtering logic.
It’s built using **Python**, **Librosa**, and **Mido**, and features a full workflow — from **audio preprocessing** and **spectrogram visualization** to **harmonic filtering**, **note grouping**, and **MIDI file generation**.

---

## Features

* **Audio Analysis**

  * Loads `.wav` audio files and visualizes the waveform.
  * Computes **Short-Time Fourier Transform (STFT)** to extract time-frequency data.

* **Pitch Detection**

  * Uses `librosa.piptrack()` for **polyphonic pitch estimation**.
  * Filters out weak signals using magnitude thresholds.

* **Harmonic Filtering**

  * Identifies and removes harmonic overtones (octaves, fifths, and thirds).
  * Retains only **fundamental notes** to improve MIDI accuracy.

* **Note Grouping Logic**

  * Groups notes played at similar timestamps.
  * Detects note durations and eliminates short-duration noise.

* **MIDI Conversion**

  * Converts detected notes to **MIDI events** using `mido`.
  * Maps timing accurately based on tempo (120 BPM default).
  * Generates a playable `.mid` file compatible with DAWs like FL Studio, Ableton, or Logic Pro.

* **Visualizations**

  * Waveform plot
  * Spectrogram (dB scale)
  * Piano roll visualization of detected notes

---

## Tech Stack

* **Python 3.x**
* **Libraries:**

  * `librosa` – audio processing and pitch detection
  * `matplotlib` – waveform and spectrogram visualization
  * `numpy` – signal and matrix computations
  * `mido` – MIDI message creation and export

---

## Project Structure

```
├── samples/
│   └── Unravel.wav            # Input audio file
├── midi_output/
│   └── output.mid             # Generated MIDI file
├── audio_to_midi.py           # Main script
└── README.md
```

---

## How It Works

1. **Load the Audio**
   The audio is loaded with `librosa.load()` and resampled if needed.
   The waveform is displayed for visual inspection.

2. **Extract Frequency Information**
   STFT computes the time–frequency matrix.
   `piptrack()` extracts pitch candidates and their magnitudes for each frame.

3. **Filter Harmonics**
   Removes harmonic overtones like octaves, fifths, and thirds to keep only the main pitch.

4. **Group Notes**
   Notes close in time are grouped together to detect chords or sustained notes.
   Filters out notes shorter than a threshold (`min_duration`).

5. **Convert to MIDI**
   Notes are converted to MIDI events with correct timing.
   The output file (`output.mid`) is saved to `midi_output/`.

6. **Visualize**
   Piano roll shows time vs. detected notes.
   Spectrogram displays frequency intensity across time.

---

## Example Output

**Detected Notes (Console Example):**

```
Notes: E4, G#4, B4 -> Start: 2.10s, Duration: 0.52s
Notes: F#4, A4 -> Start: 3.15s, Duration: 0.45s
```

**MIDI Output:**
File generated: `midi_output/output.mid`
You can open this in any DAW to hear the converted performance.

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/audio-to-midi.git
   cd audio-to-midi
   ```

2. Install dependencies:

   ```bash
   pip install librosa mido matplotlib numpy
   ```

3. Place your `.wav` file in the `samples/` folder.

4. Run the script:

   ```bash
   python audio_to_midi.py
   ```

5. Check the generated MIDI file in `midi_output/output.mid`.

---

## Future Improvements

* Integrate onset detection for more precise note timing.
* Apply machine learning models for improved pitch tracking.
* Add a GUI interface for drag-and-drop audio conversion.
* Enable multi-track MIDI export (split by instrument or frequency band).

---

## Learnings

This project helped explore:

* How to detect pitches in **polyphonic audio** (a challenging DSP problem).
* Techniques for **harmonic suppression** and **time-domain grouping**.
* MIDI structure and timing conversion using **beats to ticks mapping**.

---

## License

This project is open-source under the **MIT License** — feel free to modify and expand it.

---

Would you like me to make this version a bit **shorter and more concise** (still professional, but about half the length) for your GitHub profile page? It’ll be perfect if you want recruiters to skim and get the key info fast.
