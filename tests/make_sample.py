"""Built-in sample melody (C-major scale up & down) for smoke tests / demos."""

from models import Note

# MIDI pitches for a C-major scale (C4..C5..C4)
_PITCHES = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]


def sample_notes():
    notes = []
    for i, p in enumerate(_PITCHES):
        notes.append(Note(pitch=p, onset=i * 0.5, duration=0.5, velocity=100))
    return notes


if __name__ == "__main__":
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="sample")
    for n in sample_notes():
        inst.notes.append(pretty_midi.Note(velocity=n.velocity,
                                            pitch=n.pitch,
                                            start=n.onset,
                                            end=n.onset + n.duration))
    midi.instruments.append(inst)
    midi.write("sample.mid")
    print("wrote sample.mid")
