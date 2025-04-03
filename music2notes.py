import torch
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def vocall(input_audio):

    inputpath = input_audio
    model_output, midi_data, note_events = predict(inputpath, onset_threshold=0.85, frame_threshold=0.3,

                                                 maximum_frequency=1000)
    midi_data.write('output.midi')
    notes = []

    # Перебираем все инструменты (треки)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
            print(note.pitch)
    notes_new = []
    last_note = 'zjdajh'
    for i in notes:
        if last_note != i:
            notes_new.append(i)
            # print(pretty_midi.note_number_to_name(i.pitch), end=' ')
        last_note = i

    differences = []
    for i in range(len(notes_new) - 1):
        differences.append(str(notes_new[i + 1] - notes_new[i]))

    return differences

#print(vocall("elka.wav"))