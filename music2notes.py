#import torch
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
#import pretty_midi

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def vocall(input_audio):

    inputpath = input_audio
    model_output, midi_data, note_events = predict(inputpath, onset_threshold=0.85, frame_threshold=0.3,
                                                   maximum_frequency=1000)
    notes = midi_data.instruments[0].notes
    notes_new = []
    last_note = 'zjdajh'
    for i in notes:
        if last_note != i:
            notes_new.append(i.pitch)
            # print(pretty_midi.note_number_to_name(i.pitch), end=' ')
        last_note = i

    min_num = min(notes_new)  # Находим минимальное число
    notes_new_new = [x - min_num for x in notes_new]
    return notes_new_new