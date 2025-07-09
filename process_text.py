import os
from scipy.io import wavfile
from pydub import AudioSegment, silence

VOICE_DATA_PATH = "C:\\RobotProject\\voice_data"
VOICE_CHUNKS_PATH = "C:\\RobotProject\\voice_chunks"


def get_audio(audio_folder) -> list:
    """ returns list with audio segments """
    audios = []
    for wav in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, wav)
        if os.path.isfile(file_path) and file_path.split(".")[-1] == "wav":
            audios.append(AudioSegment.from_wav(file_path))
    return audios

def split_audio(audio_pieces, chunks_path):
    # Split on silence (min 400ms silence, threshold -60 dB)
    n = 0
    for file in audio_pieces:
        chunks = silence.split_on_silence(
            file,
            min_silence_len=800,
            silence_thresh=-55,
            keep_silence=150  # Keep 100ms at edges
        )
        # specs                 output length (in secs)
        # 300, -70, 150 ***** >0 - 28, > 10 - 10
        # 200, -70, 150 ***** >0 - 59, > 10 - 8
        # 500, -70, 150 ***** >0 - 4, > 10 - 25
        # 500, -50, 150 ***** >0 - 45, > 10 - 3
        # 800, -50, 150 ***** >0 - 6, > 10 - 27
        # 800, -60, 150 ***** >0 - 1, > 10 - 37

        for chunk in chunks:
            chunk = chunk.set_frame_rate(24000)
            chunk = chunk.set_sample_width(2)
            chunk.export(f"{chunks_path}\\TM_{n:03d}.wav", format="wav")
            n += 1
    print(n)


def audio_duration(audio_path):
    """ prints number of all files length """
    len_dist = {0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0}

    total = n = 0
    for file in os.listdir(audio_path):
        rate, data = wavfile.read(audio_path + "\\" + file)
        total += len(data)/rate
        n += 1
        if len(data)//rate not in len_dist.keys():
            len_dist[len(data)//rate] = 1
        else:
            len_dist[len(data)//rate] += 1

    print(total, n)
    for i in sorted(len_dist.keys()):
        print(i, len_dist[i], str(round(len_dist[i]/n*100, 4)).replace(".", ","))


def normalize_text(text: str):
    text = text.replace("...", ".")
    text = text.replace("θ", "тэта")
    text = text.replace("x", "икс")
    text = text.replace(" х ", " икс ")
    text = text.replace("y", "игрек")
    text = text.replace("gps", "джи пи эс")
    text = text.replace("1", "один")
    text = text.replace("2", "два")
    text = text.replace("3", "три")
    text = text.replace("4", "четыре")
    text = text.replace("5", "пять")
    text = text.replace("6", "шесть")
    text = text.replace("7", "семь")
    text = text.replace("8", "восемь")
    text = text.replace("9", "девять")
    text = text.replace("10", "десять")
    return text


def transcription(audio_path):
    import whisper

    model = whisper.load_model("turbo")
    manifesto = open("C:\\RobotProject\\manifest.txt", mode="w", encoding="UTF-8")
    for file in os.listdir(audio_path):
        rate, data = wavfile.read(audio_path + "\\" + file)
        if 1 < len(data) // rate < 12:  # length filter
            print(file[:-4], end="|", file=manifesto)
            result = model.transcribe(os.path.join(audio_path, file), language="ru", verbose=True)["text"][1::]
            # result = model.transcribe(os.path.join(audio_path, file), language="ru", word_timestamps=True, suppress_tokens=[])["text"][1::]
            print(result, end="|", file=manifesto)
            result = result.lower()
            norm_result = normalize_text(result)
            print(norm_result, file=manifesto)
    manifesto.close()


if __name__ == "__main__":
    audios = get_audio(VOICE_DATA_PATH)
    split_audio(audios, VOICE_CHUNKS_PATH)
    audio_duration(VOICE_CHUNKS_PATH)
    transcription(VOICE_CHUNKS_PATH)
