import pyaudio
import wave
import time
import base64
import requests
from pynput.keyboard import Key, Listener
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
from threading import Thread


CHUNK = 1024  # куски размером в 1024 сэмпла
FRT = pyaudio.paInt16  # битрейт
CHAN = 1  # количество каналов записи звука
RT = 24000  # частота дискретизации
MAX_REC_SEC = 50  # максимальная длина записи
OUTPUT_FILE_NAME = "output"
SERVER = "http://26.222.34.150:8000"

recording_on = False
session_on = True
wait_for_response = False
robot_is_speaking = False


def stop(key):
    global recording_on, session_on, wait_for_response
    if key == Key.space and not wait_for_response:
        if recording_on:
            print("\tНажат пробел, конец записи")
            recording_on = False
        else:
            print("\tНажат пробел, начало записи")
            recording_on = True
    if key == Key.tab and not wait_for_response:
        print("КОНЕЦ СЕССИИ")
        session_on = False


def end():
    return False


def key_listener():
    listener = Listener(on_press=stop)
    listener.start()


def record_audio(micro_index=1):
    n = 0
    print("НАЧАЛО СЕССИИ")
    audio = pyaudio.PyAudio()
    while session_on:
        stream = audio.open(format=FRT, channels=CHAN,
                            rate=RT, frames_per_buffer=CHUNK, input=True, input_device_index=micro_index)

        audio_frames = []
        print("Начните запись")
        while not recording_on and session_on:
            time.sleep(0.1)

        while len(audio_frames) < int(RT / CHUNK * MAX_REC_SEC) and recording_on and session_on:
            audio_frames.append(stream.read(CHUNK))
        print("Запись окончена")
        stream.stop_stream()
        stream.close()

        if not session_on:
            return
        wav_io = BytesIO()
        with wave.open(wav_io, 'wb') as w:
            w.setnchannels(CHAN)
            w.setsampwidth(audio.get_sample_size(FRT))
            w.setframerate(RT)
            w.writeframes(b''.join(audio_frames))
        wav_io.seek(0)
        wav_data = wav_io.getvalue()
        start = time.time()
        try:
            requests.post(SERVER + "/listen",
                          json={
                              "gen_num": n,
                              "audio": base64.b64encode(wav_data).decode("utf-8")},
                          headers={"Content-Type": "application/json"},
                          timeout=0.1)
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(1)
        while True:
            time.sleep(0.1)
            response = requests.get(SERVER + f"/speak/{n}")
            if response.status_code == 200:
                print("Получено аудио через", time.time() - start)
                ai_audio = AudioSegment.from_wav(BytesIO(response.content))
                play(ai_audio)
                start = time.time()
            elif response.status_code == 202:
                  time.sleep(1)
            else:
                print(response.status_code)
                break
        n += 1
    audio.terminate()


def micro_info_print():
    for d in range(pyaudio.PyAudio().get_device_count()):
        print(pyaudio.PyAudio().get_device_info_by_index(d))


if __name__ == '__main__':
    key_thread = Thread(target=key_listener, daemon=True)
    audio_thread = Thread(target=record_audio)
    key_thread.start()
    audio_thread.start()
    key_thread.join()
    audio_thread.join()
    while session_on:
        True