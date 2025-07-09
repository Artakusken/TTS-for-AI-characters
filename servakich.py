import base64
import io
import time
import os
import requests
import json
import asyncio
import uvicorn

from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from fastapi import FastAPI, Response, HTTPException
from TTS.api import TTS

import whisper
import torch

MODEL = "deepseek/deepseek-chat-v3-0324"
API_KEY = "ваш API ключ"

file = open("PROMPT.txt", "r", encoding="utf8")
PROMPT = file.read()
file.close()

EXECUTOR = ThreadPoolExecutor(max_workers=2)
MODEL_PATH = "run\\training\\XTTSv2.0_MT-try3"
CONFIG_PATH = "run\\training\\XTTSv2.0_MT-try3\\config.json"
EMBEDDING_WAV = "C:\\RobotProject\\voice_data\\Tar_Sound1.wav"
OUTPUT_FOLDER_PATH = "C:\\RobotProject\\voice_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MY_TTS = TTS(model_path=MODEL_PATH,
          config_path=CONFIG_PATH,
          progress_bar=True).to(DEVICE)
RECOGNITION_MODEL = whisper.load_model("small")
app = FastAPI()


def process_content(content):
    return content.replace('<think>', '').replace('</think>', '')

def chat_stream(text):
    global API_KEY, MODEL, PROMPT
    print("Начало отправки вопроса для ИИ")
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Запрос для обсуждения: '{text}'"},
            ],
            "stream": True
        }

        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        ) as response:
            if response.status_code != 200:
                print("Ошибка API:", response.status_code)
                return

            sentence = ""
            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode('utf-8').replace('data: ', '')
                    try:
                        chunk_json = json.loads(chunk_str)
                        if "choices" in chunk_json:
                            content = chunk_json["choices"][0]["delta"].get("content", "")
                            if content:
                                cleaned = process_content(content)
                                sentence += cleaned
                                if any(mark in sentence for mark in [".", "!", "?", "..."]):
                                    global TOTAL
                                    TOTAL.append(sentence.replace("...", "."))
                                    print(f"Добавлено в TOTAL: {TOTAL[-1]}")
                                    sentence = ""
                    except Exception:
                        pass

    except Exception as e:
        print(f"Критическая ошибка в chat_stream: {e}")




def new_run_tts(generation: int):
    print("Запуск синтеза голоса")
    files_path = os.path.join(OUTPUT_FOLDER_PATH, str(generation))
    os.makedirs(files_path, exist_ok=True)

    while len(TOTAL) < 1:
        time.sleep(1)
        print("Ответ ИИ ещё не получен, ожидаю ещё секунду")
    n = 1

    while TOTAL:
        s = len(TOTAL)
        text = "".join(TOTAL[0:s])
        print(s, text, TOTAL)
        MY_TTS.tts_to_file(text=text, speaker_wav=EMBEDDING_WAV, language="ru",
                        file_path=f"{files_path}\\&audio{n:03d}.wav", split_sentences=False)
        n += 1
        for _ in range(s):
            if not TOTAL:
                break
            TOTAL.pop(0)

    with open(os.path.join(files_path, "COMPLETE"), "w") as end_file:
        end_file.write("")


def listen_file_local(audio_bytes: io.BytesIO):
    import time
    global RECOGNITION_MODEL

    start = time.time()
    try:
        audio_bytes.seek(0)
        audio_file_path = os.path.join("C:\\RobotProject", "output.wav")
        with open(audio_file_path, "wb") as wav_file:
            # Читаем все данные из BytesIO и записываем в файл
            wav_file.write(audio_bytes.read())
        # Локальное распознавание текста через whisper
        time.sleep(0.1)
        user_text = RECOGNITION_MODEL.transcribe(audio=audio_file_path, language="ru", verbose=True)
        if 'стоп' in user_text:
            return ""
        print(time.time() - start)
        return user_text['text']
    except Exception as e:
        print("Не удалось распознать речь", e)
    return ""


class TTSRequest(BaseModel):
    text: str

class AudioRequest(BaseModel):
    gen_num: int
    audio: str  # base64


@app.get("/")
def read_root():
    return "Как будто здесь что-нибудь будет."


@app.post("/listen")
async def listen(audio_request: AudioRequest):
    audio_data = base64.b64decode(audio_request.audio)
    user_request = listen_file_local(io.BytesIO(audio_data))

    global TOTAL
    TOTAL = []
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(EXECUTOR, chat_stream, user_request),
        loop.run_in_executor(EXECUTOR, new_run_tts, audio_request.gen_num)
    )
    
    # Остатки старого исполнения кода функции, где используется background_tasks: BackgroundTasks
    # ai_response = chat_stream(user_request)
    # background_tasks.add_task(run_tts, ai_response, audio_request.gen_num)
    return {"status": "processing_started", "gen_num": audio_request.gen_num}


@app.get("/speak/{gen_num}")
async def speak(gen_num):
    try:
        audios_path = os.path.join(OUTPUT_FOLDER_PATH, str(gen_num))

        if not os.path.exists(audios_path):
            return Response(status_code=404)

        files = os.listdir(audios_path)

        if len(files) > 0:
            if files[0] == "COMPLETE":
                os.remove(os.path.join(audios_path, files[0]))
                os.rmdir(os.path.join(OUTPUT_FOLDER_PATH, gen_num))
                print("Все аудиофайлы ответа были отправлены")
                return Response(status_code=204)
            file_path = os.path.join(audios_path, files[0])
            file_data = io.BytesIO()
            with open(file_path, "rb") as file_bytes:
                file_data = file_bytes.read()
            os.remove(file_path)
            return Response(content=file_data, media_type="audio/wav")
        else:
            return Response(status_code=202)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="26.222.34.150", port=8000)