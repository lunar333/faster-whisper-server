import requests

url = 'http://127.0.0.1:5000/transcribe'
audio_file_path = '/root/faster-whisper/test.wav'

files = {'file': open(audio_file_path, 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    print("Transcription Results:")
    print(response.json())
else:
    print("Failed to transcribe the audio.")
    print(response.text)