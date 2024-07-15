from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import time

app = Flask(__name__)

model_size = "faster-whisper-small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        audio_path = '/root/faster-whisper/save/audio.wav'  # 实际环境中应该安全地处理文件路径和名字
        file.save(audio_path)
        
        s = time.time()
        segments, info = model.transcribe(audio_path, 
                                          beam_size=5,
                                          vad_filter=True,
                                          vad_parameters=dict(min_silence_duration_ms=500),
                                          initial_prompt="以下是中文简体的句子。")
        
        transcription = []
        for segment in segments:
            transcription.append({"start": segment.start, "end": segment.end, "text": segment.text})
        
        t_cost = time.time() - s
        response = {
            "language": info.language,
            "language_probability": info.language_probability,
            "transcription": transcription,
            "processing_time": t_cost
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)