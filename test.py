from faster_whisper import WhisperModel
import time

model_size = "faster-whisper-small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
s = time.time()

segments, info = model.transcribe("/home/zhonghuihang/data/db_16k/000001.wav", 
                                  beam_size=5,
                                  vad_filter=True,
                                  vad_parameters=dict(min_silence_duration_ms=500), # 默认2s
                                  initial_prompt="以下是中文简体的句子。"
                                  ) #以下是普通話的句子。

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
t_cost = time.time()-s
print("Faster Whisper model cost %.2f second"%(t_cost))

