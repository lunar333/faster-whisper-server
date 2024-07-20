import requests
from http import HTTPStatus
import json  # 导入json模块来处理JSON数据
import dashscope

def correct_speech_recognition_error(input_text):
    # 拼接固定的prompt与输入的文本
    content_text = f'''请你对语音识别的结果中发音相似的错误进行纠正，根据语义纠正某些发音相似的识别错误，但是不能改变总字数，并且如果要进行纠正，纠正后的地方，发音要和原来句子相似。例如，输入“那些桩架田园在果果园里感觉太亲切了”根据语义和发音相似性，可以发现桩架和庄稼发音比较相似，但是庄稼更符合语义，最后输出为：那些庄稼田园在果果眼里感觉太亲切了！
    输入：{input_text}'''
    
    messages = [{'role': 'user', 'content': content_text}]
    
    # 调用API
    response = dashscope.Generation.call("qwen-max",
                                messages=messages,
                                api_key='sk-e5c03fe7724d4bb5824774cd8661675f',
                                result_format='message',  # 结果格式为"message"
                                stream=False,  # 关闭流输出
                                incremental_output=False  # 关闭增量输出
                                )
    
    # 解析响应数据
    if isinstance(response, str):
        response = json.loads(response)  # 如果响应是字符串，解析为JSON
    return response

# 定义用于发送请求和处理响应的函数
def transcribe_audio_and_correct(url, audio_file_path):
    files = {'file': open(audio_file_path, 'rb')}
    response = requests.post(url, files=files)
    
    # 确保文件在发送请求后被关闭
    files['file'].close()

    # 检查请求的响应状态
    if response.status_code == HTTPStatus.OK:
        response_data = response.json()
        print(response_data)  # 打印响应数据以检查结构
        
        if 'transcription' in response_data and len(response_data['transcription']) > 0 and 'text' in response_data['transcription'][0]:
            transcription_text = response_data['transcription'][0]['text']
            # 调用API函数处理识别结果
            corrected_text = correct_speech_recognition_error(transcription_text)
            print("Corrected Text:")
            print(corrected_text)
        else:
            print("No transcription text available or error in transcription.")
    else:
        print("Failed to transcribe the audio.")
        print(response.text)

# 设置URL和音频文件路径
url = 'http://127.0.0.1:5000/transcribe'
audio_file_path = '/home/zhonghuihang/data/db-6/fear/wav/250703.wav'

# 调用函数
transcribe_audio_and_correct(url, audio_file_path)