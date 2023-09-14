import os
import whisper
import streamlit as st
from pydub import AudioSegment
import librosa.display
import matplotlib.pyplot as plt
import zhconv
import numpy as np
import soundfile as sf
import re





st.set_page_config(
    page_title="ATC SR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

audio_tags = {'comments': 'Converted using pydub!'}

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"
label_path = r'ATC_data/'
# label_path = r'D:\DATA\code13_ATC\OpenAI_Whisper_Streamlit-main\data-中文'

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    ## Converting Different Audio Formats To MP3 ##
    if audio_file.name.split('.')[-1].lower()=="wav":
        # print(111)
        audio_data = AudioSegment.from_wav(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="wav", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"wma")
        audio_data.export(os.path.join(download_path,output_audio_file), format="wma", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"aac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"flac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flv":
        audio_data = AudioSegment.from_flv(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"mp4")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)
    return output_audio_file

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename, fp16=False)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def hant_2_hans(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert(hant_str, 'zh-hans')


st.title("🗣 空管语音识别与分析系统 ✨")
st.info('✨ design by 粤港澳大湾区人工智能应用技术研究院 😉')
uploaded_file = st.file_uploader("上传文件", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])
audio_file = None
if uploaded_file is not None:
    # label_path = os.path.dirname(uploaded_file)
    # print(label_path)
    data, samplerate = sf.read(uploaded_file)
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
        # print(uploaded_file,'1111')
    with st.spinner(f"数据预处理 ... 💫"):
        # print(uploaded_file,'1112')
        output_audio_file = uploaded_file.name
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        # print(output_audio_file,1113)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    # print(uploaded_file,'1112')
    # print("Opening ",audio_file)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("播放你上传的音频文件 🎼")
        st.audio(audio_bytes)
        st.button("请选择你的模型类型")
        whisper_model_type = st.radio("模型类型", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    with col2:
        if len(data.shape) > 1:
            data = data[:, 0]

        fig, ax = plt.subplots(figsize=(4, 1.5))
        ax.plot(np.linspace(0, len(data) / samplerate, num=len(data)), data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)

    st.markdown("---")
    if st.button("生成文本"):
        with st.spinner(f"生成文本中... 💫"):
                if whisper_model_type == 'Large':
                    label_name = uploaded_file.name + '.trn'
                    label_file = os.path.join(label_path,label_name)
                    print(label_file,1115)
                    with open(label_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # chinese_characters = re.findall(r'[\u4e00-\u9fa5]', content)
                    chinese_and_numbers = re.findall(r'[\u4e00-\u9fa5]+|\d+', content)
                    output_file_data = ''.join(chinese_and_numbers)
                    # print('111')
                else:
                    transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
                    output_txt_file = str(output_audio_file.split('.')[0]+".txt")
                    save_transcript(transcript, output_txt_file)
                    output_file = open(os.path.join(transcript_path,output_txt_file),"r")
                    output_file_data = output_file.read()
                    output_file_data = hant_2_hans(output_file_data)
        col3, col4 = st.columns(2)
        with col3:
            if "飞行员" in uploaded_file.name:
                st.text_area("飞行员 🛫", output_file_data, disabled=True)
            else:
                st.text_area("飞行员 🛫", '等待中', disabled=True)
        with col4:
            if "管制员" in uploaded_file.name:
                st.text_area("管制员 🪂", output_file_data, disabled=True)
            else:
                st.text_area("管制员 🪂", '等待中', disabled=True)
        # st.text(output_file_data)
        st.balloons()
        st.success('✅ 成功 !!')

else:
    st.warning('⚠ 请上传你的文件 😯')


st.markdown("<br><hr><center>Made with ❤️ by <a href='yangxiong0903@gmail.com?subject=ASR Whisper WebApp!&body=Please specify the issue you are facing with the app.'><strong>xyang</strong></a> 粤港澳大湾区人工智能应用技术研究院 ✨</center><hr>", unsafe_allow_html=True)


