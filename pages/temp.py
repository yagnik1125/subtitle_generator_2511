import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
# import googletrans 
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import tempfile
import wave
import io
import subprocess
import cv2
import ffmpeg
import copy
import time
import json
import base64

# Initialize the Groq client
client = Groq(api_key="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")


# Streamlit frontend for audio input and translation
st.title("Audio Translation App")

# Audio file input
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac", "m4a"])
if uploaded_file:
    st.audio(uploaded_file, format="wav")
mic_audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🎙️ Stop Recording", key='recorder')
if mic_audio:
    st.write("mic audio through bytes")
    st.audio(mic_audio['bytes'], format='wav')
mic_audio_file_name='temp_mic_audio.wav'

if mic_audio:
    # Get the byte data from the audio recorder
    audio_bytes = mic_audio['bytes']
    audio_file_like = io.BytesIO(audio_bytes)
    with wave.open(mic_audio_file_name, 'wb') as wav_file:
        sample_width = 2  # Sample width in bytes (16 bits)
        channels = 1      # Mono
        framerate = 44100 # Sample rate

        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_bytes)
    

def translate_text(text, targ_lang):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"Translate this text to '{targ_lang}.' TRANSLATE THE TEXT EVEN IF IT IS IN '{targ_lang}'. Text:{text}. ONLY RETURN TRANSLATED TEXT DO NOT WRITE ANYTHING ELSE",
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content 
        # print(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_font_for_language(language):
    font_mapping = {
        'afrikaans': 'Noto Sans',
        'albanian': 'Noto Sans',
        'amharic': 'Noto Sans Ethiopic',
        'arabic': 'Noto Sans Arabic',
        'armenian': 'Noto Sans Armenian',
        'azerbaijani': 'Noto Sans',
        'basque': 'Noto Sans',
        'belarusian': 'Noto Sans',
        'bengali': 'Noto Sans Bengali',
        'bosnian': 'Noto Sans',
        'bulgarian': 'Noto Sans',
        'catalan': 'Noto Sans',
        'cebuano': 'Noto Sans',
        'chichewa': 'Noto Sans',
        'chinese (simplified)': 'Noto Sans CJK SC',
        'chinese (traditional)': 'Noto Sans CJK TC',
        'corsican': 'Noto Sans',
        'croatian': 'Noto Sans',
        'czech': 'Noto Sans',
        'danish': 'Noto Sans',
        'dutch': 'Noto Sans',
        'english': 'Noto Sans',
        'esperanto': 'Noto Sans',
        'estonian': 'Noto Sans',
        'filipino': 'Noto Sans',
        'finnish': 'Noto Sans',
        'french': 'Noto Sans',
        'frisian': 'Noto Sans',
        'galician': 'Noto Sans',
        'georgian': 'Noto Sans',
        'german': 'Noto Sans',
        'greek': 'Noto Sans Greek',
        'gujarati': 'Noto Sans Gujarati',
        'haitian creole': 'Noto Sans',
        'hausa': 'Noto Sans',
        'hawaiian': 'Noto Sans',
        'hebrew': 'Noto Sans Hebrew',
        'hindi': 'Noto Sans Devanagari',
        'hmong': 'Noto Sans',
        'hungarian': 'Noto Sans',
        'icelandic': 'Noto Sans',
        'igbo': 'Noto Sans',
        'indonesian': 'Noto Sans',
        'irish': 'Noto Sans',
        'italian': 'Noto Sans',
        'japanese': 'Noto Sans CJK JP',
        'javanese': 'Noto Sans',
        'kannada': 'Noto Sans Kannada',
        'kazakh': 'Noto Sans',
        'khmer': 'Noto Sans Khmer',
        'korean': 'Noto Sans CJK KR',
        'kurdish (kurmanji)': 'Noto Sans',
        'kyrgyz': 'Noto Sans',
        'lao': 'Noto Sans Lao',
        'latin': 'Noto Sans',
        'latvian': 'Noto Sans',
        'lithuanian': 'Noto Sans',
        'luxembourgish': 'Noto Sans',
        'macedonian': 'Noto Sans',
        'malagasy': 'Noto Sans',
        'malay': 'Noto Sans',
        'malayalam': 'Noto Sans Malayalam',
        'maltese': 'Noto Sans',
        'maori': 'Noto Sans',
        'marathi': 'Noto Sans Marathi',
        'mongolian': 'Noto Sans Mongolian',
        'myanmar (burmese)': 'Noto Sans Myanmar',
        'nepali': 'Noto Sans Devanagari',
        'norwegian': 'Noto Sans',
        'odia': 'Noto Sans Oriya',
        'pashto': 'Noto Sans',
        'persian': 'Noto Sans Persian',
        'polish': 'Noto Sans',
        'portuguese': 'Noto Sans',
        'punjabi': 'Noto Sans Gurmukhi',
        'romanian': 'Noto Sans',
        'russian': 'Noto Sans',
        'samoan': 'Noto Sans',
        'scots gaelic': 'Noto Sans',
        'serbian': 'Noto Sans',
        'sesotho': 'Noto Sans',
        'shona': 'Noto Sans',
        'sindhi': 'Noto Sans',
        'sinhala': 'Noto Sans Sinhala',
        'slovak': 'Noto Sans',
        'slovenian': 'Noto Sans',
        'somali': 'Noto Sans',
        'spanish': 'Noto Sans',
        'sundanese': 'Noto Sans',
        'swahili': 'Noto Sans',
        'swedish': 'Noto Sans',
        'tajik': 'Noto Sans',
        'tamil': 'Noto Sans Tamil',
        'telugu': 'Noto Sans Telugu',
        'thai': 'Noto Sans Thai',
        'turkish': 'Noto Sans',
        'ukrainian': 'Noto Sans',
        'urdu': 'Noto Sans Urdu',
        'uyghur': 'Noto Sans',
        'uzbek': 'Noto Sans',
        'vietnamese': 'Noto Sans',
        'welsh': 'Noto Sans',
        'xhosa': 'Noto Sans',
        'yiddish': 'Noto Sans',
        'yoruba': 'Noto Sans',
        'zulu': 'Noto Sans'
    }
    return font_mapping.get(language.lower(), 'Noto Sans')  # Default to Noto Sans

def audio_to_base64(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        return encoded_audio
    except Exception as e:
        raise Exception(f"Error encoding audio to Base64: {e}")
    
def display_subtitles(audio_path, transcription_segment,translation_segment, auto_play=True):
    if auto_play:
        # Embed the audio player with autoplay enabled
        st.markdown(
            f"""
            <audio controls autoplay>
                <source src="data:audio/wav;base64,{audio_to_base64(audio_path)}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.audio(audio_path, format="audio/wav", start_time=0)

    placeholder = st.empty()

    # time.sleep(2)
    # # Simulate subtitle display
    # for segment in segments:
    #     placeholder.markdown(
    #         f"<h5 style='text-align: center; color: green;'>{segment['text']}</h5>",
    #         unsafe_allow_html=True,
    #     )
    #     time.sleep(segment["end"] - segment["start"])  # Wait for the duration of the segment
    # placeholder.empty()  # Clear the subtitle at the end

    time.sleep(2)
    # Simulate subtitle display
    for i in range(len(translation_segment)):
        placeholder.markdown(
            f"<h5 style='text-align: center; color: green;'>{transcription_segment[i]['text']}</h5><br><h5 style='text-align: center; color: green;'>{translation_segment[i]['text']}</h5>",
            unsafe_allow_html=True,
        )
        time.sleep(transcription_segment[i]["end"] - transcription_segment[i]["start"])  # Wait for the duration of the segment
    placeholder.empty()  # Clear the subtitle at the end

def adjust_segments(segments):
    adjusted_segments = segments
    adjusted_segments[0]["start"] = 0
    for i in range(len(adjusted_segments) - 1):
        adjusted_segments[i+1]["start"]=adjusted_segments[i]["end"]
    
    return adjusted_segments

selected_lang_tar = st.selectbox("Select the target language for translation", ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'])

col1, col2 = st.columns(2)
segments = []
transcription_segment_file="transcription_segments.json"
translation_segment_file="translation_segments.json"

# Variables to store full transcription and translation
full_transcription = ""
full_transcription_file = "full_transcription.txt"
full_translation = ""
full_translation_file="full_translation.txt"

with col1:
    if st.button("Audio 2 Text for Uploaded Audio"):
        # st.write("Processing uploaded file...")
        if uploaded_file is not None:
            # Save the uploaded file to a temporary directory
            with open("temp_audio_file", "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = "temp_audio_file"

            # Load the audio using pydub
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)  # Ensure mono channel
            audio = audio.set_frame_rate(16000)  # Ensure frame rate is 16000 Hz

            # Split the audio into chunks (30 sec per chunk)
            chunk_duration_ms = 30000  
            chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

            segments.clear()

            filename = f"chunk.wav"
            audio.export(filename, format="wav")
            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filename, file.read()),  # Required audio file
                    model="whisper-large-v3",  # Required model for transcription
                    prompt="transcribe",
                    response_format="verbose_json",  # Optional
                    temperature=0.0  # Optional
                )
            transcription_segment=transcription.segments
            with open('full_transcription.txt', 'w') as full_transcription_file:
                for seg in transcription_segment:
                    if seg['text']:
                        full_transcription_file.write(seg['text'] + "\n")
            transcription_segment=adjust_segments(transcription_segment)
            # Save segments to file
            with open(transcription_segment_file, "w") as f:
                json.dump(transcription_segment, f)
            
            translation_segment=copy.deepcopy(transcription_segment)

            # for seg in translation_segment:
            #     # st.write(seg['text'])
            #     seg['text']=translate_text(seg['text'], selected_lang_tar)
            #     full_translation += seg['text'] + "\n"
            # Open 'full_translation.txt' for writing/creating the file
            with open('full_translation.txt', 'w') as full_translation_file:
                for seg in translation_segment:
                    # Translate the text and write to the file
                    seg['text'] = translate_text(seg['text'], selected_lang_tar)
                    if seg['text']:
                        full_translation_file.write(seg['text'] + "\n")
                
            translation_segment=adjust_segments(translation_segment)
            # segments=adjust_segments(translation_segment)
            # Save segments to file
            with open(translation_segment_file, "w") as f:
                json.dump(translation_segment, f)

        else:
            st.error("Please upload an audio file.")

with col2:
    if st.button("Audio 2 Text for Mic recorded Audio"):
        # st.write("Processing recorded audio...")
        if mic_audio is not None:
            audio_file_like.seek(0)
            buffer_data = audio_file_like.read()
            # Save the uploaded file to a temporary directory
            with open("temp_audio_file", "wb") as f:
                f.write(buffer_data)
            audio_path = "temp_audio_file"

            # Load the audio using pydub
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)  # Ensure mono channel
            audio = audio.set_frame_rate(16000)  # Ensure frame rate is 16000 Hz

            # Split the audio into chunks (30 sec per chunk)
            chunk_duration_ms = 30000  
            chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

            

            segments.clear()

            filename = f"chunk.wav"
            audio.export(filename, format="wav")
            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filename, file.read()),  # Required audio file
                    model="whisper-large-v3",  # Required model for transcription
                    prompt="transcribe",
                    response_format="verbose_json",  # Optional
                    temperature=0.0  # Optional
                )
            transcription_segment=transcription.segments
            with open('full_transcription.txt', 'w') as full_transcription_file:
                for seg in transcription_segment:
                    if seg['text']:
                        full_transcription_file.write(seg['text'] + "\n")
            transcription_segment=adjust_segments(transcription_segment)
            # Save segments to file
            with open(transcription_segment_file, "w") as f:
                json.dump(transcription_segment, f)
            
            translation_segment=copy.deepcopy(transcription_segment)

            # for seg in translation_segment:
            #     # st.write(seg['text'])
            #     seg['text']=translate_text(seg['text'], selected_lang_tar)
            #     full_translation += seg['text'] + "\n"
            # Open 'full_translation.txt' for writing/creating the file
            with open('full_translation.txt', 'w') as full_translation_file:
                for seg in translation_segment:
                    # Translate the text and write to the file
                    seg['text'] = translate_text(seg['text'], selected_lang_tar)
                    if seg['text']:
                        full_translation_file.write(seg['text'] + "\n")
                
            translation_segment=adjust_segments(translation_segment)
            # segments=adjust_segments(translation_segment)
            # Save segments to file
            with open(translation_segment_file, "w") as f:
                json.dump(translation_segment, f)

        else:
            st.error("Please upload an audio file.")

if st.button("Play Audio with Subtitles"):
    if os.path.exists(transcription_segment_file) and os.path.exists(translation_segment_file):
        transcription_segment,translation_segment = [],[]
        with open(transcription_segment_file, "r") as f:
            transcription_segment = json.load(f)
        with open(translation_segment_file, "r") as f:
            translation_segment = json.load(f)
        
        display_subtitles("temp_audio_file", transcription_segment,translation_segment)
    else:
        st.error("No segments file found. Please process an audio file first.")

# # Inside your Streamlit layout
# if os.path.exists(segment_file):
#     # Provide the download button if the segments file exists
#     with open(segment_file, "r") as f:
#         segments = json.load(f)
    
#     st.download_button(
#         label="Download Segments JSON",
#         data=json.dumps(segments),  # Convert the segments dictionary to JSON
#         file_name="segments.json",  # The name of the file the user will download
#         mime="application/json"  # MIME type for JSON
#     )
# else:
#     st.error("No segments file found. Please process an audio file first.")

# Add a button to download the full translation as a text file
if os.path.exists('full_transcription.txt'):
    with open('full_transcription.txt', 'r') as file:
        full_transcription_content = file.read()

    st.download_button(
        label="Download Full Transcription (TXT)",
        data=full_transcription_content,  # Use the file content as the download data
        file_name="full_transcription.txt",  # The name of the file the user will download
        mime="text/plain"  # MIME type for plain text files
    )

# Add a button to download the full translation as a text file
if os.path.exists('full_translation.txt'):
    with open('full_translation.txt', 'r') as file:
        full_translation_content = file.read()

    st.download_button(
        label="Download Full Translation (TXT)",
        data=full_translation_content,  # Use the file content as the download data
        file_name="full_translation.txt",  # The name of the file the user will download
        mime="text/plain"  # MIME type for plain text files
    )

# Add a button to download the segments as JSON
if os.path.exists(transcription_segment_file):
    with open(transcription_segment_file, "r") as f:
        segments = json.load(f)
    
    st.download_button(
        label="Download transcription Segments JSON",
        data=json.dumps(segments),  # Convert the segments dictionary to JSON
        file_name="transcription_segments.json",  # The name of the file the user will download
        mime="application/json"  # MIME type for JSON
    )
    st.download_button(
        label="Download translation Segments JSON",
        data=json.dumps(segments),  # Convert the segments dictionary to JSON
        file_name="translation_segments.json",  # The name of the file the user will download
        mime="application/json"  # MIME type for JSON
    )
else:
    st.error("No segments file found. Please process an audio file first.")
