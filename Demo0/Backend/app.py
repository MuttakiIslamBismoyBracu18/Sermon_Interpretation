from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import pandas as pd
import moviepy.editor as mp
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deepface import DeepFace
import threading
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Directories
UPLOAD_FOLDER = 'uploads'
GRAPH_FOLDER = 'graphs'
AUDIO_FOLDER = 'audios'
FRAME_FOLDER = 'frames'
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

progress_status = {"step": "", "progress": 0}
progress_lock = threading.Lock()

# Load Dress Classification Model
dress_model_path = r"E:\PRTS\Dataset\SC\MensDressClassifier\Model_Training\best_model_vgg16.h5"
try:
    dress_model = tf.keras.models.load_model(dress_model_path, compile=False)
    print("Dress classification model loaded successfully!")
except Exception as e:
    print(f"Error loading dress classification model: {e}")

# Load ASR & Emotion Recognition Models
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio_emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
text_emotion_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)

# Emotion Mapping
emotion_labels = ['neutral', 'angry', 'happy', 'sad', 'surprise', 'fear']
emotion_to_num = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

emotion_mapping = {
    "anger": "angry",
    "disgust": "angry",
    "fear": "surprise",  
    "sadness": "sad",
    "joy": "happy",
    "optimism": "happy",
    "love": "happy",
    "surprise": "surprise",
    "neutral": "neutral",
    "trust": "neutral"
}

def map_emotion(emotion):
    return emotion_mapping.get(emotion.lower(), "neutral")

# ---------------------- Eye Detection ---------------------- #
# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_eyes_in_frame(frame):
    """Detect eyes in a frame using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eyes_detected = False
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Region of interest (face)
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(eyes) > 0:
            eyes_detected = True
            break

    return eyes_detected

def generate_eye_graph(eye_data, graph_filename):
    """Generate a graph showing eye presence over time."""
    plt.figure(figsize=(15, 6))
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(eye_data, columns=['Time', 'Eyes_Detected'])
    
    # Plot the data
    plt.plot(df['Time'], df['Eyes_Detected'], 'b-', label='Eyes in Camera')
    plt.plot(df['Time'], 1 - df['Eyes_Detected'], 'r-', label='Eyes Not in Camera')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Eye Presence')
    plt.title('Eye Contact with Camera Over Time')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.legend()
    plt.grid(True)
    
    graph_path = os.path.join(GRAPH_FOLDER, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# ---------------------- Audio Processing ---------------------- #
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    signal, sample_rate = torchaudio.load(audio_path)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        signal = resampler(signal)

    chunk_duration = 1.0
    chunk_samples = int(16000 * chunk_duration)
    num_chunks = len(signal[0]) // chunk_samples

    transcript_list = []
    for i in range(num_chunks):
        chunk = signal[:, i * chunk_samples: (i + 1) * chunk_samples]
        input_values = asr_processor(chunk.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = asr_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = asr_processor.decode(predicted_ids[0])

        transcript_list.append({"Time": i, "Text": text})

    return transcript_list

# ---------------------- Emotion Processing ---------------------- #
def classify_audio_emotions(text):
    if text.strip():
        emotions = audio_emotion_model(text)
        detected_emotion = emotions[0][0]['label']
        return map_emotion(detected_emotion)
    return "neutral"

def classify_text_emotions(text):
    if text.strip():
        emotions = text_emotion_model(text)
        detected_emotion = emotions[0][0]['label']
        return map_emotion(detected_emotion)
    return "neutral"

# ---------------------- Video Processing ---------------------- #
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    video_emotions = []
    frames = []
    total_frames = 0
    eyes_in_camera_frames = 0
    eye_data = []  # To store time and eye detection status
    
    # For 20-second chunk feedback
    chunk_duration = 20  # seconds
    current_chunk = 0
    chunk_emotions = defaultdict(list)
    chunk_eye_data = defaultdict(list)
    chunk_transcripts = defaultdict(list)
    video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        current_time = frame_count / fps
        current_chunk = int(current_time // chunk_duration)

        # Detect eyes in the frame
        eyes_detected = detect_eyes_in_frame(frame)
        if eyes_detected:
            eyes_in_camera_frames += 1

        # Record eye data every second
        if frame_count % int(fps) == 0:
            current_time_rounded = round(frame_count / fps, 2)
            eye_data.append((current_time_rounded, int(eyes_detected)))
            chunk_eye_data[current_chunk].append(int(eyes_detected))
            
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                video_emotions.append({"Time": current_time_rounded, "Video Emotion": dominant_emotion})
                chunk_emotions[current_chunk].append(dominant_emotion)

                frame_filename = f"{FRAME_FOLDER}/{uuid.uuid4()}.jpg"
                cv2.imwrite(frame_filename, frame)
                frames.append(frame_filename)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1

    cap.release()

    # Calculate eye presence percentage
    eyes_in_camera_percent = (eyes_in_camera_frames / total_frames) * 100
    eyes_not_in_camera_percent = 100 - eyes_in_camera_percent

    print(f"Eyes in camera: {eyes_in_camera_percent:.2f}%")
    print(f"Eyes not in camera: {eyes_not_in_camera_percent:.2f}%")

    # Generate eye graph
    eye_graph_filename = "eye_graph.png"
    generate_eye_graph(eye_data, eye_graph_filename)

    return {
        "video_emotions": video_emotions,
        "frames": frames,
        "eyes_in_camera_percent": eyes_in_camera_percent,
        "eyes_not_in_camera_percent": eyes_not_in_camera_percent,
        "eye_graph_filename": eye_graph_filename,
        "chunk_emotions": chunk_emotions,
        "chunk_eye_data": chunk_eye_data,
        "video_duration": video_duration
    }

# ---------------------- Process Chunk Feedback ---------------------- #
def process_chunk_feedback(video_duration, chunk_emotions, chunk_eye_data, transcript_segments):
    chunk_duration = 20  # seconds
    num_chunks = int(video_duration // chunk_duration) + (1 if video_duration % chunk_duration > 0 else 0)
    feedback_chunks = []
    
    for chunk_idx in range(num_chunks):
        start_time = chunk_idx * chunk_duration
        end_time = min((chunk_idx + 1) * chunk_duration, video_duration)
        
        # Get eye contact percentage for this chunk
        eye_data = chunk_eye_data.get(chunk_idx, [])
        eye_percent = (sum(eye_data) / len(eye_data)) * 100 if eye_data else 0
        
        # Get most common video emotion for this chunk
        video_emotions = chunk_emotions.get(chunk_idx, [])
        video_emotion_counts = defaultdict(int)
        for emo in video_emotions:
            video_emotion_counts[emo] += 1
        dominant_video_emotion = max(video_emotion_counts.items(), key=lambda x: x[1])[0] if video_emotion_counts else "neutral"
        
        # Get audio and text emotions for this chunk's transcript
        chunk_transcript = [t for t in transcript_segments if start_time <= t["Time"] < end_time]
        chunk_text = " ".join([t.get("Text", "") for t in chunk_transcript])
        
        audio_emotion = classify_audio_emotions(chunk_text)
        text_emotion = classify_text_emotions(chunk_text)
        
        # Determine emotion match
        emotion_match = False
        emotions = [dominant_video_emotion, audio_emotion, text_emotion]
        if emotions.count(emotions[0]) >= 2 or emotions.count(emotions[1]) >= 2 or emotions.count(emotions[2]) >= 2:
            emotion_match = True
        
        feedback_chunks.append({
            "chunk": chunk_idx + 1,
            "start_time": start_time,
            "end_time": end_time,
            "transcript": chunk_text,
            "eye_contact_percent": round(eye_percent, 2),
            "dominant_video_emotion": dominant_video_emotion,
            "dominant_audio_emotion": audio_emotion,
            "dominant_text_emotion": text_emotion,
            "emotion_match": emotion_match
        })
    
    return feedback_chunks

# ---------------------- Graph & CSV Generation ---------------------- #
def generate_filtered_csvs(df, base_filename):
    matching_df = df[(df['Video Emotion'] == df['Audio Emotion']) & (df['Audio Emotion'] == df['Text Emotion'])]
    mismatching_df = df[(df['Video Emotion'] != df['Audio Emotion']) | (df['Audio Emotion'] != df['Text Emotion'])]

    matching_csv = os.path.join(DATA_FOLDER, f"{base_filename}_matching.csv")
    mismatching_csv = os.path.join(DATA_FOLDER, f"{base_filename}_mismatching.csv")

    matching_df.to_csv(matching_csv, index=False)
    mismatching_df.to_csv(mismatching_csv, index=False)

    return matching_csv, mismatching_csv

def generate_emotion_graph(df, graph_filename, filter_type="all"):
    plt.figure(figsize=(15, 6))

    if df.empty:
        print("Dataframe is empty. No data to plot.")
        return None

    colors = {'Audio Emotion': 'blue', 'Video Emotion': 'green', 'Text Emotion': 'red'}
    markers = {'Audio Emotion': 'o', 'Video Emotion': '^', 'Text Emotion': 's'}

    if filter_type == "all":
        for column in ['Audio Emotion', 'Video Emotion', 'Text Emotion']:
            emotion_nums = df[column].map(lambda x: emotion_to_num.get(x, -1)).dropna()
            if emotion_nums.empty:
                continue
            time_values = df['Time'].loc[emotion_nums.index]
            plt.plot(time_values, emotion_nums, color=colors[column], marker=markers[column], label=column)

    elif filter_type == "highlight":
        mismatching_df = df[(df['Video Emotion'] != df['Audio Emotion']) |
                            (df['Audio Emotion'] != df['Text Emotion']) |
                            (df['Video Emotion'] != df['Text Emotion'])]
        for column in ['Audio Emotion', 'Video Emotion', 'Text Emotion']:
            emotion_nums = mismatching_df[column].map(lambda x: emotion_to_num.get(x, -1)).dropna()
            time_values = mismatching_df['Time'].loc[emotion_nums.index]
            plt.scatter(time_values, emotion_nums, color=colors[column], marker=markers[column], label=column)

    elif filter_type == "fade":
        matching_df = df[(df['Video Emotion'] == df['Audio Emotion']) &
                         (df['Audio Emotion'] == df['Text Emotion'])]
        for column in ['Audio Emotion', 'Video Emotion', 'Text Emotion']:
            emotion_nums = matching_df[column].map(lambda x: emotion_to_num.get(x, -1)).dropna()
            time_values = matching_df['Time'].loc[emotion_nums.index]
            plt.scatter(time_values, emotion_nums, color=colors[column], marker=markers[column], label=column)

    elif filter_type == "sudden":
        for column in ['Audio Emotion', 'Video Emotion', 'Text Emotion']:
            non_neutral_df = df[df[column] != 'neutral']
            if not non_neutral_df.empty:
                emotion_nums = non_neutral_df[column].map(lambda x: emotion_to_num.get(x, -1))
                time_values = non_neutral_df['Time']
                plt.plot(
                    time_values,
                    emotion_nums,
                    color=colors[column],
                    marker=markers[column],
                    label=f"{column} (Non-Neutral)",
                )
    plt.yticks(range(len(emotion_labels)), emotion_labels)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion")
    plt.title(f"Emotion Trends Over Time ({filter_type.capitalize()} Graph)")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(GRAPH_FOLDER, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# ---------------------- Find Matching Emotions ---------------------- #
def find_matching_emotions(df):
    matching_times = df[(df['Video Emotion'] == df['Audio Emotion']) & (df['Audio Emotion'] == df['Text Emotion'])]
    return matching_times[['Time', 'Video Emotion']]

# ---------------------- Dress Code ---------------------- #
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to match model input size
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def classify_dress(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    dress_predictions = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every second (skip frames to reduce computation)
        if frame_count % fps == 0:
            processed_frame = preprocess_frame(frame)
            prediction = dress_model.predict(processed_frame)
            predicted_class = np.argmax(prediction, axis=1)[0]
            dress_predictions.append(predicted_class)

        frame_count += 1

    cap.release()

    # Map predictions to class names
    class_names = {0: "Formal", 1: "Formal", 2: "Informal"}
    dress_predictions = [class_names[pred] for pred in dress_predictions]

    # Return the most frequent dress code
    from collections import Counter
    most_common_dress = Counter(dress_predictions).most_common(1)[0][0]
    return most_common_dress

# ---------------------- Routes ---------------------- #
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        file = request.files['file']
        update_progress("Uploading video file...", 10)

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        dress_code = classify_dress(video_path)
        print(f"🔹 Detected Dress Code: {dress_code}")

        update_progress("Processing video frames for emotions...", 20)
        video_results = process_video(video_path)
        video_emotions = video_results["video_emotions"]
        frames = video_results["frames"]
        eyes_in_camera_percent = video_results["eyes_in_camera_percent"]
        eyes_not_in_camera_percent = video_results["eyes_not_in_camera_percent"]
        eye_graph_filename = video_results["eye_graph_filename"]
        chunk_emotions = video_results["chunk_emotions"]
        chunk_eye_data = video_results["chunk_eye_data"]
        video_duration = video_results["video_duration"]

        update_progress("Extracting audio from video...", 40)
        audio_path = extract_audio(video_path)

        update_progress("Transcribing audio to text...", 60)
        transcript_segments = transcribe_audio(audio_path)

        if not transcript_segments:
            update_progress("Audio transcription failed.", 100)
            return jsonify({"error": "Audio transcription failed. No text was generated."}), 500

        update_progress("Classifying emotions for audio and text...", 80)
        audio_emotions = [{"Time": t["Time"], "Audio Emotion": classify_audio_emotions(t.get("Text", ""))} for t in transcript_segments]
        text_emotions = [{"Time": t["Time"], "Text Emotion": classify_text_emotions(t.get("Text", ""))} for t in transcript_segments]

        update_progress("Processing chunk feedback...", 85)
        chunk_feedback = process_chunk_feedback(video_duration, chunk_emotions, chunk_eye_data, transcript_segments)

        update_progress("Merging data and generating CSV...", 90)
        df = pd.DataFrame(transcript_segments).merge(pd.DataFrame(video_emotions), on="Time", how="outer") \
                                              .merge(pd.DataFrame(audio_emotions), on="Time", how="outer") \
                                              .merge(pd.DataFrame(text_emotions), on="Time", how="outer") \
                                              .fillna(method="ffill")

        base_filename = str(uuid.uuid4())
        csv_filename = f"{base_filename}_emotions.csv"
        csv_path = os.path.join(DATA_FOLDER, csv_filename)

        df = df[['Time', 'Audio Emotion', 'Video Emotion', 'Text Emotion']]
        df.to_csv(csv_path, index=False)

        update_progress("Generating emotion graphs...", 95)
        generate_filtered_csvs(df, base_filename)
        graph_filenames = {}
        for filter_type in ['all', 'highlight', 'fade', 'sudden']:
            graph_filename = f"{filter_type}_graph.png"
            graph_path = generate_emotion_graph(df, graph_filename, filter_type)
            graph_filenames[filter_type] = graph_filename
        
        # Add eye graph to the returned graphs
        graph_filenames['eye_graph'] = eye_graph_filename

        transcript_text = "\n".join([f"{seg['Time']}s: {seg.get('Text', 'N/A')}" for seg in transcript_segments if seg.get("Text")])

        update_progress("Processing complete!", 100)
        return jsonify({
            "dress_code": dress_code,
            "frames": frames,
            "graphs": graph_filenames,
            "csv": csv_filename,
            "transcript": transcript_text,
            "video_emotions": df.to_dict(orient="records"),
            "eyes_in_camera_percent": eyes_in_camera_percent,
            "eyes_not_in_camera_percent": eyes_not_in_camera_percent,
            "chunk_feedback": chunk_feedback,
            "video_duration": video_duration
        })
    except Exception as e:
        print(f"Error: {e}")
        update_progress("An error occurred.", 100)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/matching_emotions', methods=['GET'])
def get_matching_emotions():
    csv_files = sorted(os.listdir(DATA_FOLDER), reverse=True)
    if not csv_files:
        return jsonify({"error": "No emotion data available."}), 404
    
    latest_csv = os.path.join(DATA_FOLDER, csv_files[0])
    df = pd.read_csv(latest_csv)
    matching_emotions = find_matching_emotions(df)
    return jsonify(matching_emotions.to_dict(orient='records'))

@app.route('/graphs/<path:filename>')
def serve_graph(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

@app.route('/download_csv/<filename>')
def download_csv(filename):
    return send_from_directory(DATA_FOLDER, filename, as_attachment=True)

@app.route('/generate_graphs', methods=['GET'])
def generate_all_graphs():
    csv_files = sorted(os.listdir(DATA_FOLDER), reverse=True)
    if not csv_files:
        return jsonify({"error": "No emotion data available."}), 404

    latest_csv = os.path.join(DATA_FOLDER, csv_files[0])
    df = pd.read_csv(latest_csv)

    graph_filenames = {}
    for filter_type in ['all', 'highlight', 'fade', 'sudden']:
        graph_filename = f"{filter_type}_graph.png"
        graph_path = generate_emotion_graph(df, graph_filename, filter_type)
        graph_filenames[filter_type] = graph_filename 

    print("Generated graphs:", graph_filenames) 
    return jsonify(graph_filenames)

# ---------------------- Progress ---------------------- #
def update_progress(step, progress):
    with progress_lock:
        progress_status["step"] = step
        progress_status["progress"] = progress

@app.route('/progress', methods=['GET'])
def get_progress():
    with progress_lock:
        return jsonify(progress_status)

if __name__ == '__main__':
    app.run(debug=True)