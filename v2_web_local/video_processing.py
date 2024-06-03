import face_recognition
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

people_faces = {}

def extract_frames(video_path, frame_rate):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        if count % frame_rate == 0 and success:
            frames.append(image)
        count += 1

    vidcap.release()
    return frames

def detect_and_encode_faces(frame):
    image = frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings, face_locations, image

def process_frames(frames):
    global people_faces
    known_people = pd.DataFrame(columns=['person_id'])
    known_encodings = []
    emotion_records = []
    people = 0
    frame_count = 0

    for frame_path in frames:
        face_encodings, face_locations, image = detect_and_encode_faces(frame_path)
        frame_emotions = []
        frame_count += 1

        for idx, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
            if not known_encodings:
                person_id = 0
                known_people = pd.concat([known_people, pd.DataFrame({'person_id': [person_id]})], ignore_index=True)
                known_encodings.append(encoding)
                people += 1
            else:
                list_known_faces = np.atleast_2d(np.array(known_encodings))
                matches = face_recognition.compare_faces(
                                            known_face_encodings=list_known_faces, 
                                            face_encoding_to_check=encoding,
                                            tolerance=0.6)
                if True in matches:
                    first_match_index = matches.index(True)
                    person_id = known_people.iloc[first_match_index]['person_id']
                else:
                    person_id = known_people['person_id'].max() + 1
                    known_people = pd.concat([known_people, pd.DataFrame({'person_id': [person_id]})], ignore_index=True)
                    known_encodings.append(encoding)
                    people += 1

            top, right, bottom, left = location
            face_image = image[top:bottom, left:right]

            if person_id not in people_faces:
                people_faces[person_id] = face_image
            
            try:
                emotion_result  = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                emotions = emotion_result[0]['emotion']
                emotions = {emotion: round(value, 2) for emotion, value in emotions.items()}
                dom_emotion = emotion_result[0]['dominant_emotion']
            except Exception as e:
                emotions = {}
                dom_emotion = 'unknown'

            frame_emotions.append((person_id, emotions))

        for person_id, emotions in frame_emotions:
            emotion_record = {
                'frame_count': frame_count,
                'frame_array': frame_path,
                'person_id': person_id,
                'dom_emotion': dom_emotion,
            }
            emotion_record.update(emotions)
            emotion_records.append(emotion_record)

    return pd.DataFrame(emotion_records), known_encodings, people

# def display_people_faces():
#     global people_faces
#     num_faces = len(people_faces)

#     if num_faces == 1:
#         fig, ax = plt.subplots(figsize=(5, 5))
#         person_id, face_img = next(iter(people_faces.items()))
#         ax.imshow(face_img)
#         ax.axis('off')
#         ax.set_title(f'Person ID: {person_id}')
#         return [fig]
#     else:
#         cols = 3
#         rows = (num_faces + cols - 1) // cols
#         fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
#         axes = axes.flatten()
#         for idx, (person_id, face_img) in enumerate(people_faces.items()):
#             axes[idx].imshow(face_img)
#             axes[idx].axis('off')
#             axes[idx].set_title(f'Person ID: {person_id}')
#         for idx in range(len(people_faces), len(axes)):
#             axes[idx].axis('off')
#         return [fig]

def display_people_faces():
    global people_faces
    num_faces = len(people_faces)

    if num_faces == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        person_id, face_img = next(iter(people_faces.items()))
        ax.imshow(face_img)
        ax.axis('off')
        ax.set_title(f'Person ID: {person_id}')
        return [fig]
    else:
        cols = 3
        rows = (num_faces + cols - 1) // cols
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
        fig.suptitle('Diconário de Pessoas', fontsize=24)
        axes = axes.flatten()
        for idx, (person_id, face_img) in enumerate(people_faces.items()):
            axes[idx].imshow(face_img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Person ID: {person_id}')
        for idx in range(len(people_faces), len(axes)):
            axes[idx].axis('off')
        return [fig]

def generate_insights(dataframe):
    dataframe.sort_values(by=['person_id', 'frame_count'], inplace=True)
    person_ids = dataframe['person_id'].unique()
    figures = []
    emotion_list = ['happy', 'sad', 'angry', 'neutral', 'fear', 'disgust', 'surprise']

    for person_id in person_ids:
        fig, axs = plt.subplots(nrows=len(emotion_list), ncols=1, figsize=(10, 15))
        fig.suptitle(f'Evolução das Emoções da Pessoa de ID {person_id}', fontsize=16)
        person_data = dataframe[dataframe['person_id'] == person_id]

        for idx, emotion in enumerate(emotion_list):
            if emotion in person_data.columns:
                ax = axs[idx] if len(emotion_list) > 1 else axs
                ax.plot(person_data['frame_count'], person_data[emotion], label=emotion, marker='o', linestyle='-')
                ax.set_title(f'Nível de Emoção: {emotion}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('%')
                ax.legend()
                ax.set_ylim(0, 100)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        figures.append(fig)

    return figures

def create_pdf_with_figures(face_plots, emotion_plots, pdf_path):
    with PdfPages(pdf_path) as pdf:
        # Tamanho padrão das páginas
        page_size = (8.5, 11)

        # Primeira página com título
        fig, ax = plt.subplots(figsize=page_size)
        ax.text(0.5, 0.9, 'Análise de Sentimentos', fontsize=24, ha='center', va='top')
        ax.axis('off') 
        pdf.savefig(fig)
        plt.close(fig)

        # face plots
        for fig in face_plots:
            fig.set_size_inches(page_size)
            pdf.savefig(fig)
            plt.close(fig)
        
        # emotion plots
        for fig in emotion_plots:
            fig.set_size_inches(page_size)
            pdf.savefig(fig)
            plt.close(fig)
    
    return pdf_path

# def create_pdf_with_figures(face_plots, emotion_plots, pdf_path):
#     with PdfPages(pdf_path) as pdf:
#         for fig in face_plots:
#             pdf.savefig(fig)
#             plt.close(fig)
        
#         for fig in emotion_plots:
#             pdf.savefig(fig)
#             plt.close(fig)
#     return pdf_path

# Função main da v1, será usada no app.py
def process_video(video_path):
    frame_rate = 30
    frames = extract_frames(video_path=video_path, frame_rate=frame_rate)
    emotion_df, known_people_updated, num_people = process_frames(frames)
    face_plots = display_people_faces()
    emotion_plots = generate_insights(emotion_df)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join('uploads', f'results_{timestamp}.pdf')
    create_pdf_with_figures(face_plots=face_plots, emotion_plots=emotion_plots, pdf_path=pdf_path)
    return pdf_path
