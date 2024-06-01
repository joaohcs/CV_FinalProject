## Primeira Etapa - Rúbrica C
## Conseguir montar uma versão local que roda por linha de comando para gerar relatório a partir de input de vídeo

# - Professor consegue rodar uma versão local lendo o README.
# - Detecção de rostos por frame.
# - Identificação de diferentes pessoas. Não precisa reconhecer, apenas distinguir que são diferentes.
# - Identificação de emoções.
# - Identificação de estados (baseados em driver detection).
# - Linha de comando que gera um relatório em txt. - VOU TENTAR PDF P/ COLOCAR GRÁFICOS

# Notes
# Lembre de instalar cmake para poder instalar face-recognition
# Instale também as dependências das quais o face_recognition precisa
# Então instale dlib e por fim face_recognition

# -------------------------------------------------------------------------------

# Bibliotecas
import face_recognition # Para detecção e reconhecimento de rostos - detect_and_encode_faces()
import cv2 # Processamento do vídeo - criação de lista de frames 

import numpy as np

import pandas as pd # Criar e manipular o df que será usado para guardar as emoções

from deepface import DeepFace # Para análise de sentimento

import matplotlib.pyplot as plt # Para Dicionário de Pessoas e gráficos de insights

import subprocess # Manipulação PDF relatório
from matplotlib.backends.backend_pdf import PdfPages # Para gerar o PDF do relatório

## Dicionário global para guardar rostos das pessoas por ID
people_faces = {}


## Processando vídeo (transformando vídeo em frames)
def extract_frames(video_path, frame_rate):
    # Abre arquivo do vídeo
    vidcap = cv2.VideoCapture(video_path)
    frames = [] # Lista de Frames
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        # Selecionar frames (com base em frame_rate)
        if count % frame_rate == 0 and success:
            frames.append(image)
        count += 1

    # Release recursos
    vidcap.release()
    return frames



## Detectando rostos
    # Recebe a imagem e devolve os encodings e localizações dos rostos
    # Devolve tbm a própria imagem para fins de debug e observabilidade

def detect_and_encode_faces(frame):
    
    image = frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings, face_locations, image


## Função para Processar Frames
    # Para cada frame:
        # Gera encodings e puxa localizações de pessoas na img
        # Compara encodings com lista de known_people
            # Se não reconhece, add a Known_people
            # Se conhece, pega o index (id) da pessoa
        # Processa emoções de cada pessoa
        # Add emoções ao dataframe junto com id da pessoa

def process_frames(frames):
    global people_faces # Acessando dicionário que guarda fotos de pessoas para consulta
    known_people = pd.DataFrame(columns=['person_id'])
    known_encodings = [] # Guardar encodings para comparação. Guardar em dfs causa erro durante retrieval
    emotion_records = []
    people = 0
    frame_count = 0

    for frame_path in frames:
        face_encodings, face_locations, image = detect_and_encode_faces(frame_path)
        frame_emotions = []
        frame_count += 1

        for idx, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
            if not known_encodings:
                print('first person added\n')
                person_id = 0
                known_people = pd.concat([known_people, pd.DataFrame({'person_id': [person_id]})], ignore_index=True)
                known_encodings.append(encoding)
                people += 1
            else:
                print("segunda pessoa")
                list_known_faces = np.atleast_2d(np.array(known_encodings))
                print(list_known_faces)
                print('--listknown^, encodingatualabaixo\n')
                print(encoding)
                matches = face_recognition.compare_faces(
                                            known_face_encodings=list_known_faces, 
                                            face_encoding_to_check=encoding,
                                            tolerance=0.6) # A documentação sugeriu 0.6, é também o default
                print('---matches\n')
                print(matches)
                if True in matches:
                    first_match_index = matches.index(True)
                    person_id = known_people.iloc[first_match_index]['person_id']
                else:
                    person_id = known_people['person_id'].max() + 1
                    known_people = pd.concat([known_people, pd.DataFrame({'person_id': [person_id]})], ignore_index=True)
                    known_encodings.append(encoding)
                    people += 1


            # Localização da pessoa na imagem
            top, right, bottom, left = location
            face_image = image[top:bottom, left:right]

            # Para cada pessoa, será guardada uma foto para identificação futura
            if person_id not in people_faces:
                people_faces[person_id] = face_image
            
            # Analisando as emoções com DeepFace
            try:
                emotion_result  = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                print(emotion_result)
                emotions = emotion_result[0]['emotion']
                emotions = {emotion: round(value, 2) for emotion, value in emotions.items()} # Para ficar em %
                dom_emotion = emotion_result[0]['dominant_emotion']
                print('---emotions\n')
                print(f"emotions: {emotions}, dom_emot: {dom_emotion}")
            except Exception as e:
                print(f"Deepface error: {e}, idx: {idx}")
                emotions = {}
                dom_emotion = 'unknown'

            frame_emotions.append((person_id, emotions))
            print(f"frame-emotions: {frame_emotions}")

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
    # pd.DataFrame(emotion_records) é um df que contém uma linha para cada pessoa em cada frame e seus respectivos sentimentos
    # known_encodings é uma lista de embeddings (encodings) de pessoas já identificadas
    # people é o número de pessoas identificadas no total


## Função para gerar "Dicionário de pessoas" - imagens
    # Com ele o usuário poderá saber a quem corresponde cada person_ID, 
    # e consequentemente os sentimentos daquele person_ID
def display_people_faces():
    global people_faces
    num_faces = len(people_faces)

    if num_faces == 1: # Para o caso de só ter uma pessoa
        fig, ax = plt.subplots(figsize=(5, 5))
        person_id, face_img = next(iter(people_faces.items())) # Acessando o primeiro key-value pair
        ax.imshow(face_img)
        ax.axis('off')
        ax.set_title(f'Person ID: {person_id}')
        return [fig] # Retorna uma lista com a figura
    else: # Para o caso de múltiplas pessoas na reunião
        cols = 3 # Colunas de subplots
        rows = (num_faces + cols -1) // cols # Linhas necessárias no subplot
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()
        for idx, (person_id, face_img) in enumerate(people_faces.items()):
            axes[idx].imshow(face_img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Person ID: {person_id}')
        for idx in range(len(people_faces), len(axes)): # Para tirar subplots desnecessários
            axes[idx].axis('off')
        return [fig] # Retorna uma lista com a figura



## Função para gerar insights a partir do DataFrame de análise de sentimentos 
def generate_insights(dataframe):

    # Sorting o Dataframe por frame e por peerson_id para o plotting ficar correto
    dataframe.sort_values(by=['person_id', 'frame_count'], inplace=True)

    # IDs únicos
    person_ids = dataframe['person_id'].unique()
    figures = [] # Lista que vai guardar plots (objetos matplotlib) para cada pessoa.

    # Lista de Emoções
    emotion_list = ['happy', 'sad', 'angry', 'neutral', 'fear', 'disgust', 'surprise']

    for person_id in person_ids:
        fig, axs = plt.subplots(nrows=len(emotion_list), ncols=1, figsize=(10, 15))
        fig.suptitle(f'Evolução das Emoções da Pessoa de ID {person_id}', fontsize=16)

        # Filtrando dados para a pessoa
        person_data = dataframe[dataframe['person_id'] == person_id]

        # Plotando cada emoção em um subplot
        for idx, emotion in enumerate(emotion_list):
            if emotion in person_data.columns:
                ax = axs[idx] if len(emotion_list) > 1 else axs
                ax.plot(person_data['frame_count'], person_data[emotion], label=emotion, marker='o', linestyle='-')
                ax.set_title(f'Nível de Emoção: {emotion}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Percentual de Emoção')
                ax.legend()
                ax.set_ylim(0, 100)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        figures.append(fig)

    return figures

## Função para gerar PDF de relatório
def create_pdf_with_figures(face_plots, emotion_plots, pdf_path):
    with PdfPages(pdf_path) as pdf:
        # Adicionando Dicionário de Pessoas ao Relatório
        for fig in face_plots:
            pdf.savefig(fig)
            plt.close(fig)
        
        # Adicionando os plots de emoções para cada pessoa
        for fig in emotion_plots:
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path # Devolve o caminho do PDF, será usado apra abrir o PDF

## Função para abrir o pdf - Direto no VSCode
def open_pdf_in_vscode(pdf_path):
    try:
        # Abrindo diretamente no VSCode
        subprocess.run(['code','-r', pdf_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Falha em abrir PDF no VSCode: {e}")
    except FileNotFoundError:
        print("VSCode command-line 'code' não instalada ou encontrada")



## Função para poder executar o código via command-line

print('-- TESTE FINAL --\n')
print('-- Criação de lista de Frames\n')
video_path = 'video_teste.mp4'
frame_rate = 30
frames = extract_frames(video_path=video_path, frame_rate=frame_rate)
print('-- Definindo Path para PDF\n')
pdf_path = 'results.pdf'
print('-- Processando Frames\n')
emotion_df, known_people_updated, num_people = process_frames(frames)
print(f'emotion df:\n {emotion_df}')
print(f'known_encodings:\n {known_people_updated}')
print(f'nº de pessoas:\n {num_people}')
print('-- Gerando Dicionário de Pessoas\n')
face_plots = display_people_faces()
print('-- Gerando Insights\n')
emotion_plots = generate_insights(emotion_df)
print('-- Criando relatório\n')
pdf_caminho = create_pdf_with_figures(face_plots=face_plots, emotion_plots=emotion_plots, pdf_path=pdf_path)
print('-- Abrir PDF no VSCode\n')
open_pdf_in_vscode(pdf_caminho)
