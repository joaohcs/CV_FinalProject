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
import cv2
import os
import numpy as np

import pandas as pd # Criar e manipular o df que será usado para guardar as emoções

from deepface import DeepFace # Para análise de sentimento

import matplotlib.pyplot as plt # Para Dicionário de Pessoas


## Dicionário global para guardar rostos das pessoas por ID
people_faces = {}

## Processando vídeo (transformando vídeo em frames)




## Detectando rostos
    # Recebe a imagem e devolve os encodings e localizações dos rostos
    # Devolve tbm a própria imagem para fins de debug e observabilidade

def detect_and_encode_faces(image_path):
    
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings, face_locations, image

output = detect_and_encode_faces("imgs/people.jpeg")
print(output)
print('---\n')

## Processa Frames
    # Para cada frame:
        # Gera encodings e puxa localizações de pessoas na img
        # Compara encodings com lista de known_people
            # Se não reconhece, add a Known_people
            # Se conhece, pega o index (id) da pessoa
        # Processa emoções de cada pessoa
        # Add emoções ao dataframe junto com id da pessoa

def process_frames(frames):
    global people_faces # Acessando dicionário que guarda fotos de pessoas para consulta
    known_people = pd.DataFrame(columns=['person_id', 'encodings'])
    emotion_records = []
    people = 0

    for frame_path in frames:
        face_encodings, face_locations, image = detect_and_encode_faces(frame_path)
        frame_emotions = []

        for idx, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
            if known_people.empty:
                print('first person added\n')
                person_id = 0
                new_row = pd.DataFrame({'person_id': person_id, 'encodings': encoding})
                known_people = pd.concat([known_people, new_row], ignore_index=True)
                people +=1
            else:
                print("segunda pessoa")
                list_known_faces = np.atleast_2d(np.array(known_people['encodings'].tolist()))
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
                    new_row = pd.DataFrame({'person_id': person_id, 'encodings': encoding})
                    known_people = pd.concat([known_people, new_row], ignore_index=True)
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
                'frame': frame_path,
                'person_id': person_id,
                'dom_emotion': dom_emotion,
            }
            emotion_record.update(emotions)
            emotion_records.append(emotion_record)

    return pd.DataFrame(emotion_records), known_people, people
    # pd.DataFrame(emotion_records) é um df que contém uma linha para cada pessoa em cada frame e seus respectivos sentimentos
    # known_people é uma lista de embeddings (encodings) de pessoas já identificadas
    # people é o número de pessoas identificadas no total

frames = ['imgs/people.jpeg'] # Lista de frames (Pegar da transformação de vídeo)
emotion_df, known_people_updated, num_people = process_frames(frames)

print('emotion_df:')       
print(emotion_df)
print("---\n")
print(known_people_updated)



## Função para gerar "Dicionário de pessoas" - imagens
    # Com ele o usuário poderá saber a quem corresponde cada person_ID, 
    # e consequentemente os sentimentos daquele person_ID
def display_people_faces():
    global people_faces
    fig, axes = plt.subplots(nrows=1, ncols=len(people_faces), figsize=(15, 5))
    for idx, (person_id, face_img) in enumerate(people_faces.items()):
        if len(people_faces) > 1:
            ax = axes[idx]
        else:
            ax = axes
        ax.imshow(face_img)
        ax.axis('off')
        ax.set_title(f'Person ID: {person_id}')

    plt.show()


print('---faces\n')
display_people_faces()


## Função para gerar PDF de relatório


## Função para poder executar o código via command-line

