# CV_FinalProject
Repo for Final Project of Computer Vision course at Insper


# Rúbrica C

Para a Rúbrica C, foi criado um script que permite gerar, a partir de um vídeo de uma reunião on-line, um relatório acerca das emoções dos participantes.

O script é executável através da interface de linha de comando (CLI - Command Line Interface)
Comando: ""

São 5 funções principais, explicadas abaixo:
* Transformação do vídeo: Recebe vídeo como input e devolve lista de frames para serem processados
* Detecção e Encoding de Faces: Recebe um frame e devolve (i) uma lista com os embeddings (encodings) de cada rosto, (ii) uma lista com as coordenadas dos rostos e (iii) o próprio frame. A função utiliza a biblioteca `face_recognition` para criação dos encodings e extração das coordenadas (locations) dos rostos.
* Processamento de Frames: Recebe a lista de frames e usando a função anterior, para cada frame, itera pelos encodings e coordenadas dos rostos. Esta função avalia se o rosto já é cconhecido, computa sua emoção no frame em questão, e adiciona as informações de (i) frame, (ii) person_ID, (iii) % de cada emoção, (iv) emoção dominante em um DataFrame para cada pessoa no Frame. A biblioteca `DeepFace` é usada para análise de sentimento, e a `face_recognition` para comparação de econdings (para identificação de pessoas já observadas em frames passados). A função retorna o DataFrame, uma lista dos encodings de rostos conhecidos, e o número de pessoas reconhecidas no vídeo.
* Display de Rostos: Como o script identifica pessoas apenas pelo ID, mostrar o rosto da pessoa correspondente a cada ID é necessário para identificação, permitindo assim que o usuário saiba quem é o ID=2 que teve aquela configuração de sentimentos. A função recebe um dicionario com pares de 'person_id'-'imagens' e plota os rostos com o person_ID como título.
* Criação de PDF: Dispondo do DataFrame com as emoções de cada pessoa por frame e o dicionário com os rostos das pessoas por ID, é criado um PDF com insights acerca das emoções do participante ao longo da reunião. A função retorna o PDF do relatório.

### Como usar - Dicas
* Clonar Repositório
* Instalar dependências
* Rodar comando

# Rúbrica B - Flask App Local


# Rúbrica A - Deploy Público Flask App