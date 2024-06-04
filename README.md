# CV_FinalProject
Repo for Final Project of Computer Vision course at Insper


# Rúbrica C

Para a Rúbrica C, foi criado um script que permite gerar, a partir de um vídeo de uma reunião on-line, um relatório acerca das emoções dos participantes.

O script é executável através da interface de linha de comando (CLI - Command Line Interface)
Comando: "python3 v1_cl_local.py path/para/video.mp4"

São 5 funções principais, explicadas abaixo:
* Transformação do vídeo: Recebe vídeo como input e devolve lista de frames para serem processados
* Detecção e Encoding de Faces: Recebe um frame e devolve (i) uma lista com os embeddings (encodings) de cada rosto, (ii) uma lista com as coordenadas dos rostos e (iii) o próprio frame. A função utiliza a biblioteca `face_recognition` para criação dos encodings e extração das coordenadas (locations) dos rostos.
* Processamento de Frames: Recebe a lista de frames e usando a função anterior, para cada frame, itera pelos encodings e coordenadas dos rostos. Esta função avalia se o rosto já é cconhecido, computa sua emoção no frame em questão, e adiciona as informações de (i) frame, (ii) person_ID, (iii) % de cada emoção, (iv) emoção dominante em um DataFrame para cada pessoa no Frame. A biblioteca `DeepFace` é usada para análise de sentimento, e a `face_recognition` para comparação de econdings (para identificação de pessoas já observadas em frames passados). A função retorna o DataFrame, uma lista dos encodings de rostos conhecidos, e o número de pessoas reconhecidas no vídeo.
* Display de Rostos: Como o script identifica pessoas apenas pelo ID, mostrar o rosto da pessoa correspondente a cada ID é necessário para identificação, permitindo assim que o usuário saiba quem é o ID=2 que teve aquela configuração de sentimentos. A função recebe um dicionario com pares de 'person_id'-'imagens' e plota os rostos com o person_ID como título.
* Geração de Insights: Função cria diversos subplots mostrando a evolução de cada tipo de emoção para cada pessoa no vídeo
* Criação de PDF: Dispondo do DataFrame com as emoções de cada pessoa por frame e o dicionário com os rostos das pessoas por ID, é criado um PDF com insights acerca das emoções do participante ao longo da reunião. A função retorna o PDF do relatório.
* Abrir PDF: A função abre o PDF dentro do próprio VSCode.
* Main: Chama todas as funções montando a pipeline da aplicação


### Como usar - Dicas
* Clonar Repositório
* Instalar dependências que estão em requirements.txt
* Rodar comando 'python3 v1_cl_local.py path/para/video.mp4'

Obs: O VSCode cria um virtual environment ao tentar rodar o script. 

# Rúbrica B - Flask App Local
A Rúbrica B consistia em criar um servidor local usando flask e um HTML cru apenas para prova de conceito, no qual o usuário conseguisse subir um vídeo e receber de volta o report. 

Nesta etapa, melhorei a formatação do report e ao invés do HTML cru adicionei algumas modificações para ficar melhorar a UI.

Dentro da pasta da v2, em Uploads, é possível encontrar um exemplo de report gerado

### Como usar - Dicas
* Clonar Repositório
* Instalar as dependências em requirements.txt
* Rodar 'flask run' no terminal

# Rúbrica A - Deploy Público Flask App
O objetivo da Rúbrica A era realizar o deploy público da aplicação. Não consegui. 

Testei o Heroku e o Railway, além de pesquisar algumas alternativas usando Docker. O problema principal foi que a face_recognition, biblioteca usada na aplicação, precisava que fosse instalada antes a biblioteca dlib que por sua vez precisava do cmake e de outras dependências que precisaram ser instaladas a nível de sistema (Com sudo apt install..). 

Para isso, seria necessário acessar uma CLI dentro da plataforma ou ferramenta para deploy e instalar no próprio sistema essas dependências, foi o que descobri tentando resolver o problema. Vou testar depois como fazer isso, por não ter conseguido busquei melhorar o HTML da Rúbrica B, o que também era um objetivo da Rúbrica A.