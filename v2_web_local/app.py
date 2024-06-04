## Segunda Etapa - Rúbrica B
## Criar uma versão web de baixa fidelidade, ainda para rodar localmente

# - Interface web de baixa fidelidade (prova de conceito crua).
# - Exibição de relatório formatado.
# - Melhorias na apresentação do relatório

from flask import (Flask, render_template, 
                   request, redirect, url_for, send_from_directory,
                   )
import os
from video_processing import process_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config["ALLOWED_EXTENSIONS"] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url) # Redireciona para a mesma página do request
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(filepath)
        pdf_path = process_video(filepath)
        return redirect(url_for('download_page', filename=os.path.basename(pdf_path)))

    return redirect(request.url)

@app.route('/download/<filename>')
def download_page(filename):
    return render_template('download.html', filename=filename)


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['uploads']):
        os.makedirs(app.config['uploads'])
    app.run(debug=True)




