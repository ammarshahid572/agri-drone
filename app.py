#Import necessary libraries
import os
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from werkzeug.utils import secure_filename
from agriClassifier import  agriClass
#Initialize the Flask app

app = Flask(__name__)
UPLOAD_FOLDER='static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'png', 'gif'}

camera = cv2.VideoCapture(0)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/form')
def form():
    form='''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post action="/upload" enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''
    return form
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    filename=""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print("No file part")
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            print("no file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dst=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dst)
            #classifier(dst)
            pclass, confi, params=agriClass(dst)
            print(pclass)
            return render_template('imageClassify.html',
                               image=filename,
                               pre_class=pclass,
                               confidence="{:.2f}".format(confi),
                               N= params["N"],
                               P= params["P"],
                               K=params["K"],
                               ph= params["PH"],
                               C= params["C"]
                               )
    return 'Error'

@app.route('/classify')
def classify():
    success, frame = camera.read()  # read the camera frame
    if not success:
        return ("Error Openening camera")
    else:
        cv2.imwrite('static/image.jpg',frame)
        pclass, confi, params=agriClass('static/image.jpg')
        return render_template('imageClassify.html',
                               image="image.jpg",
                               pre_class=pclass,
                               confidence="{:.2f}".format(confi),
                               N= params["N"],
                               P= params["P"],
                               K=params["K"],
                               ph= params["PH"],
                               C= params["C"]
                               )

    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    
