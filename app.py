from flask import Flask, request, jsonify
from os import path, system, listdir
from multiprocessing import Process
from transformers import pipeline
from cv2 import FONT_HERSHEY_SIMPLEX, imread, imwrite, putText
from flask import Flask, render_template, request
from json import loads

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
app = Flask(__name__)

position = (50, 50)  
font = FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 255)  
thickness = 2  
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', '.com'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def caption_image(image_path):
    res = image_to_text(image_path)
    ans = res[0]['generated_text']
    image = imread(image_path)
    image_with_text = putText(image, ans, position, font, font_scale, color, thickness)
    image_path = str(image_path).split('/')[1]
    imwrite(f"static/captions/{image_path}", image_with_text)
    return ans

def run_command(command): system(command)
def caption(image_paths):
    
    result = {}
    for image_path in str(image_paths).split(','):
        image_path = image_path.strip()
        result.update({image_path.split('/')[1]:""})
        if len(image_path) > 1:
            caption_result = caption_image(image_path)
            result[image_path.split('/')[1]] = caption_result

    file = open("response.txt", "a")
    file.write("Caption result\n")
    file.write(str(result))
    file.write("\n")
    file.close()
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect(image_paths): system(f"python detect.py --source {image_paths} --save-crop")

@app.route('/upload', methods=['POST'])
def upload_images():
    '''
        Route to upload multiple image files using postman etc
    '''

    files = request.files.getlist('file')
    image_paths = ""

    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            image_path = path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            image_paths += image_path + ","
            
        else:
            return jsonify({'error': 'Invalid file type'})
        
    image_paths = image_paths.removesuffix(',')
    detect_image_p = Process(target=detect, args=[image_paths])
    cationing_process = Process(target=caption, args=[image_paths])

    detect_image_p.start()
    cationing_process.start()
    detect_image_p.join()
    cationing_process.join()

    response = {
    "caption_model_response" : {},
    "classification_model_response" : {}
    }
    
    file = open("./response.txt", 'r')
    file.readline()
    response["caption_model_response"] = loads(file.readline().replace('\'', '\"'))
    file.readline()
    response["classification_model_response"] = loads(file.readline().replace('\'', '\"'))
    file.close()

    return jsonify(response)

@app.route('/')
def upload_form():
    ''' home route'''
    return render_template('index.html')

@app.route('/view_results')
def view():
    '''
        Route for viewing Resultant images in html
    '''
    captions_image_names = listdir('./static/captions')
    detections = [folder for folder in listdir('./static/runs') if path.isdir(path.join('./static/runs', folder))]
    first_detection_images = []

    for detection_folder in detections:
        detection_images = sorted(listdir(path.join('./static/runs', detection_folder)))
        if detection_images:
            first_detection_images.append(path.join(detection_folder, detection_images[0]))

    
    with open('response.txt', 'r') as file:
        file_content = file.readlines()

    return render_template('view.html', captions_image_names=captions_image_names, detection_names=first_detection_images, file_content=file_content)

if __name__ == '__main__':
    app.run(debug=True)