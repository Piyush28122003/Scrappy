from flask import request, Response, Flask, jsonify
from waitress import serve
from PIL import Image
import json
import numpy as np
from keras.models import load_model
from flask_cors import CORS
# Define a list of class names corresponding to your model's output classes
class_names = ["Glass", "Metal", "Paper", "Plastic", "Battery", "Biological", "Trash"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    image = Image.open(buf.stream)
    boxes = detect_objects_on_image(image)
    json_boxes = json.dumps(boxes)
    print(json_boxes)
    return json_boxes

def detect_objects_on_image(image):
    model = load_model("./keras_model.h5", compile=False)
    # model.summary()  # Print the model summary

    input_shape = model.input_shape[1:3]
    resized_image = image.resize(input_shape)
    input_data = np.expand_dims(np.array(resized_image) / 255.0, axis=0)

    predictions = model.predict(input_data)

    boxes = []
    for prediction in predictions:
        x1, y1, x2, y2 = [round(coord) for coord in prediction[:4]]
        class_id = np.argmax(prediction)
        probability = round(float(prediction[class_id]), 2)
        object_type = class_names[class_id]
        boxes.append(object_type)

    # print(boxes)
    return boxes


if __name__ == '__main__':
    app.run(port=4000, debug=True)












#without AWS code 

# from flask import request, Flask, jsonify
# from PIL import Image
# import numpy as np
# from keras.models import load_model
# from flask_cors import CORS
# import json

# # Define a list of class names corresponding to your model's output classes
# class_names = ["Glass", "Metal", "Paper", "Plastic", "Battery", "Biological", "Trash"]

# app = Flask(__name__)
# CORS(app)

# # Load the model once during app initialization
# MODEL_PATH = "./keras_model.h5"  # Adjust this path if needed
# model = load_model(MODEL_PATH, compile=False)

# @app.route("/")
# def root():
#     # Returning HTML content directly for the root endpoint
#     with open("index.html") as file:
#         return file.read()

# @app.route("/detect", methods=["POST"])
# def detect():
#     buf = request.files["image_file"]
#     image = Image.open(buf.stream)
    
#     # Detecting objects in the uploaded image
#     boxes = detect_objects_on_image(image)
#     json_boxes = json.dumps(boxes)
#     print(json_boxes)
    
#     # Return the boxes as a JSON response
#     return jsonify(boxes)

# def detect_objects_on_image(image):
#     # Prepare image for model input (resize and normalize)
#     input_shape = model.input_shape[1:3]  # Get the model's input shape (height, width)
#     resized_image = image.resize(input_shape)  # Resize image to model's input size
#     input_data = np.expand_dims(np.array(resized_image) / 255.0, axis=0)  # Normalize image
    
#     # Predict using the model
#     predictions = model.predict(input_data)
    
#     # Prepare the detection results
#     boxes = []
#     for prediction in predictions:
#         x1, y1, x2, y2 = [round(coord) for coord in prediction[:4]]
#         class_id = np.argmax(prediction[4:])  # Assume class_id starts from index 4
#         probability = round(float(prediction[class_id]), 2)
#         object_type = class_names[class_id]
        
#         boxes.append({
#             'object_type': object_type,
#             'probability': probability,
#             'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
#         })
    
#     return boxes

# if __name__ == '__main__':
#     # Ensure that the app is only run in debug mode when not in production
#     app.run(port=4000, debug=True)


