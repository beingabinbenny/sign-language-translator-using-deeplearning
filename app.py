"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from flask import jsonify
import torch
from flask import Flask, render_template, request, redirect, Response
import time
import display


import openai 
import os 
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPEN_API_KEY"]

print(openai.api_key)

def chat_with_gpt(prompt):
    response = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()




app = Flask(__name__)




#'''
# Load Pre-trained Model
#model = torch.hub.load(
 #       "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
#        )#.autoshape()  # force_reload = recache latest code
#'''
# Load Custom Model
model = torch.hub.load("ultralytics/yolov5", "custom", path = "/Users/abinbenny/Downloads/300epochYolov5x.pt", force_reload=True)

# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

from io import BytesIO



def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    img_BGR = None  # Initialize img_BGR before the loop
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            # Convert the frame from BGR to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds of the skin color in HSV
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create a mask to segment the skin color
            mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

            # Apply a Gaussian blur to the mask
            blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

            # Apply a threshold to the blurred mask to get a binary image
            _, thresh = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour (assumed to be the hand)
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                # Create a mask for the largest contour
                hand_mask = np.zeros_like(mask)
                cv2.drawContours(hand_mask, [max_contour], -1, (255, 255, 255), -1)

                # Apply the hand mask to the original frame
                hand = cv2.bitwise_and(frame, frame, mask=hand_mask)

                # Set the background to white
                background = np.ones_like(frame, np.uint8) * 255

                # Invert the hand mask to get the background
                background_mask = cv2.bitwise_not(hand_mask)

                # Apply the inverted mask to the background frame
                background_removed = cv2.bitwise_and(background, background, mask=background_mask)

                # Combine the hand image and the background
                result = cv2.add(hand, background_removed)
            
                ret,buffer=cv2.imencode('.jpg',result)
                frame=buffer.tobytes()
                
                #print(type(frame))

                img = Image.open(io.BytesIO(frame))
                results = model(img, size=640)
                #print(results)
                #print(results.pandas().xyxy[0])
                #results.render()  # updates results.imgs with boxes and labels
                
                
                
                #results.print()  # print results to screen
                
                
                #results.show() 
                #print(results.imgs)
                #print(type(img))
                
                
                #print(results)
                # Check if there are no detections
                if len(results.xyxy[0]) == 0:
                    no_detection="no detections"
                    display.set_class_name(no_detection)
                    display.print_class_name()
                else:
                    # Extract class names from the detection results and print
                    class_indices = results.xyxy[0][:, -1].cpu().numpy().astype(int)
                    class_names = [results.names[i] for i in class_indices]

                    
                    
                    
                    # Convert class names to lowercase strings and get the first element
                    class_name = class_names[0].lower()

                    
                    display.set_class_name(class_name)
                    display.print_class_name()
                    # Print class_names and its data type
                    #print("Class name:", class_name)
                    #data_type_bytes = str(type(class_name)).encode('utf-8')
                    #print("Data type of class_name:", data_type_bytes)
                    
                    
                
                
                #plt.imshow(np.squeeze(results.render()))
                #print(type(img))
                #print(img.mode)
                
                #convert remove single-dimensional entries from the shape of an array
                img = np.squeeze(results.render()) #RGB
                # read image as BGR
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

                #print(type(img))
                #print(img.shape)
                #frame = img
                #ret,buffer=cv2.imencode('.jpg',img)
                #frame=buffer.tobytes()
                #print(type(frame))
                #for img in results.imgs:
                    #img = Image.fromarray(img)
                #ret,img=cv2.imencode('.jpg',img)
                #img=img.tobytes()

                #encode output image to bytes
                #img = cv2.imencode('.jpg', img)[1].tobytes()
                #print(type(img))
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



       


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user = request.form.get('myTextBox')
        response = chat_with_gpt(user)

        return render_template('index.html', user=user, response=response)
    return render_template('index.html')


@app.route('/get_class_name', methods=["GET"])
def get_class_name():
    class_name = display.print_class_name()
    return class_name





@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
'''                        
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
'''
'''
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)

        # for debugging
        # data = results.pandas().xyxy[0].to_json(orient="records")
        # return data

        results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")

    return render_template("index.html")
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    '''
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code
    model.eval()
    '''
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

# Docker Shortcuts
# docker build --tag yolov5 .
# docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device="/dev/video0:/dev/video0" yolov5
