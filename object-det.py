import cv2
import numpy as np
import requests

# URL of the Caffe model and prototxt file
caffe_model_url = 'https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel?raw=true'
prototxt_url = 'https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt?raw=true'

# Download the prototxt file
prototxt_response = requests.get(prototxt_url)
with open('deploy.prototxt', 'wb') as f:
    f.write(prototxt_response.content)

# Download the caffemodel file
caffemodel_response = requests.get(caffe_model_url)
with open('mobilenet_iter_73000.caffemodel', 'wb') as f:
    f.write(caffemodel_response.content)

# Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam")
        break
    
    if frame is None:
        print("Error: Captured frame is None")
        continue

    # Get the frame dimensions
    (h, w) = frame.shape[:2]
    
    # Preprocess the frame for the MobileNet SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])
            
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
