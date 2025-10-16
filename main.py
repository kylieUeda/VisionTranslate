from ultralytics import YOLO
import cv2      # refers to OpenCV(Open Source Computer Vision Library)

# load a model
model = YOLO('yolov8n.pt')

# Access webcam
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()  # ret = a success flag (short for return)
    if not ret:
        break

# Run YOLO detection
results = model(frame)          # frame: the image from the webcam you want to analyze.
                                # YOLO runs its trained neural network on the frame and outputs 1) hte onjects' boundary, 2) confidence score, 3) class IDs
    
annotated = results[0].plot()   # results[0]: takes the first (and only) detection result
                                # .plot(): draw the boundary and labels on top of the image


# Show result
cv2.imshow("Detected", annotated)

# Press q to quit
if cv2.waitKey(1) & 0xFF == ord('q'):
    exit()

cap.release()
cv2.destroyAllWindows()