from ultralytics import YOLO
import cv2      # refers to OpenCV(Open Source Computer Vision Library)

# load a model
model = YOLO('yolov8n.pt')

# Access webcam
cap = cv2.VideoCapture(0)

# selected_id = None  # track the selected object

# def select_object(event, x, y, flags, param):
#     global selected_id
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # On click, find the closest detected object to the click
#         boxes = param[0].boxes.xyxy.cpu().numpy()
#         if len(boxes) > 0:
#             distances = [(abs((b[0]+b[2])/2 - x) + abs((b[1]+b[3])/2 - y)) for b in boxes]
#             selected_id = distances.index(min(distances))
#             print(f"Selected object #{selected_id}")

while True: 
    ret, frame = cap.read()  # ret = a success flag (short for return)
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)          # frame: the image from the webcam you want to analyze.
                                    # YOLO runs its trained neural network on the frame and outputs 1) hte onjects' boundary, 2) confidence score, 3) class IDs
        
    annotated = results[0].plot()   # results[0]: takes the first (and only) detection result
                                    # .plot(): draw the boundary and labels on top of the image

    # cv2.setMouseCallback("Detected", select_object, [results[0]])

    # # Highlight selected object
    # if selected_id is not None and len(results[0].boxes) > selected_id:
    #     box = results[0].boxes[selected_id].xyxy[0]
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)


    # Show result
    cv2.imshow("Detected", annotated)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()