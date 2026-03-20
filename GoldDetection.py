from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("best.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)


# Define ROI coodrinate (fixed box)
roi_x1, roi_y1 = 200,100
roi_x2, roi_y2 = 500,400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Draw the ROI box on full frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255,0,0), 2)

    #Crop only ROI area for detection
    roi = frame[roi_y1:roi_y2 ,roi_x1:roi_x2]

    # Run detection
    results = model(roi, conf=0.1)[0]

    # Loop over ALL detected boxes (no class filtering)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()

        #conversion of ROI coordinates box to ful frame again
        x1 += roi_x1
        x2 += roi_x1
        y1 += roi_y1
        y2 += roi_y1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FORCE label to "Gold" for every detection
        label = f"Gold {conf:.2f}"
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("All Objects Shown as Gold", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

