import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
cap = cv2.VideoCapture('./video_1.mp4')

# Check if video loaded
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create named window once
cv2.namedWindow("YOLOv8 Object Tracking", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends or fails

    try:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot() if results and results[0] is not None else frame
    except Exception as e:
        print("Tracking error:", e)
        frame_ = frame

    # Show result in a single window
    cv2.imshow("YOLOv8 Object Tracking", frame_)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
# Save the last frame if needed
# cv2.imwrite('last_frame.jpg', frame_)

