from ultralyticsplus import YOLO, render_result
import cv2
from ultralyticsplus import render_result
model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
video_path = r"C:\Users\karti\Downloads\pattern_live.mp4"
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
