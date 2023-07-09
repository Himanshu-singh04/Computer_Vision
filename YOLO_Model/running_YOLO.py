from ultralytics import YOLO
import cv2

model = YOLO('C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\Computer_Vision\\YOLO_Weights\\yolov8n.pt')
result = model("C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\Computer_Vision\\image_1.jpg", show =True)
cv2.waitKey(0)