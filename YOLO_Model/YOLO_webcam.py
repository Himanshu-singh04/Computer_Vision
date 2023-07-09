from ultralytics import YOLO
import cv2
import cvzone

vid_cap = cv2.VideoCapture(0)
vid_cap.set(3,640) #width
vid_cap.set(4,480) #height


while True:
    success, img = vid_cap.read()
    model = YOLO('C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\Computer_Vision\\YOLO_Weights\\yolov8l.pt')
    result = model(img, stream = True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(x2),int(y1),int(y2)
                print(x1, x2, y1, y2)
                cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),4)
    cv2.imshow("image",img)
    cv2.waitKey(1)