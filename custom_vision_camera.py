import time
import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# custom vision api
credentials = ApiKeyCredentials(in_headers={"Prediction-key": "your key"})
predictor = CustomVisionPredictionClient("your endpoint", credentials)
projectID = "your project id"
publish_iteration_name="your publish iteration name"

#设置图像分辨率
img_height = 720
img_width = 1280
#打开默认摄像头
cap = cv2.VideoCapture(0)
#设置摄像头分辨率
cap.set(3, img_width)
cap.set(4, img_height)

#判断是否打开
if not cap.isOpened():
    raise IOError('Cannot open webcam!')

#无限循环
while True:
    #每一帧
    ret,frame = cap.read()
    #获取键盘事件
    flag = cv2.waitKey(1)
    #Esc，退出
    if flag == 27:
        break
    
    #响应事件
    if flag == ord('g'):
        cv2.imwrite(f'Resources/Images/capture.jpg', frame)
        time.sleep(0.1)
        start_time = time.time()
        # open and detect the captured image
        with open(f'Resources/Images/capture.jpg', mode="rb") as captured_image:
            results = predictor.detect_image(projectID, publish_iteration_name, captured_image)
        timeconsuming = time.time() - start_time
        # Display the results.
        if results.predictions[0].probability > 0.6:
            print("\t" + results.predictions[0].tag_name + ": {0:.2f}%".format(results.predictions[0].probability * 100) + " ,time consuming:" + str(timeconsuming))
            bbox = results.predictions[0].bounding_box
            cv2.putText(frame, str(results.predictions[0].tag_name + ": {0:.2f}%".format(results.predictions[0].probability * 100) + " ,time consuming:" + str(timeconsuming)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 1)               
            cv2.rectangle(frame,[int(bbox.left * img_width), int(bbox.top * img_height)], [int((bbox.left + bbox.width) * img_width),
                            int((bbox.top + bbox.height) * img_height)], (0, 255, 255), thickness=5)
            cv2.imwrite(f'Resources/Images/results.jpg',frame)
            cv2.imshow("results", frame)
            cv2.waitKey(1000)
            cv2.destroyWindow("results")

    cv2.imshow("capture", frame)

#关闭
cap.release()
cv2.destroyAllWindows()