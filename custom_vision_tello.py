import time
import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from djitellopy import tello

# custom vision api
credentials = ApiKeyCredentials(in_headers={"Prediction-key": "your key"})
predictor = CustomVisionPredictionClient("your endpoint", credentials)
projectID = "your projet id"
publish_iteration_name="your iteration name"

#飞行参数
lr, fb, ud, yv = 0, 0, 0, 0
speed = 50

#设置图像分辨率
img_height = 720
img_width = 1280
#连接Tello无人机
me = tello.Tello()
me.connect()
#显示电池电量
print(me.get_battery())
me.streamon()

#无限循环
while True:
    #复位无人机速度向量
    lr, fb, ud, yv = 0, 0, 0, 0
    #获取图片每一帧
    frame = me.get_frame_read().frame
    frame = cv2.resize(frame, (img_width, img_height))
    #获取键盘事件
    flag = cv2.waitKey(1)
    #Esc，退出
    if flag == 27:
        break
    #q，起飞
    if flag == ord('q'):
        me.land()
    #e，降落
    if flag == ord('e'):
        me.takeoff()     
    #a，左移
    if flag == ord('a'):
        lr = -speed
    #d，左移
    if flag == ord('d'):
        lr = speed    
    #w，前进
    if flag == ord('w'):
        fb = -speed
    #s，后退
    if flag == ord('s'):
        fb = speed
    #a，左移
    if flag == ord('i'):
        ud = -speed
    #d，左移
    if flag == ord('k'):
        ud = speed    
    #w，前进
    if flag == ord('j'):
        yv = -speed
    #s，后退
    if flag == ord('l'):
        yv = speed
    #发送飞行指令
    me.send_rc_control(lr, fb, ud, yv)    
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

#降落无人机
me.land()
me.streamoff()
cv2.destroyAllWindows()