import time
import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from djitellopy import tello

# custom vision api
credentials = ApiKeyCredentials(in_headers={"Prediction-key": "you key"})
predictor = CustomVisionPredictionClient("https://birdcustomvisiontest.cognitiveservices.azure.com/", credentials)
projectID = "your project id"
publish_iteration_name="your published iteration name"

#飞行参数
lr, fb, ud, yv = 0, 0, 0, 0
speed = 50

#设置图像分辨率
img_height = 720
img_width = 1280

#单个图片最大识别对象数（按概率降序）
MAX_OBJ_COUNT = 5

#采纳识别结果的概率阈值
PROB_THERSHOLD = 0.6

#连接Tello无人机
me = tello.Tello()
me.connect()
#显示电池电量
print(me.get_battery())
#打开视频流
me.streamon()

#无限循环
while True:
    #复位无人机速度向量
    lr, fb, ud, yv = 0, 0, 0, 0
    #获取无人机图像
    frame = me.get_frame_read().frame
    frame = cv2.resize(frame, (img_width, img_height))
    # 在图像右上角提示用户如何操作
    cv2.putText(frame, "Press ESC to Quit, Press g to Start Detect. "+f"Battery: {me.get_battery()}", (800,30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (70, 140, 50), 1)

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
    #i，上升
    if flag == ord('i'):
        ud = -speed
    #k，下降
    if flag == ord('k'):
        ud = speed    
    #j，顺时针旋转
    if flag == ord('j'):
        yv = -speed
    #l，逆时针旋转
    if flag == ord('l'):
        yv = speed
    #发送飞行指令
    me.send_rc_control(lr, fb, ud, yv)    
    #响应事件，若按g键，则启动图像识别
    if flag == ord('g'):
        cv2.imwrite(f'Resources/Images/capture.jpg', frame)
        time.sleep(0.1)
        start_time = time.time()
        # open and detect the captured image
        with open(f'Resources/Images/capture.jpg', mode="rb") as captured_image:
            results = predictor.detect_image(projectID, publish_iteration_name, captured_image)
        # 计算本次识别消耗的时间（含网络传输）
        timeconsuming = time.time() - start_time
        # 输出结果.
        # 首先确认检测到的结果中是否有概率超过阈值的，如有，才展示。顺带按概率从大到小排个序
        filtered_sorted_results = sorted(list(filter(lambda x: x.probability > PROB_THERSHOLD, results.predictions)), key=lambda x : x.probability, reverse=True)
        if len(filtered_sorted_results) > 0:
            # 打印相关信息
            print("\n-------New Detection!------")

            # 记录展示的物体序号
            obj_index = 0

            # 遍历每个达到要求的检测，但只取前几个（具体几个由变量MAX_OBJ_COUNT控制），然后在图像上标注信息
            for prediction in filtered_sorted_results[:MAX_OBJ_COUNT]:
                # 在控制台打印检测到的物体信息
                print(f"\t No.{obj_index + 1}:  {prediction.tag_name}:{round(prediction.probability * 100, 2)}% ,time consuming: {round(timeconsuming, 2)} s")

                # 获取检测到的物体的矩形范围
                bbox = prediction.bounding_box

                # 在图像上打印检测到的物体信息
                cv2.putText(frame, f"No.{obj_index + 1}:  {prediction.tag_name}:{round(prediction.probability * 100, 2)}% ,time consuming: {round(timeconsuming, 2)} s", (10, 60 + (15 * obj_index)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

                # 根据bbox数据，用彩色矩形框选出物体  
                cv2.rectangle(frame,[int(bbox.left * img_width), int(bbox.top * img_height)], [int((bbox.left + bbox.width) * img_width),
                                int((bbox.top + bbox.height) * img_height)], (0, 255, 255), thickness=4)
                
                # 在物体矩形框左上角打上序号，以便和文字信息对应
                cv2.putText(frame, f"No.{obj_index + 1}", (int(bbox.left * img_width) + 15, int(bbox.top * img_height) + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)  

                # 物体序号递增
                obj_index += 1

            # 存储此图像
            cv2.imwrite(f'Resources/Images/results.jpg',frame)

            # 展示图像3秒
            cv2.imshow("results", frame)
            cv2.waitKey(3000)

            cv2.destroyWindow("results")

    # 展示图像
    cv2.imshow("capture", frame)

#降落无人机
me.land()
#关闭视频流
me.streamoff()
cv2.destroyAllWindows()