from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
index = 0

class VideoCamera(object):

    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
        self.background = self.get_background()

    def __del__(self):
        self.video.release()

    def get_background(self):
        global index
        cap = self.video
        time.sleep(3)
        background = 0
        for i in range(30):
            ret, background = cap.read()
        background = np.flip(background, axis=1)
        return background

    def get_frame(self):

        # success, image = self.video.read()
        # # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)

        background =self.background
        cap = self.video
        ret, img = cap.read()

        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = (35, 35)
        blurred = cv2.GaussianBlur(hsv, value, 0)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        img[np.where(mask == 255)] = background[np.where(mask == 255)]
        # cv2.imshow('Display', img)
        k = cv2.waitKey(10)
        cv2.imwrite('test.jpg',img)
        img2 = cv2.imread('test.jpg')
        # jpg = cv2.imencode('.jpg', img2)
        # return jpg.tobytes()
        return cv2.imencode('.jpg',img2)[1].tostring()


app = Flask(__name__)
@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 5000)
