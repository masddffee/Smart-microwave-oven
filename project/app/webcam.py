import cv2
import threading


class WebCam:
    def __init__(self, device_id=0):  # 初始化
        self.device_id = device_id
        self.cap = cv2.VideoCapture(self.device_id)  # 開啟鏡頭
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.thread = None  # 執行緒
        self.is_opened = True  # 是否開啟
        self.ret = False  # 是否讀取到幀
        self.frame = None  # 幀

    def _update(self):  # 更新
        while self.is_opened:  # 如果開啟
            self.ret, self.frame = self.cap.read()

    def open(self):
        self.is_opened = True
        self.thread = threading.Thread(target=self._update)
        self.thread.daemon = True  # 設定為服務執行緒
        self.thread.start()  # 執行緒開始

    def close(self):
        self.is_opened = False
        self.thread.join()
        self.cap.release()

    def release(self):
        self.cap.release()

    def read(self):
        return self.ret, self.frame
    def isOpened(self):
        return self.is_opened            
        
