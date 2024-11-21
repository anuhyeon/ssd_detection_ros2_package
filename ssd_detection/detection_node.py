import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from PIL import Image as PILImage

#from ssd_detection.misc import Timer  # 작성한 타이머 관련 코드 import

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        ###======================= 토픽 관련 세팅 =======================
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.image_subscription = self.create_subscription(
            Image,   # from sensor_msgs.msg import Image 메세지 타입 지정
            'camera/camera/color/image_raw',  # 퍼블리싱 노드에서 발행하는 RGB 이미지 토픽
            self.detect_callback,
            qos_profile
        )
        self.bridge = CvBridge()
        self.get_logger().info("Detection Node started")

        ###======================= SSD 설정 부분 =======================
        # Step 1: Initialize model with the best available weights
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device 사용 가능한지 확인
        #import pdb; pdb.set_trace()
        self.device = 'cuda'
        self.get_logger().info("Device: %s" % self.device)  # device 정보 확인

        # Load the pre-trained model with the best available weights
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.net = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=0.5).to(self.device)
        self.net.eval()

        # Initialize the inference transforms
        self.preprocess = self.weights.transforms()
        self.class_names = self.weights.meta["categories"]

        ###=============================================================

    def detect_callback(self, data):
        ###======================= 모델 추론 부분 =======================
        # ROS Image 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
        
        pil_image = PILImage.fromarray(cv_image)

        # 모델에 입력하기 위해 이미지 전처리
        input_image = pil_to_tensor(pil_image).unsqueeze(0).to(self.device)
        input_image = self.preprocess(input_image)

        # 모델 추론 수행
        with torch.no_grad():
            detections = self.net(input_image)[0]

        ###======================= 결과 시각화 코드 =======================
        # 결과 박스와 라벨 시각화
        boxes = detections['boxes']
        labels = detections['labels']
        #scores = detections['scores']
        print(labels)
        # 박스를 그릴 컬러 설정
        colors = [(255, 0, 0) for _ in range(len(boxes))]

        # 시각화
        pil_image = to_pil_image(input_image.squeeze(0).cpu())
        result_image = draw_bounding_boxes(
            pil_to_tensor(pil_image),
            boxes=boxes,
            labels=[self.class_names[i] for i in labels],
            colors=colors,
            width=3
        )
        cv_image_result = cv2.cvtColor(result_image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)


        # Displaying the predictions
        cv2.imshow('object_detection', cv_image_result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

