import cv2
import torch
import torchvision.transforms as transforms
from torch import nn
import timm
class FacePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.preprocess(image).unsqueeze(0)

    # Adding the missing method to handle camera frames
    def load_and_preprocess_image_from_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.preprocess(frame).unsqueeze(0)


class DenseLandmarkPredictor(nn.Module):
    def __init__(self, num_landmarks=703):
        super(DenseLandmarkPredictor, self).__init__()
        self.num_landmarks = num_landmarks  # Save the number of landmarks as an instance variable
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
        self.regressor = nn.Linear(self.backbone.num_features, self.num_landmarks * 2)  # Use the instance variable here

    def forward(self, x):
        features = self.backbone(x)
        landmarks = self.regressor(features)
        return landmarks.view(-1, self.num_landmarks, 2)  # Use the instance variable here


class MorphableModel:
    def __init__(self):
        self.counter = 0

    def fit(self, landmarks_2d, image):
        self.counter += 1
        if self.counter % 50 == 0:
            print(f"Fitting model to landmarks... {self.counter}")


class CameraFaceProcessor:
    def __init__(self, camera_index=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = FacePreprocessor()
        self.predictor = DenseLandmarkPredictor().to(self.device)
        self.predictor.eval()
        self.morphable_model = MorphableModel()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
        if not self.cap.isOpened():
            raise Exception(f"Failed to open camera at index {camera_index}")

    def capture_and_process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame, trying again...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)  # Attempt to reinitialize the camera
                if not self.cap.isOpened():
                    print("Failed to reinitialize camera")
                    break
                continue
            frame_processed = self.preprocessor.load_and_preprocess_image_from_frame(frame)
            landmarks = self.predict_landmarks(frame_processed)
            self.morphable_model.fit(landmarks.cpu().detach(), frame_processed.cpu())
            self.display_frame_with_landmarks(frame, landmarks)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def predict_landmarks(self, image):
        image = image.to(self.device)
        return self.predictor(image)

    def display_frame_with_landmarks(self, frame, landmarks):
        landmarks = landmarks.squeeze().detach().cpu().numpy()  # Detach before converting to numpy
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow('Live Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def load_and_preprocess_image_from_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.preprocessor.preprocess(frame).unsqueeze(0)  # Add batch dimension




def find_working_camera():
    # Test typically used camera indices
    for index in range(0, 10):
        cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            print(f"No camera found at index {index}")
        else:
            print(f"Camera found at index {index}")
            cap.release()
            return index
    return -1

working_camera_index = find_working_camera()
print(f"Working camera index: {working_camera_index}")

processor = CameraFaceProcessor()
processor.capture_and_process()
