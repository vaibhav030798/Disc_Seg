import numpy as np
import cv2
import os
import glob

class OpticDiscExtractor:
    def __init__(self, model_config, model_weights, labels_path, confidence_threshold=0.3, nms_threshold=0.3):
        self.model_config = model_config
        self.model_weights = model_weights
        try:
            self.labels = open(labels_path).read().strip().split('\n')
        except Exception as e:
            print(f"Error reading labels file: {e}")
            self.labels = []
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.model_config, self.model_weights)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            self.net = None

    def extract_optic_discs(self, images_folder_path, output_folder_path):
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for file in glob.glob(images_folder_path):
            try:
                print('Image Name :', file)
                image = cv2.imread(file)
                if image is None:
                    print(f"Unable to read image: {file}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                (H, W) = image.shape[:2]

                layer_name = self.net.getUnconnectedOutLayersNames()
                blob = cv2.dnn.blobFromImage(image_rgb, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                layers_outputs = self.net.forward(layer_name)

                boxes = []
                confidences = []
                class_ids = []

                for output in layers_outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > self.confidence_threshold:
                            if self.labels[class_id] == 'optic disc':
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY,  width, height) = box.astype('int')
                                x = int(centerX - (width/2))
                                y = int(centerY - (height/2))

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                detection_nms = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

                if len(detection_nms) > 0:
                    for i in detection_nms.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # Increase crop image size
                        x -= 10
                        y -= 10
                        w += 20
                        h += 20
                        optic_disc_region = image[max(0, y):min(y+h, H), max(0, x):min(x+w, W)]
                        filename = os.path.splitext(os.path.basename(file))[0] + "_optic_disc.jpg"
                        cv2.imwrite(os.path.join(output_folder_path, filename), optic_disc_region)
            except Exception as e:
                print(f"Error processing image {file}: {e}")

# Usage
if __name__ == "__main__":
    model_config = 'config.cfg'
    model_weights = 'model.weights'  # model weights
    labels_path = 'obj.names'
    images_folder_path = r"C:\Vaibhav-AI\NetrAI\Glaucoma_competition\JustRAIGS challenge_training_dataset\JustRAIGS challenge_training_dataset\JustRAIGS_Train_4\4\*.jpg"
    output_folder_path = r"C:\Vaibhav-AI\NetrAI\Glaucoma_competition\cupTodiscRatio\disc-seg\crop_image"  # save crop image folder path

    extractor = OpticDiscExtractor(model_config, model_weights, labels_path)
    extractor.extract_optic_discs(images_folder_path, output_folder_path)
