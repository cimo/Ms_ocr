import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import os


def download_east_model():
    """
    Download the pre-trained EAST text detector
    """
    model_url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    model_path = "/home/root/src/library/CRAFT-pytorch-master/model/frozen_east_text_detection.pb"

    if not os.path.exists(model_path):
        print("Downloading EAST text detection model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    return model_path


def load_east_model(model_path):
    """
    Load the EAST model using OpenCV's DNN module
    """
    net = cv2.dnn.readNet(model_path)
    return net


def decode_predictions(scores, geometry, min_confidence):
    """
    Decode the predictions from EAST model
    """
    rectangles = []
    confidences = []

    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(height):
        scores_data = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        angles_data = geometry[0][4][y]

        for x in range(width):
            score = scores_data[x]

            if score < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            endX = int(offsetX + (cos * x1_data[x]) + (sin * x2_data[x]))
            endY = int(offsetY - (sin * x1_data[x]) + (cos * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(score)

    return rectangles, confidences


def detect_text(image_path, min_confidence, width=1024, height=1024):
    """
    Detect text in an image using EAST model
    """
    # Load the pre-trained EAST model
    model_path = download_east_model()
    net = load_east_model(model_path)

    # Load and preprocess image
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Calculate new dimensions and ratios
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize image
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define output layers
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Create blob and perform forward pass
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Decode predictions
    (rectangles, confidences) = decode_predictions(scores, geometry, min_confidence)

    # Apply non-maximum suppression
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    # Scale boxes back to original image size
    results = []
    for startX, startY, endX, endY in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        results.append((startX, startY, endX, endY))

    return results


def non_max_suppression(boxes, probs=None, overlapThresh=0.2):
    """
    Apply non-maximum suppression to avoid overlapping boxes
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")


def visualize_detections(image_path, boxes):
    """
    Draw detected text regions on the image
    """
    image = cv2.imread(image_path)
    for startX, startY, endX, endY in boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
    return image


# Example usage
if __name__ == "__main__":
    # Path to your image
    image_path = "/home/root/src/library/CRAFT-pytorch-master/data/test_2.jpg"

    # Detect text
    boxes = detect_text(image_path, min_confidence=0.1)

    # Visualize results
    result_image = visualize_detections(image_path, boxes)
    cv2.imwrite(
        "/home/root/src/library/CRAFT-pytorch-master/result/detection_result.jpg",
        result_image,
    )
    print(f"Found {len(boxes)} text regions")
