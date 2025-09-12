from paddleocr import PaddleOCR
import cv2

def _loadImage(imagePath: str):
    return cv2.imread(imagePath)

def _initializeOcr():
    return PaddleOCR(
        use_angle_cls=True, lang='japan',
        det_model_dir='modelli/ch_PP-OCRv3_det_infer',
        rec_model_dir='modelli/japan_PP-OCRv3_rec_infer',
        cls_model_dir='modelli/cls_infer'
    )

def _extractText(ocrModel, image):
    results = ocrModel.ocr(image, cls=True)
    extractedText = []

    for a in results:
        for b in a:
            extractedText.append(b[1][0])

    return extractedText

def runOcr(imagePath: str):
    image = _loadImage(imagePath)
    ocrModel = _initializeOcr()
    textList = _extractText(ocrModel, image)

    for a in textList:
        print(a)

# Esempio di utilizzo
if __name__ == "__main__":
    runOcr("/home/app/file/input/test_1_en.jpg")
