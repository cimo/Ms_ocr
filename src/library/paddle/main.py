from paddleocr import PaddleOCR

ocr = PaddleOCR(
    device="cpu",
    lang="japan",
    text_detection_model_dir="/home/app/src/library/paddle/PP-OCRv5_server_det/",
    text_recognition_model_dir="/home/app/src/library/paddle/PP-OCRv5_server_rec/",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

ocrResultList = ocr.predict("/home/app/file/input/1_jp.jpg")

for ocrResult in ocrResultList:  
    ocrResult.print()  
    ocrResult.save_to_img("/home/app/file/output/paddle/")  
    ocrResult.save_to_json("/home/app/file/output/paddle/") 
