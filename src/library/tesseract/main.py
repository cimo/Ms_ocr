# Source
import helper

def pageLayoutOld():
    imgeRead = helper.preprocessorHelper.read("/home/app/file/output/craft/1_jp_heatmap_text.jpg")

    grayImage = helper.cv2.cvtColor(imgeRead, helper.cv2.COLOR_BGR2GRAY)

    #sobelX = helper.cv2.Sobel(grayImage, helper.cv2.CV_64F, 1, 0, ksize=3)
    #sobelY = helper.cv2.Sobel(grayImage, helper.cv2.CV_64F, 0, 1, ksize=3)
    #gradientMagnitude = helper.cv2.magnitude(sobelX, sobelY)
    #gradientMagnitude = helper.cv2.convertScaleAbs(gradientMagnitude)

    _, thresh = helper.cv2.threshold(grayImage, 180, 255, helper.cv2.THRESH_BINARY)

    kernel = helper.cv2.getStructuringElement(helper.cv2.MORPH_RECT, (5, 3))
    dilated = helper.cv2.dilate(thresh, kernel, iterations=2)

    helper.cv2.imwrite("/home/app/file/output/craft/thresh.jpg", thresh)

    contours, _ = helper.cv2.findContours(dilated, helper.cv2.RETR_EXTERNAL, helper.cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for a in contours:
        rect = helper.cv2.boundingRect(a)

        if rect[2] > 10 and rect[3] > 10:
            boxes.append(rect)

    filteredBoxes = []
    
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        isNested = False

        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j:
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    isNested = True
                    break

        if not isNested:
            filteredBoxes.append((x1, y1, w1, h1))

    for (x, y, w, h) in filteredBoxes:
        helper.cv2.rectangle(grayImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

    helper.cv2.imwrite("/home/app/file/output/craft/page_layout.jpg", grayImage)

    return thresh, filteredBoxes

def pageLayout():
    imgeRead = helper.preprocessorHelper.read("/home/app/file/output/craft/2_en_heatmap_text.jpg")
    grayImage = helper.cv2.cvtColor(imgeRead, helper.cv2.COLOR_BGR2GRAY)

    _, thresh = helper.cv2.threshold(grayImage, 180, 255, helper.cv2.THRESH_BINARY)

    kernel = helper.cv2.getStructuringElement(helper.cv2.MORPH_RECT, (5, 3))
    dilated = helper.cv2.dilate(thresh, kernel, iterations=1)

    helper.cv2.imwrite("/home/app/file/output/craft/dilated.jpg", dilated)

    contours, _ = helper.cv2.findContours(dilated, helper.cv2.RETR_EXTERNAL, helper.cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for a in contours:
        x, y, w, h = helper.cv2.boundingRect(a)

        if w > 4 and h > 4:
            boxes.append((x, y, w, h))

    # Rimuovi box nidificati
    filteredBoxes = []

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        isNested = False

        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j:
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    isNested = True

                    break

        if not isNested:
            filteredBoxes.append((x1, y1, w1, h1))

    # Step: Raggruppa per riga con tolleranza verticale
    verticalTolerance = 10
    horizontalGap = 3
    rows = []

    for box in filteredBoxes:
        x, y, w, h = box
        assigned = False

        for row in rows:
            ry = row[0][1]

            if abs(y - ry) <= verticalTolerance:
                row.append(box)
                assigned = True

                break

        if not assigned:
            rows.append([box])

    # Unisci box orizzontalmente all'interno di ogni riga
    mergedBoxes = []

    for row in rows:
        row.sort(key=lambda b: b[0])  # ordina da sinistra a destra
        current = row[0]

        for nextBox in row[1:]:
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = nextBox

            if x2 <= x1 + w1 + horizontalGap:
                # Unisci
                newX = min(x1, x2)
                newY = min(y1, y2)
                newW = max(x1 + w1, x2 + w2) - newX
                newH = max(y1 + h1, y2 + h2) - newY
                current = (newX, newY, newW, newH)
            else:
                mergedBoxes.append(current)
                current = nextBox

        mergedBoxes.append(current)

    # Disegna i box finali
    for (x, y, w, h) in mergedBoxes:
        helper.cv2.rectangle(grayImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

    helper.cv2.imwrite("/home/app/file/output/craft/page_layout.jpg", grayImage)

    return thresh, mergedBoxes


def Main():
    helper.executeCraft()

    #imageGray, imageBox, imageResult = helper.preprocess()

    #helper.result(imageGray, imageBox, imageResult)

    #thresh, boxList = pageLayout()

Main()

# TO DO - Integrate dewarp

#python3 main.py "1_jp.jpg" "jp" "False" "True"
