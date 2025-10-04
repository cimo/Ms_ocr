import os
import shutil
import json
import numpy
import cv2
import pandas
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition, PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

PATH_PADDLE_SYSTEM = "/home/app/.paddlex/"
PATH_PADDLE_FILE_OUTPUT = "/home/app/file/output/paddle/"
PATH_PADDLE_LIBRARY = "/home/app/src/library/paddle/"

pathFont = f"{PATH_PADDLE_SYSTEM}fonts/PingFang-SC-Regular.ttf"

pathModelLayout = f"{PATH_PADDLE_LIBRARY}PP-DocLayout_plus-L/"
pathModelTableClassification = f"{PATH_PADDLE_LIBRARY}PP-LCNet_x1_0_table_cls/"
pathModelTableWired = f"{PATH_PADDLE_LIBRARY}RT-DETR-L_wired_table_cell_det/"
pathModelTableWireless = f"{PATH_PADDLE_LIBRARY}RT-DETR-L_wireless_table_cell_det/"
pathModelTextDetection = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_server_det/"
pathModelTextRecognition = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_server_rec/"

isDebug = True
device = "cpu"

ocr = PaddleOCR(
    text_detection_model_dir=pathModelTextDetection,
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_dir=pathModelTextRecognition,
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device=device
)

def _inferenceTextDetection(input):
    model = TextDetection(model_dir=pathModelTextDetection, model_name="PP-OCRv5_server_det")
    outputList = model.predict(input=input, batch_size=1)

    for output in outputList:
        if isDebug:
            output.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.jpg")
            output.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.json")

def _inferenceTextRecognition(input):
    model = TextRecognition(model_dir=pathModelTextRecognition, model_name="PP-OCRv5_server_rec")
    outputList = model.predict(input=input, batch_size=1)

    for output in outputList:
        if isDebug:
            output.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.jpg")
            output.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.json")

def _filterOverlapBox(boxList):
    resultBoxFilterList = []

    for box in boxList:
        x1, y1, x2, y2 = map(float, box["coordinate"])
        
        isIgnore = False
        
        for boxFilter in resultBoxFilterList:
            xf1, yf1, xf2, yf2 = map(float, boxFilter["coordinate"])

            overlapX = max(0, min(x2, xf2) - max(x1, xf1))
            overlapY = max(0, min(y2, yf2) - max(y1, yf1))

            minWidth = min(x2 - x1, xf2 - xf1)
            minHeight = min(y2 - y1, yf2 - yf1)

            if overlapX / minWidth > 0.9 and overlapY / minHeight > 0.9:
                isIgnore = True

                break

        if isIgnore:
            continue

        resultBoxFilterList.append(box)
    
    return resultBoxFilterList
'''
def _jsonToExcel(data, fontSize, pathExcel):
    workbook = Workbook()
    sheet = workbook.active

    dataSortList = sorted(data, key=lambda x: (x["coordinate"][1], x["coordinate"][0]))

    rowEdgeList = set()

    for item in dataSortList:
        _, y1, _, y2 = item["coordinate"]

        rowEdgeList.add(round(y1, 1))
        rowEdgeList.add(round(y2, 1))

    rowEdgeList = sorted(rowEdgeList)

    rowObject = {}

    for a in range(len(rowEdgeList) - 1):
        rowObject[(rowEdgeList[a], rowEdgeList[a + 1])] = a + 1

    columnPositionList = []

    for dataSort in dataSortList:
        x1, _, _, _ = dataSort["coordinate"]

        if not any(abs(x1 - a) < 5 for a in columnPositionList):
            columnPositionList.append(x1)
    
    columnPositionList = sorted(columnPositionList)

    for dataSort in dataSortList:
        x1, y1, x2, y2 = dataSort["coordinate"]
        textList = dataSort["rec_texts"]

        columnIndex = None

        for key, value in enumerate(columnPositionList):
            if abs(x1 - value) < 5:
                columnIndex = key + 1

                break

        rowStart = None
        rowEnd = None

        for (ya, yb), rowNumber in rowObject.items():
            if abs(ya - y1) < 5:
                rowStart = rowNumber

            if abs(yb - y2) < 5:
                rowEnd = rowNumber

        if rowStart is None:
            for (ya, yb), rowNumber in rowObject.items():
                if ya <= y1 < yb:
                    rowStart = rowNumber

                    break

        if rowEnd is None:
            for (ya, yb), rowNumber in rowObject.items():
                if ya < y2 <= yb:
                    rowEnd = rowNumber

                    break

        if rowEnd is None:
            rowEnd = rowStart

        if rowEnd > rowStart:
            sheet.merge_cells(start_row=rowStart, start_column=columnIndex, end_row=rowEnd, end_column=columnIndex)

            if len(textList) > 1:
                totalLineList = rowEnd - rowStart + 1
                baseLineList = len(textList)

                if baseLineList < totalLineList:
                    extraSlot = totalLineList - baseLineList

                    lineList = []

                    for key, value in enumerate(textList):
                        lineList.append(value)

                        if key < baseLineList - 1:
                            lineList.append("\n")

                            if extraSlot > 0:
                                lineList.append("\n")

                                extraSlot -= 1

                    value = "".join(lineList)
                else:
                    value = "\n".join(textList)
            else:
                value = textList[0] if textList else ""
        else:
            value = textList[0] if len(textList) == 1 else "\n".join(textList)

        cell = sheet.cell(row=rowStart, column=columnIndex)
        cell.value = value
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        #columnLetter = get_column_letter(columnIndex)
        #sheet.column_dimensions[columnLetter].width = (x2 - x1) / 7
        #sheet.row_dimensions[rowStart].height = (y2 - y1) * 0.75

    workbook.save(pathExcel)
'''
def _processTable(mode, data, input, count):
    if isDebug:
        inputHeight, inputWidth = input.shape[:2]
        resultImage = Image.new("RGB", (inputWidth, inputHeight), (255, 255, 255))
        imageDraw = ImageDraw.Draw(resultImage)
        resultMergeList = []

    boxList = data.get("boxes", [])

    boxFilterList = _filterOverlapBox(boxList)

    for box in boxFilterList:
        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        imageCrop = input[y1:y2, x1:x2, :]

        resultList = ocr.predict(input=imageCrop)

        for result in resultList:
            coordinateList = [float(a) for a in coordinateList]
            textList = result.get("rec_texts", []) or [""]

            if isDebug:
                lineNumber = max(1, len(textList))
                cellHeight = y2 - y1
                fontSize = max(8, int(cellHeight * 0.6 / lineNumber))
                font = ImageFont.truetype(pathFont, fontSize)

                bboxeList = [imageDraw.textbbox((0, 0), value, font=font) for value in textList]
                textHeightList = [bbox[3] - bbox[1] for bbox in bboxeList]
                textWidthList = [bbox[2] - bbox[0] for bbox in bboxeList]

                totalTextHeight = sum(textHeightList)

                if lineNumber > 1:
                    extraSpace = (cellHeight - totalTextHeight) / (lineNumber + 1)
                else:
                    extraSpace = (cellHeight - totalTextHeight) / 2

                currentY = y1 + extraSpace

                for key, value in enumerate(textList):
                    textWidth = textWidthList[key]
                    textHeight = textHeightList[key]
                    textPositionX = x1 + (x2 - x1 - textWidth) // 2
                    textPositionY = int(currentY)

                    imageDraw.text((textPositionX, textPositionY), value, font=font, fill=(0, 0, 0))

                    currentY += textHeight + extraSpace

                resultMergeList.append({
                    "coordinate": coordinateList,
                    "rec_texts": textList
                })

    if isDebug:
        resultImage.save(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.jpg", format="JPEG")

        with open(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

    #_jsonToExcel(resultMergeList, fontSize, f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.xlsx")

def _inferenceTableWireless(input, count):
    model = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}.json")

        _processTable("wireless", data, input, count)

def _inferenceTableWired(input, count):
    model = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}.json")

        _processTable("wired", data, input, count)

def _extractTableCell(data, input, count):
    labelNameList = data.get("label_names", [])
    scoreList = data.get("scores", [])

    if len(labelNameList) == len(scoreList):
        resultIndex = int(numpy.argmax(scoreList))
        resultLabel = labelNameList[resultIndex]

        if (resultLabel == "wired_table"):
            if isDebug:
                cv2.imwrite(f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}_crop.jpg", input)

            _inferenceTableWired(input, count)
        elif (resultLabel == "wireless_table"):
            if isDebug:
                cv2.imwrite(f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}_crop.jpg", input)

            _inferenceTableWireless(input, count)

def _inferenceTableClassification(input, count):
    model = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/classification/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/classification/{count}.json")

    _extractTableCell(data, input, count)

def _extractTable(data, input):
    boxList = data.get("boxes", [])

    count = 0

    for box in boxList:
        if str(box.get("label", "")).lower() != "table":
            continue

        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        inputCrop = input[y1:y2, x1:x2, :]

        _inferenceTableClassification(inputCrop, count)

        count += 1

def removeOutputDir():
    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}layout/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}layout/")

    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}table/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}table/")

    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}export/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}export/")

def createOutputDir():
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}layout/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/classification/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/wired/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}export/", exist_ok=True)

def preprocess(pathImage, sizeLimit=2048):
    imageInput = Image.open(pathImage)
    imageInput = numpy.array(imageInput)

    height, width = imageInput.shape[:2]
    maxSide = max(height, width)

    if maxSide > sizeLimit:
        scale = sizeLimit / maxSide
        newWidth = int(width * scale)
        newHeight = int(height * scale)
        imageInput = cv2.resize(imageInput, (newWidth, newHeight))

    return imageInput

def inferenceLayout(input):
    model = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.json")

        _extractTable(data, input)

# Execution
#removeOutputDir()

#createOutputDir()

#imageInput = preprocess("/home/app/file/input/1_jp.jpg")

#inferenceLayout(imageInput)

with open(f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/0_result.json", "r", encoding="utf-8") as file:
    data = json.load(file)

#_jsonToExcel(data, 14, f"{PATH_PADDLE_FILE_OUTPUT}table/wired/1_result.xlsx")

class DynamicTableExtractor:
    """
    Estrae automaticamente tabelle da JSON PaddleOCR senza conoscere la struttura.
    Gestisce automaticamente celle mergiate in base alle coordinate.
    """
    
    def __init__(self, tolerance_y: float = 10, tolerance_x: float = 10):
        """
        Args:
            tolerance_y: Tolleranza in pixel per raggruppare righe
            tolerance_x: Tolleranza in pixel per raggruppare colonne
        """
        self.tolerance_y = tolerance_y
        self.tolerance_x = tolerance_x
        self.merge_info = []  # Lista di celle da mergiare (row, col_start, col_end)
    
    def extract_cells(self, json_data: List[Dict]) -> List[Dict]:
        """
        Estrae informazioni dalle celle.
        
        Returns:
            Lista di dict con x1, y1, x2, y2, center_x, center_y, text
        """
        cells = []
        for item in json_data:
            coord = item.get('coordinate', [])
            texts = item.get('rec_texts', [''])
            
            if len(coord) >= 4:
                x1, y1, x2, y2 = coord[:4]
                text = '\n'.join([t.strip() for t in texts if t.strip()])
                
                cells.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'text': text.strip(),
                    'width': x2 - x1,
                    'height': y2 - y1
                })
        
        return cells
    
    def cluster_values(self, values: List[float], tolerance: float) -> List[int]:
        """
        Raggruppa valori simili in cluster.
        
        Returns:
            Lista di indici cluster per ogni valore
        """
        if not values:
            return []
        
        sorted_indices = numpy.argsort(values)
        sorted_values = numpy.array(values)[sorted_indices]
        
        clusters = []
        current_cluster = 0
        cluster_assignment = [0] * len(values)
        
        cluster_assignment[sorted_indices[0]] = 0
        
        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] > tolerance:
                current_cluster += 1
            cluster_assignment[sorted_indices[i]] = current_cluster
        
        return cluster_assignment
    
    def detect_grid(self, cells: List[Dict]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Rileva automaticamente la griglia (righe e colonne).
        Usa i bordi sinistri (x1) invece dei centri per identificare le colonne.
        
        Returns:
            (row_indices, col_indices, row_positions, col_positions)
        """
        if not cells:
            return [], [], [], []
        
        # Estrai coordinate
        y_centers = [cell['center_y'] for cell in cells]
        x_lefts = [cell['x1'] for cell in cells]  # Usa bordo sinistro invece del centro
        
        # Cluster per righe (Y) - usa centro
        row_indices = self.cluster_values(y_centers, self.tolerance_y)
        
        # Cluster per colonne (X) - usa bordo sinistro
        col_indices = self.cluster_values(x_lefts, self.tolerance_x)
        
        # Calcola posizioni medie dei cluster
        num_rows = max(row_indices) + 1
        num_cols = max(col_indices) + 1
        
        row_positions = []
        for i in range(num_rows):
            row_y_values = [y_centers[j] for j in range(len(y_centers)) if row_indices[j] == i]
            row_positions.append(numpy.mean(row_y_values) if row_y_values else 0)
        
        col_positions = []
        for i in range(num_cols):
            col_x_values = [x_lefts[j] for j in range(len(x_lefts)) if col_indices[j] == i]
            col_positions.append(numpy.mean(col_x_values) if col_x_values else 0)
        
        return row_indices, col_indices, row_positions, col_positions
    
    def detect_merged_cells(self, cells: List[Dict], row_indices: List[int], 
                           col_indices: List[int], col_positions: List[float]) -> List[Dict]:
        """
        Rileva celle che dovrebbero essere mergiate in base alla loro larghezza.
        Usa x2 della cella per determinare quante colonne copre realmente.
        
        Returns:
            Lista di dict con informazioni sulle celle da mergiare
        """
        if not col_positions or len(col_positions) < 2:
            return []
        
        merged_cells = []
        
        for idx, cell in enumerate(cells):
            row_idx = row_indices[idx]
            col_idx = col_indices[idx]
            cell_x2 = cell['x2']
            
            # Trova quante colonne questa cella attraversa
            # controllando quali col_positions sono coperte dalla cella
            col_end = col_idx
            for i in range(col_idx + 1, len(col_positions)):
                # Se la posizione della prossima colonna è prima della fine della cella
                if col_positions[i] < cell_x2 - self.tolerance_x:
                    col_end = i
                else:
                    break
            
            # Se copre più di una colonna, è una merge
            if col_end > col_idx:
                merged_cells.append({
                    'cell_idx': idx,
                    'row': row_idx,
                    'col_start': col_idx,
                    'col_end': col_end,
                    'text': cell['text']
                })
        
        return merged_cells
    
    def build_table_matrix(self, cells: List[Dict], row_indices: List[int], 
                          col_indices: List[int], merged_cells: List[Dict]) -> pandas.DataFrame:
        """
        Costruisce la matrice della tabella gestendo le celle mergiate.
        
        Returns:
            DataFrame con la tabella ricostruita
        """
        if not cells:
            return pandas.DataFrame()
        
        # Determina dimensioni
        num_rows = max(row_indices) + 1
        num_cols = max(col_indices) + 1
        
        # Crea matrice vuota
        matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Crea set di celle mergiate per riferimento rapido
        merged_cells_set = {}
        for merge in merged_cells:
            key = (merge['row'], merge['col_start'])
            merged_cells_set[key] = merge
        
        # Riempi matrice
        for idx, cell in enumerate(cells):
            row_idx = row_indices[idx]
            col_idx = col_indices[idx]
            text = cell['text']
            
            # Controlla se questa cella è mergiata
            merge_key = (row_idx, col_idx)
            if merge_key in merged_cells_set:
                merge_info = merged_cells_set[merge_key]
                # Inserisci il testo solo nella prima colonna del merge
                matrix[row_idx][col_idx] = text
                # Salva info per mergiare in Excel
                self.merge_info.append({
                    'row': row_idx,
                    'col_start': col_idx,
                    'col_end': merge_info['col_end']
                })
            else:
                # Cella normale
                if matrix[row_idx][col_idx]:
                    matrix[row_idx][col_idx] += ' ' + text
                else:
                    matrix[row_idx][col_idx] = text
        
        # Crea DataFrame
        df = pandas.DataFrame(matrix)
        
        # Rimuovi righe completamente vuote
        df = df.replace('', pandas.NA)
        df = df.dropna(how='all')
        df = df.fillna('')
        
        # Rimuovi colonne completamente vuote
        df = df.loc[:, (df != '').any(axis=0)]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def extract_table(self, json_data: List[Dict]) -> pandas.DataFrame:
        """
        Estrae la tabella completa dal JSON.
        
        Args:
            json_data: Output JSON di PaddleOCR
        
        Returns:
            DataFrame con la tabella ricostruita
        """
        # Reset merge info
        self.merge_info = []
        
        # Estrai celle
        cells = self.extract_cells(json_data)
        
        if not cells:
            return pandas.DataFrame()
        
        # Rileva griglia usando bordi sinistri
        row_indices, col_indices, row_positions, col_positions = self.detect_grid(cells)
        
        # Rileva celle mergiate
        merged_cells = self.detect_merged_cells(cells, row_indices, col_indices, col_positions)
        
        # Costruisci tabella
        df = self.build_table_matrix(cells, row_indices, col_indices, merged_cells)
        
        return df
    
    def apply_merges_to_excel(self, worksheet, header_offset: int = 1):
        """
        Applica i merge alle celle in Excel e allinea tutto center/middle.
        
        Args:
            worksheet: Foglio Excel openpyxl
            header_offset: Offset per l'header (1 se c'è header, 0 altrimenti)
        """
        # Allinea tutte le celle center/middle
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        
        # Applica merge
        for merge in self.merge_info:
            row = merge['row'] + header_offset + 1  # +1 per Excel (1-based)
            col_start = merge['col_start'] + 1
            col_end = merge['col_end'] + 1
            
            if col_start < col_end:
                # Merge range
                start_cell = f"{get_column_letter(col_start)}{row}"
                end_cell = f"{get_column_letter(col_end)}{row}"
                merge_range = f"{start_cell}:{end_cell}"
                
                try:
                    worksheet.merge_cells(merge_range)
                    # Assicura allineamento center anche per celle mergiate
                    worksheet[start_cell].alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                except Exception as e:
                    print(f"⚠️  Impossibile mergiare {merge_range}: {e}")
    
    def to_excel(self, json_data: List[Dict], output_file: str, sheet_name: str = 'Sheet1') -> pandas.DataFrame:
        """
        Estrae la tabella e salva in Excel con celle mergiate.
        
        Args:
            json_data: Output JSON di PaddleOCR
            output_file: Nome file Excel di output
            sheet_name: Nome del foglio
        
        Returns:
            DataFrame estratto
        """
        df = self.extract_table(json_data)
        
        if df.empty:
            print("⚠️  Nessuna tabella trovata nel JSON")
            return df

        # Salva in Excel
        with pandas.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)
            
            worksheet = writer.sheets[sheet_name]
            
            # Applica merge (nessun offset header)
            self.apply_merges_to_excel(worksheet, header_offset=0)
            
            # Auto-dimensiona colonne
            for idx, column in enumerate(worksheet.columns, 1):
                max_length = 0
                column_letter = worksheet.cell(row=1, column=idx).column_letter
                
                for cell in column:
                    try:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                    except:
                        pass
                
                adjusted_width = min(max_length + 3, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✓ Tabella estratta: {len(df)} righe x {len(df.columns)} colonne")
        if self.merge_info:
            print(f"✓ Celle mergiate: {len(self.merge_info)}")
        print(f"✓ File salvato: {output_file}")
        
        return df
    
extractor = DynamicTableExtractor(tolerance_y=10, tolerance_x=15)
extractor.to_excel(data, f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/0_result.xlsx")