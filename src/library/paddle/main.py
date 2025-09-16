from huggingface_hub import snapshot_download
from paddleocr import LayoutDetection, TableCellsDetection
from PIL import Image, ImageDraw

snapshot_download(
    repo_id="PaddlePaddle/PP-DocLayout_plus-L",
    local_dir="/home/app/src/library/paddle/PP-DocLayout_plus-L"
)

snapshot_download(
    repo_id="PaddlePaddle/RT-DETR-L_wired_table_cell_det",
    local_dir="/home/app/src/library/paddle/RT-DETR-L_wired_table_cell_det"
)

img_path = "/home/app/file/input/1_jp.jpg"
img = Image.open(img_path).convert("RGB")

# ============================
# 1. Layout Detection
# ============================
layout_model = LayoutDetection(model_name="PP-DocLayout_plus-L")
layout_output = layout_model.predict(img_path, batch_size=1, layout_nms=True)

# Salvataggio layout completo
for res in layout_output:
    res.save_to_img(save_path="/home/app/file/output/paddle/layout.jpg")
    res.save_to_json(save_path="/home/app/file/output/paddle/layout.json")

# ============================
# 2. Filtra solo tabelle e crea table.jpg
# ============================
tables = []

for res in layout_output:
    for box in res.json['res']['boxes']:  # <- rimuovere le parentesi
        if box['label'].lower() == "table":
            tables.append(box)

if tables:
    # Crea immagine bianca uguale all'originale
    table_img = Image.new('RGB', img.size, color=(255, 255, 255))
    draw = ImageDraw.Draw(table_img)

    # Riempi le aree delle tabelle con l'immagine originale
    for table_box in tables:
        x1, y1, x2, y2 = map(int, table_box['coordinate'])
        crop = img.crop((x1, y1, x2, y2))
        table_img.paste(crop, (x1, y1))

    table_img.save("/home/app/file/output/paddle/table.jpg")

    # ============================
    # 3. Rilevamento celle su table.jpg
    # ============================
    cell_model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
    output = cell_model.predict("/home/app/file/output/paddle/table.jpg", threshold=0.3, batch_size=1)

    # Salvataggio immagine e json con le celle
    for res in output:
        res.save_to_img(save_path="/home/app/file/output/paddle/table.jpg")   # sovrascrive table.jpg
        res.save_to_json(save_path="/home/app/file/output/paddle/table.json")