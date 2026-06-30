import cv2
from pathlib import Path

img_dir = Path("head_dataset/images/train")
lbl_dir = Path("head_dataset/labels/train")
out_dir = Path("head_dataset/_check"); out_dir.mkdir(exist_ok=True)

for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
    img = cv2.imread(str(img_path)); h, w = img.shape[:2]
    lbl = lbl_dir / (img_path.stem + ".txt")
    if lbl.exists() and lbl.stat().st_size:
        for line in lbl.read_text().splitlines():
            _, xc, yc, bw, bh = map(float, line.split())
            x1 = int((xc-bw/2)*w); y1 = int((yc-bh/2)*h)
            x2 = int((xc+bw/2)*w); y2 = int((yc+bh/2)*h)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(str(out_dir / img_path.name), img)
print("done — open the _check folder and scroll through")