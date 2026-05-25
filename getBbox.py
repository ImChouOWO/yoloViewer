import os
from pathlib import Path

import cv2
from tqdm import tqdm

from shipDetects import MainDetects
from pathlib import Path
root = Path(__file__).parent
weights = root / "detectModels" / "weights" / "best_0511.pt"


INPUT_DIR = "imgs"
OUTPUT_DIR = "output"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",".bmp",
}



def draw_black_label(
    image,
    text,
    x,
    y,
    font_scale=0.8,
    thickness=2,
    padding=6,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_w, text_h = text_size

    x1 = int(x)
    y1 = int(y)
    x2 = x1 + text_w + padding * 2
    y2 = y1 + text_h + padding * 2 + baseline

    img_h, img_w = image.shape[:2]

    if x2 > img_w:
        x1 = max(0, img_w - (text_w + padding * 2))
        x2 = img_w

    if y1 < 0:
        y1 = 0
        y2 = text_h + padding * 2 + baseline

    if y2 > img_h:
        y2 = img_h
        y1 = max(0, img_h - (text_h + padding * 2 + baseline))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (x1 + padding, y1 + padding + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )



def render_detection_result(image, detections):
    for det in detections:
        cx = float(det["x"])
        cy = float(det["y"])
        w = float(det["w"])
        h = float(det["h"])

        cls_name = det["class"]
        conf = float(det.get("conf", 0.0))

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # bbox
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 黑底白字：cls / w / x
        label = f"cls: {cls_name} | w: {int(w)} | h: {int(h)}"

        draw_black_label(
            image=image,
            text=label,
            x=x1,
            y=y1 - 36,
            font_scale=0.8,
            thickness=2,
            padding=6
        )

    return image



def detect_folder(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = MainDetects(
        weights=weights,
        conf_thres=0.55,
        iou_thres=0.5,
        img_size=1280,
        device="",
        camIndex=0,
        enable_log=False,
    )

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    results = []

    for image_path in tqdm(image_paths, desc="Detecting images"):
        image = cv2.imread(str(image_path))

        if image is None:
            results.append({
                "file": str(image_path),
                "status": "failed_read",
                "detections": []
            })
            continue

        detections = detector.run(image)

        rendered = render_detection_result(
            image=image.copy(),
            detections=detections
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), rendered)

        results.append({
            "file": str(image_path),
            "output": str(output_path),
            "status": "ok",
            "detections": detections
        })

    detector.close()

    return results



def run_detection():
    return detect_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )



if __name__ == "__main__":
    run_detection()