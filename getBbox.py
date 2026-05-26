#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from shipDetects import MainDetects


ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = ROOT / "detectModels" / "weights" / "best_0522.pt"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
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
        thickness,
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
        -1,
    )

    cv2.putText(
        image,
        text,
        (x1 + padding, y1 + padding + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_detection_result(
    image,
    detections,
    show_conf=True,
    show_size=True,
):
    img_h, img_w = image.shape[:2]

    for det in detections:
        cx = float(det.get("x", 0.0))
        cy = float(det.get("y", 0.0))
        w = float(det.get("w", 0.0))
        h = float(det.get("h", 0.0))

        cls_name = str(det.get("class", "unknown"))
        conf = float(det.get("conf", 0.0))

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label_parts = [cls_name]

        if show_conf:
            label_parts.append(f"{conf:.2f}")

        if show_size:
            label_parts.append(f"w:{w:.0f} h:{h:.0f}")

        label = " | ".join(label_parts)

        draw_black_label(
            image=image,
            text=label,
            x=x1,
            y=y1 - 36,
            font_scale=0.8,
            thickness=2,
            padding=6,
        )

    return image


def collect_images(input_dir: Path, recursive: bool):
    if recursive:
        image_paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    else:
        image_paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    image_paths.sort()
    return image_paths


def detect_folder(
    input_dir,
    output_dir,
    weights=DEFAULT_WEIGHTS,
    conf_thres=0.55,
    iou_thres=0.5,
    img_size=1280,
    device="",
    cam_index=0,
    recursive=False,
    save_json=False,
    show_conf=True,
    show_size=True,
):
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    if not weights.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] YOLO folder rendering")
    print("=" * 80)
    print(f"[INFO] input_dir : {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] weights   : {weights}")
    print(f"[INFO] conf_thres: {conf_thres}")
    print(f"[INFO] iou_thres : {iou_thres}")
    print(f"[INFO] img_size  : {img_size}")
    print(f"[INFO] recursive : {recursive}")
    print("=" * 80)

    detector = MainDetects(
        weights=weights,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        img_size=img_size,
        device=device,
        camIndex=cam_index,
        enable_log=False,
    )

    image_paths = collect_images(input_dir, recursive=recursive)

    if not image_paths:
        detector.close()
        print("[WARN] No images found.")
        return []

    results = []

    for image_path in tqdm(image_paths, desc="Detecting images"):
        image = cv2.imread(str(image_path))

        if image is None:
            results.append({
                "file": str(image_path),
                "output": None,
                "status": "failed_read",
                "detections": [],
            })
            continue

        detections = detector.run(image)

        rendered = render_detection_result(
            image=image.copy(),
            detections=detections,
            show_conf=show_conf,
            show_size=show_size,
        )

        if recursive:
            rel_path = image_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = output_dir / image_path.name

        ok = cv2.imwrite(str(output_path), rendered)

        results.append({
            "file": str(image_path),
            "output": str(output_path),
            "status": "ok" if ok else "failed_write",
            "detections": detections,
        })

    detector.close()

    if save_json:
        json_path = output_dir / "detection_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] JSON saved: {json_path}")

    print("=" * 80)
    print("[DONE]")
    print(f"[INFO] total images: {len(image_paths)}")
    print(f"[INFO] output_dir  : {output_dir}")
    print("=" * 80)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO folder detection renderer"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="輸入圖片資料夾，這個路徑就是原本 GUI 指定的資料夾",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="輸出渲染後圖片的資料夾",
    )

    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help=f"模型權重路徑，預設：{DEFAULT_WEIGHTS}",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.55,
        help="confidence threshold",
    )

    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=1280,
        help="YOLO input image size",
    )

    parser.add_argument(
        "--device",
        default="",
        help="推論裝置，例如 ''、'cpu'、'0'",
    )

    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="MainDetects camIndex",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="遞迴處理 input-dir 底下所有子資料夾圖片",
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="輸出 detection_results.json",
    )

    parser.add_argument(
        "--no-conf",
        action="store_true",
        help="標籤不顯示 confidence",
    )

    parser.add_argument(
        "--no-size",
        action="store_true",
        help="標籤不顯示 bbox w/h",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    detect_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        device=args.device,
        cam_index=args.cam_index,
        recursive=args.recursive,
        save_json=args.save_json,
        show_conf=not args.no_conf,
        show_size=not args.no_size,
    )


if __name__ == "__main__":
    main()