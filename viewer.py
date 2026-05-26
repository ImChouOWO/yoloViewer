#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from shipDetects import MainDetects
from pathlib import Path

# ============================================================
# optional import
# tkfilebrowser 支援多選資料夾，但不是所有環境都一定有安裝
# ============================================================
try:
    if sys.platform == "darwin":
        os.environ.setdefault("LANG", "en_US.UTF-8")
        os.environ.setdefault("LC_ALL", "en_US.UTF-8")

    import tkfilebrowser

except Exception:
    tkfilebrowser = None


root = Path(__file__).parent
weights = root / "detectModels" / "weights" / "best_0522.pt"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


class YOLOFolderViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Model GUI Validator")
        self.root.geometry("1600x900")

        # =========================
        # YOLO 初始化設定
        # =========================
        self.weights_path = weights
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.cam_index = 1

        self.yoClass = None

        # =========================
        # 資料夾與圖片狀態
        # =========================
        self.folder_vars = []
        self.folder_paths = []

        self.image_paths = []
        self.current_index = 0

        # =========================
        # 推論與顯示狀態
        # =========================
        self.current_original_frame = None
        self.current_locations = []
        self.bbox_vars = []

        self.current_display_image = None
        self.last_result_frame = None

        self.is_running = False

        # 16:9 顯示比例
        self.display_aspect_w = 16
        self.display_aspect_h = 9

        self.build_ui()
        self.init_model_async()

    # ============================================================
    # GUI 建立
    # ============================================================
    def build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.btn_add_folders = tk.Button(
            top_frame,
            text="匯入多個資料夾",
            command=self.add_multiple_folders,
            width=20
        )
        self.btn_add_folders.pack(side=tk.LEFT, padx=5)

        self.btn_export_results = tk.Button(
            top_frame,
            text="匯出辨識結果",
            command=self.export_results_dialog,
            width=18
        )
        self.btn_export_results.pack(side=tk.LEFT, padx=5)

        self.btn_load_selected = tk.Button(
            top_frame,
            text="載入勾選資料夾",
            command=self.load_selected_folders,
            width=18
        )
        self.btn_load_selected.pack(side=tk.LEFT, padx=5)

        self.btn_prev = tk.Button(
            top_frame,
            text="上一張",
            command=self.prev_image,
            width=12
        )
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(
            top_frame,
            text="下一張",
            command=self.next_image,
            width=12
        )
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_rerun = tk.Button(
            top_frame,
            text="重新辨識目前圖片",
            command=self.rerun_current_image,
            width=18
        )
        self.btn_rerun.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(
            top_frame,
            text="清除資料夾",
            command=self.clear_folders,
            width=14
        )
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # =========================
        # 主畫面
        # =========================
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # =========================
        # 左側資料夾清單
        # =========================
        left_frame = tk.LabelFrame(main_frame, text="資料夾清單")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.folder_canvas = tk.Canvas(left_frame, width=340)
        self.folder_scrollbar = tk.Scrollbar(
            left_frame,
            orient="vertical",
            command=self.folder_canvas.yview
        )

        self.folder_list_frame = tk.Frame(self.folder_canvas)

        self.folder_list_frame.bind(
            "<Configure>",
            lambda e: self.folder_canvas.configure(
                scrollregion=self.folder_canvas.bbox("all")
            )
        )

        self.folder_canvas.create_window(
            (0, 0),
            window=self.folder_list_frame,
            anchor="nw"
        )

        self.folder_canvas.configure(
            yscrollcommand=self.folder_scrollbar.set
        )

        self.folder_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.folder_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # =========================
        # 中間圖像顯示區
        # =========================
        center_frame = tk.LabelFrame(main_frame, text="16:9 辨識結果")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_canvas = tk.Canvas(
            center_frame,
            bg="black",
            highlightthickness=0
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        # =========================
        # 右側資訊區
        # =========================
        right_frame = tk.LabelFrame(main_frame, text="目前圖片資訊 / bbox 顯示控制")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        info_frame = tk.LabelFrame(right_frame, text="圖片資訊")
        info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.info_text = tk.Text(info_frame, width=45, height=20)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bbox_control_frame = tk.LabelFrame(right_frame, text="bbox 個別顯示")
        bbox_control_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        bbox_btn_frame = tk.Frame(bbox_control_frame)
        bbox_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_select_all_bbox = tk.Button(
            bbox_btn_frame,
            text="全選 bbox",
            command=self.select_all_bboxes,
            width=12
        )
        self.btn_select_all_bbox.pack(side=tk.LEFT, padx=3)

        self.btn_unselect_all_bbox = tk.Button(
            bbox_btn_frame,
            text="取消全選",
            command=self.unselect_all_bboxes,
            width=12
        )
        self.btn_unselect_all_bbox.pack(side=tk.LEFT, padx=3)

        self.bbox_canvas = tk.Canvas(bbox_control_frame, width=360)
        self.bbox_scrollbar = tk.Scrollbar(
            bbox_control_frame,
            orient="vertical",
            command=self.bbox_canvas.yview
        )

        self.bbox_list_frame = tk.Frame(self.bbox_canvas)

        self.bbox_list_frame.bind(
            "<Configure>",
            lambda e: self.bbox_canvas.configure(
                scrollregion=self.bbox_canvas.bbox("all")
            )
        )

        self.bbox_canvas.create_window(
            (0, 0),
            window=self.bbox_list_frame,
            anchor="nw"
        )

        self.bbox_canvas.configure(
            yscrollcommand=self.bbox_scrollbar.set
        )

        self.bbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.bbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # =========================
        # 底部狀態列
        # =========================
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("初始化中...")

        self.status_label = tk.Label(
            bottom_frame,
            textvariable=self.status_var,
            anchor="w"
        )
        self.status_label.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=10,
            pady=5
        )

    # ============================================================
    # YOLO 初始化
    # ============================================================
    def init_model_async(self):
        thread = threading.Thread(target=self.init_model)
        thread.daemon = True
        thread.start()

    def init_model(self):
        try:
            self.set_status("正在載入 YOLO 模型...")

            self.yoClass = MainDetects(
                weights=self.weights_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                camIndex=self.cam_index,
                img_size=1280
            )

            self.set_status("YOLO 模型載入完成")

        except Exception as e:
            self.set_status("YOLO 模型載入失敗")
            self.show_error("模型載入錯誤", str(e))

    # ============================================================
    # macOS 多資料夾選擇
    # ============================================================
    def select_multiple_folders_macos(self):
        script = """
        set selectedFolders to choose folder with multiple selections allowed
        set outputText to ""
        repeat with oneFolder in selectedFolders
            set outputText to outputText & POSIX path of oneFolder & linefeed
        end repeat
        return outputText
        """

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return []

            folders = [
                os.path.abspath(line.strip())
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            return folders

        except Exception:
            return []

    # ============================================================
    # tkfilebrowser 多資料夾選擇
    # Windows / Linux 可用
    # ============================================================
    def select_multiple_folders_tkfilebrowser(self):
        try:
            if tkfilebrowser is None:
                return []

            folders = tkfilebrowser.askopendirnames(
                parent=self.root,
                title="選擇多個圖片資料夾"
            )

            if not folders:
                return []

            return [
                os.path.abspath(folder)
                for folder in folders
                if folder
            ]

        except Exception:
            return []

    # ============================================================
    # fallback 多資料夾選擇
    # 不是真正 Ctrl 多選，而是連續選取多個資料夾
    # 優點：不需要額外套件，Windows/macOS/Linux 都可用
    # ============================================================
    def select_multiple_folders_fallback(self):
        folders = []

        try:
            while True:
                folder = filedialog.askdirectory(
                    parent=self.root,
                    title="選擇圖片資料夾"
                )

                if not folder:
                    break

                folder = os.path.abspath(folder)

                if folder not in folders:
                    folders.append(folder)

                keep_selecting = messagebox.askyesno(
                    "繼續選擇",
                    "是否要繼續加入其他資料夾？"
                )

                if not keep_selecting:
                    break

            return folders

        except Exception as e:
            self.show_error("資料夾選擇錯誤", str(e))
            return []

    # ============================================================
    # 自動選擇多資料夾模式
    # ============================================================
    def select_multiple_folders_auto(self):
        folders = []

        # macOS 優先使用 osascript
        if sys.platform == "darwin":
            folders = self.select_multiple_folders_macos()

            if folders:
                self.set_status("使用 macOS 原生多資料夾選擇模式")
                return folders

        # Windows / Linux / macOS fallback 皆可嘗試 tkfilebrowser
        folders = self.select_multiple_folders_tkfilebrowser()

        if folders:
            self.set_status("使用 tkfilebrowser 多資料夾選擇模式")
            return folders

        # 最後退回 Tkinter 連續選取模式
        self.set_status("使用 Tkinter 連續選取資料夾模式")
        folders = self.select_multiple_folders_fallback()

        return folders
    
    def export_results_dialog(self):
        if self.yoClass is None:
            self.show_warning("提示", "YOLO 模型尚未載入完成")
            return

        if self.is_running:
            self.show_warning("提示", "目前仍在推論中，請稍後再匯出")
            return

        if not self.image_paths:
            self.show_warning(
                "提示",
                "尚未載入圖片。請先匯入資料夾並按下「載入勾選資料夾」。"
            )
            return

        output_dir = filedialog.askdirectory(
            parent=self.root,
            title="選擇匯出辨識結果的資料夾"
        )

        if not output_dir:
            self.set_status("已取消匯出")
            return

        thread = threading.Thread(
            target=self.export_results,
            args=(output_dir,)
        )
        thread.daemon = True
        thread.start()


    def export_results(self, output_dir):
        self.is_running = True

        try:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            total = len(self.image_paths)
            success_count = 0
            failed_count = 0

            self.set_status(f"開始匯出辨識結果，共 {total} 張圖片...")

            for idx, image_path in enumerate(self.image_paths, start=1):
                self.set_status(
                    f"正在匯出 {idx}/{total}：{os.path.basename(image_path)}"
                )

                frame = cv2.imread(image_path)

                if frame is None:
                    failed_count += 1
                    continue

                # 使用 GUI 當下已初始化的 YOLO 權重與參數
                locations = self.yoClass.run(frame)

                rendered = self.draw_results(
                    frame.copy(),
                    locations
                )

                output_path = self.build_export_output_path(
                    image_path=image_path,
                    output_dir=output_dir
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                ok = cv2.imwrite(output_path, rendered)

                if ok:
                    success_count += 1
                else:
                    failed_count += 1

            self.set_status(
                f"匯出完成：成功 {success_count} 張，失敗 {failed_count} 張，輸出路徑：{output_dir}"
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "匯出完成",
                    f"成功匯出 {success_count} 張圖片\n"
                    f"失敗 {failed_count} 張\n\n"
                    f"輸出路徑：\n{output_dir}"
                )
            )

        except Exception as e:
            self.set_status("匯出失敗")
            self.show_error("匯出錯誤", str(e))

        finally:
            self.is_running = False


    def build_export_output_path(self, image_path, output_dir):
        """
        產生匯出檔案路徑。

        若 image_path 來自已匯入的資料夾，會保留相對路徑，避免不同資料夾中同名圖片互相覆蓋。
        例如：
        input folder: /data/cam01
        image:        /data/cam01/a/b/001.jpg
        output:       /output/cam01/a/b/001.jpg
        """
        image_path = os.path.abspath(image_path)
        output_dir = os.path.abspath(output_dir)

        matched_folder = None

        for folder in self.folder_paths:
            folder_abs = os.path.abspath(folder)

            try:
                common = os.path.commonpath([image_path, folder_abs])
            except ValueError:
                continue

            if common == folder_abs:
                matched_folder = folder_abs
                break

        if matched_folder is not None:
            folder_name = os.path.basename(matched_folder.rstrip(os.sep))
            rel_path = os.path.relpath(image_path, matched_folder)
            output_path = os.path.join(output_dir, folder_name, rel_path)
        else:
            output_path = os.path.join(output_dir, os.path.basename(image_path))

        return output_path


    # ============================================================
    # 資料夾操作
    # ============================================================
    def add_multiple_folders(self):
        folders = self.select_multiple_folders_auto()

        if not folders:
            self.set_status("未選擇任何資料夾")
            return

        added_count = 0

        for folder in folders:
            before_count = len(self.folder_paths)
            self.add_folder_to_list(folder)

            if len(self.folder_paths) > before_count:
                added_count += 1

        self.set_status(f"已匯入 {added_count} 個資料夾")

    def add_single_folder(self):
        folder = filedialog.askdirectory(
            parent=self.root,
            title="選擇圖片資料夾"
        )

        if not folder:
            return

        before_count = len(self.folder_paths)
        self.add_folder_to_list(folder)

        if len(self.folder_paths) > before_count:
            self.set_status(f"已加入資料夾：{folder}")
        else:
            self.set_status("此資料夾已經存在")

    def add_folder_to_list(self, folder):
        folder = os.path.abspath(folder)

        if folder in self.folder_paths:
            return

        self.folder_paths.append(folder)

        var = tk.BooleanVar(value=True)
        self.folder_vars.append(var)

        row = tk.Frame(self.folder_list_frame)
        row.pack(fill=tk.X, anchor="w", padx=5, pady=3)

        cb = tk.Checkbutton(
            row,
            variable=var,
            text=os.path.basename(folder),
            anchor="w",
            justify="left"
        )
        cb.pack(side=tk.TOP, anchor="w")

        path_label = tk.Label(
            row,
            text=folder,
            fg="gray",
            wraplength=310,
            justify="left",
            anchor="w"
        )
        path_label.pack(side=tk.TOP, anchor="w")

    def clear_folders(self):
        self.folder_paths.clear()
        self.folder_vars.clear()
        self.image_paths.clear()
        self.current_index = 0

        self.current_original_frame = None
        self.current_locations = []
        self.bbox_vars.clear()

        self.last_result_frame = None

        for widget in self.folder_list_frame.winfo_children():
            widget.destroy()

        for widget in self.bbox_list_frame.winfo_children():
            widget.destroy()

        self.image_canvas.delete("all")
        self.info_text.delete("1.0", tk.END)

        self.set_status("已清除所有資料夾")

    def load_selected_folders(self):
        selected_folders = []

        for folder, var in zip(self.folder_paths, self.folder_vars):
            if var.get():
                selected_folders.append(folder)

        if not selected_folders:
            self.show_warning("提示", "請至少勾選一個資料夾")
            return

        image_paths = []

        for folder in selected_folders:
            for root_dir, _, files in os.walk(folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()

                    if ext in IMAGE_EXTS:
                        image_paths.append(os.path.join(root_dir, file))

        image_paths.sort()

        if not image_paths:
            self.show_warning("提示", "勾選的資料夾中沒有找到圖片")
            return

        self.image_paths = image_paths
        self.current_index = 0

        self.set_status(f"已載入 {len(self.image_paths)} 張圖片")
        self.run_current_image_async()

    # ============================================================
    # 圖片切換
    # ============================================================
    def prev_image(self):
        if not self.image_paths:
            self.set_status("尚未載入圖片")
            return

        if self.current_index > 0:
            self.current_index -= 1
            self.run_current_image_async()
        else:
            self.set_status("目前已經是第一張")

    def next_image(self):
        if not self.image_paths:
            self.set_status("尚未載入圖片")
            return

        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.run_current_image_async()
        else:
            self.set_status("目前已經是最後一張")

    def rerun_current_image(self):
        if not self.image_paths:
            self.set_status("尚未載入圖片")
            return

        self.run_current_image_async()

    # ============================================================
    # 推論流程
    # ============================================================
    def run_current_image_async(self):
        if self.is_running:
            self.set_status("目前仍在推論中")
            return

        if self.yoClass is None:
            self.show_warning("提示", "YOLO 模型尚未載入完成")
            return

        thread = threading.Thread(target=self.run_current_image)
        thread.daemon = True
        thread.start()

    def run_current_image(self):
        self.is_running = True

        try:
            image_path = self.image_paths[self.current_index]

            self.set_status(
                f"正在辨識 {self.current_index + 1}/{len(self.image_paths)}："
                f"{os.path.basename(image_path)}"
            )

            frame = cv2.imread(image_path)

            if frame is None:
                self.set_status(f"讀取圖片失敗：{image_path}")
                return

            # ====================================================
            # 核心辨識方法
            # 保持你的原本邏輯：只呼叫 MainDetects.run(frame)
            # ====================================================
            locations = self.yoClass.run(frame)

            self.root.after(
                0,
                lambda: self.update_current_result_ui(image_path, frame, locations)
            )

        except Exception as e:
            self.set_status("推論失敗")
            self.show_error("推論錯誤", str(e))

        finally:
            self.is_running = False

    def update_current_result_ui(self, image_path, frame, locations):
        self.current_original_frame = frame.copy()
        self.current_locations = locations

        self.build_bbox_checkboxes(locations)

        self.redraw_current_result()
        self.show_info(image_path, locations)

        self.set_status(
            f"完成 {self.current_index + 1}/{len(self.image_paths)}，"
            f"偵測數量：{len(locations)}"
        )

    # ============================================================
    # bbox 個別顯示控制
    # ============================================================
    def build_bbox_checkboxes(self, locations):
        for widget in self.bbox_list_frame.winfo_children():
            widget.destroy()

        self.bbox_vars.clear()

        if not locations:
            empty_label = tk.Label(
                self.bbox_list_frame,
                text="目前圖片沒有偵測到 bbox",
                fg="gray",
                anchor="w",
                justify="left"
            )
            empty_label.pack(fill=tk.X, padx=5, pady=5)
            return

        for i, obj in enumerate(locations):
            var = tk.BooleanVar(value=True)
            self.bbox_vars.append(var)

            cls_name = obj.get("class", "unknown")
            conf = float(obj.get("conf", 0.0))

            x = float(obj.get("x", 0.0))
            y = float(obj.get("y", 0.0))
            bw = float(obj.get("w", 0.0))
            bh = float(obj.get("h", 0.0))

            label_text = (
                f"#{i + 1} {cls_name} {conf:.2f}\n"
                f"x={x:.1f}, y={y:.1f}, w={bw:.1f}, h={bh:.1f}"
            )

            cb = tk.Checkbutton(
                self.bbox_list_frame,
                text=label_text,
                variable=var,
                command=self.redraw_current_result,
                anchor="w",
                justify="left",
                wraplength=330
            )
            cb.pack(fill=tk.X, padx=5, pady=4, anchor="w")

    def select_all_bboxes(self):
        for var in self.bbox_vars:
            var.set(True)

        self.redraw_current_result()

    def unselect_all_bboxes(self):
        for var in self.bbox_vars:
            var.set(False)

        self.redraw_current_result()

    def get_visible_locations(self):
        if not self.current_locations:
            return []

        if not self.bbox_vars:
            return self.current_locations

        visible_locations = []

        for obj, var in zip(self.current_locations, self.bbox_vars):
            if var.get():
                visible_locations.append(obj)

        return visible_locations

    def redraw_current_result(self):
        if self.current_original_frame is None:
            return

        visible_locations = self.get_visible_locations()

        result_frame = self.draw_results(
            self.current_original_frame.copy(),
            visible_locations
        )

        self.show_image_16_9(result_frame)

    # ============================================================
    # 繪製 bbox 結果
    # ============================================================
    def draw_results(self, frame, locations):
        h, w = frame.shape[:2]

        for obj in locations:
            try:
                x = float(obj["x"])
                y = float(obj["y"])
                bw = float(obj["w"])
                bh = float(obj["h"])

                x1 = int(x - bw / 2)
                y1 = int(y - bh / 2)
                x2 = int(x + bw / 2)
                y2 = int(y + bh / 2)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                cls_name = obj.get("class", "unknown")
                conf = float(obj.get("conf", 0.0))

                label = f"{cls_name} | w:{int(bw)} h:{int(bh)}"

                box_color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                # =========================
                # 黑底白字標籤
                # =========================
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.4
                font_thickness = 3

                text_size, baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    font_thickness
                )

                text_w, text_h = text_size

                text_x = x1
                text_y = y1 - 12

                if text_y - text_h - baseline < 0:
                    text_y = y1 + text_h + 12

                bg_x1 = text_x
                bg_y1 = text_y - text_h - baseline - 8
                bg_x2 = text_x + text_w + 12
                bg_y2 = text_y + baseline + 8

                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(w - 1, bg_x2)
                bg_y2 = min(h - 1, bg_y2)

                cv2.rectangle(
                    frame,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (text_x + 6, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

            except Exception:
                continue

        return frame

    # ============================================================
    # 16:9 顯示邏輯
    # ============================================================
    def get_16_9_area(self):
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()

        if canvas_w <= 1:
            canvas_w = 900

        if canvas_h <= 1:
            canvas_h = 600

        target_ratio = self.display_aspect_w / self.display_aspect_h
        canvas_ratio = canvas_w / canvas_h

        if canvas_ratio > target_ratio:
            area_h = canvas_h
            area_w = int(area_h * target_ratio)
        else:
            area_w = canvas_w
            area_h = int(area_w / target_ratio)

        offset_x = (canvas_w - area_w) // 2
        offset_y = (canvas_h - area_h) // 2

        return area_w, area_h, offset_x, offset_y

    def show_image_16_9(self, frame):
        self.last_result_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        area_w, area_h, offset_x, offset_y = self.get_16_9_area()

        img_w, img_h = pil_img.size

        scale = min(area_w / img_w, area_h / img_h)

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if new_w <= 0:
            new_w = 1

        if new_h <= 0:
            new_h = 1

        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        canvas_img = Image.new("RGB", (area_w, area_h), (0, 0, 0))

        paste_x = (area_w - new_w) // 2
        paste_y = (area_h - new_h) // 2

        canvas_img.paste(resized, (paste_x, paste_y))

        self.current_display_image = ImageTk.PhotoImage(canvas_img)

        self.image_canvas.delete("all")

        self.image_canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self.current_display_image
        )

        self.image_canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + area_w,
            offset_y + area_h,
            outline="white",
            width=2
        )

    def on_canvas_resize(self, event):
        if self.current_original_frame is not None:
            self.redraw_current_result()

    # ============================================================
    # 右側資訊顯示
    # ============================================================
    def show_info(self, image_path, locations):
        self.info_text.delete("1.0", tk.END)

        self.info_text.insert(tk.END, f"圖片路徑:\n{image_path}\n\n")

        self.info_text.insert(
            tk.END,
            f"目前進度:\n{self.current_index + 1} / {len(self.image_paths)}\n\n"
        )

        self.info_text.insert(tk.END, f"偵測數量:\n{len(locations)}\n\n")
        self.info_text.insert(tk.END, "辨識結果:\n")

        for i, obj in enumerate(locations, start=1):
            self.info_text.insert(tk.END, "-" * 42 + "\n")
            self.info_text.insert(tk.END, f"No. {i}\n")

            for key, value in obj.items():
                self.info_text.insert(tk.END, f"{key}: {value}\n")

        self.info_text.insert(tk.END, "-" * 42 + "\n")

    # ============================================================
    # UI helper
    # ============================================================
    def set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def show_error(self, title, message):
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def show_warning(self, title, message):
        self.root.after(0, lambda: messagebox.showwarning(title, message))


if __name__ == "__main__":
    root = tk.Tk()
    
    app = YOLOFolderViewer(root)
    root.mainloop()