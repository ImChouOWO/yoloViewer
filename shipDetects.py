import os, sys
from typing import Any, Dict, Optional
from pathlib import Path
import torch
import cv2
import torch.nn as nn
from detectModels.utils.datasets import letterbox
from detectModels.utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from detectModels.models.experimental import attempt_load
from detectModels.utils.torch_utils import select_device
import json

FILE = os.path.abspath(__file__)

detectModels_ROOT = os.path.join(os.path.dirname(FILE), 'detectModels')
if detectModels_ROOT not in sys.path:
    sys.path.insert(0, detectModels_ROOT)

CONFIG = f"{detectModels_ROOT}/config/config.json"
def get_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

class MainDetects:

    def __init__(
        self,
        weights=f"{detectModels_ROOT}/weights/modelData.bin",
        conf_thres=0.55,
        iou_thres=0.5,
        img_size=1280,
        device='',
        camIndex: int = 0,
        enable_log: bool = False,
        log_dir: str = "./logs",
        rotate_seconds: int = 60,
        write_every: int = 1,
        log_prefix: str = "yolo_det",
        camera_meta: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        config = get_config(CONFIG)
        
    

        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)

        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.eval()
        self.half = (self.device.type != 'cpu')
        if self.half:
            self.model.half()

        self.size_check = config.get("size_check", 60)
    
        if camIndex <= 0 or camIndex >=7:
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
        else:
            cam_conf = config.get(f"cam{camIndex}", {})
            self.conf_thres = cam_conf.get("conf_thres", conf_thres)
            self.iou_thres = cam_conf.get("iou_thres", iou_thres)

        img_size = config.get("img_size", img_size)
        self.model_ref = self.model.module if hasattr(self.model, "module") else self.model
        self.stride = int(self.model_ref.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = self.model_ref.names

        dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        dummy = dummy.half() if self.half else dummy
        with torch.no_grad():
            _ = self.model(dummy)

        self._enable_log = enable_log
        self._logger = None
        self._log_locations = None
        self._close_log = None
        self._camera_meta = camera_meta or {}
        self._extra_meta = extra_meta or {}

        # if self._enable_log:
        #     try:
        #         from saveLog import make_det_logger
        #         self._logger, self._log_locations, self._close_log = make_det_logger(
        #             log_dir=f"{log_dir}/cam{camIndex}",
        #             rotate_seconds=rotate_seconds,
        #             write_every=write_every,
        #             prefix=log_prefix,
        #         )
        #     except Exception:
        #         self._enable_log = False

    def close(self):
        if self._close_log:
            self._close_log()

    def shiptypeAdj(self, nowName: str, wh:list = [None,None], size_threshold: int = 60):
        nowName = nowName.strip().lower().replace(" ", "")
        w = wh[0]
        h = wh[1]
        if w > 0 and h > 0:
            if nowName in  [
                "ship",
                "tanker",
                "container",
                "bulkcarrier",
                "ferry",
                "cruiseship",
            ] :
                return "Merchant Vessel"

            elif nowName in [
                "patrolvessel",
                "localvessel",
            ]:
                return "Government Vessel"

            elif nowName in [
                "navalvessel",
            ]:
                return "Naval Vessel"

            elif nowName in [
                "tug",
                "pilotvessel",
            ]:
                return "Tug"

            elif nowName in [
                "fishing",
            ]:
                return "Fishing"

            elif nowName in [
                "sailingvessel",
                "pleasurecraft",
            ]:
                return "Pleasure Craft"

            elif nowName in [
                "sampan",
            ]:
                return "Small Craft"

            else:
                return "Merchant Vessel"
        else:
            return "Ship"

    def run(
        self,
        frame,
        camera: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.img_size = check_img_size(self.img_size, s=self.stride)

        img, _, _ = letterbox(frame, new_shape=self.img_size, auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        result = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                cx, cy, w, h = xywh
                cls = int(cls)
                result.append({
                    "x": cx,
                    "y": cy,
                    "w": w,
                    "h": h,
                    "conf": round(float(conf), 3),
                    "class": self.shiptypeAdj(self.names[cls],[w,h], self.size_check),
                })

        if self._enable_log and self._log_locations:
            cam_meta = dict(self._camera_meta)
            if camera:
                cam_meta.update(camera)
            if frame is not None and hasattr(frame, "shape"):
                cam_meta.setdefault("w", int(frame.shape[1]))
                cam_meta.setdefault("h", int(frame.shape[0]))

            ext_meta = dict(self._extra_meta)
            if extra:
                ext_meta.update(extra)

            self._log_locations(
                result,
                camera=cam_meta if cam_meta else None,
                extra=ext_meta if ext_meta else None,
            )

        return result



