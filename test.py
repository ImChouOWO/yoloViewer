#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path


# =========================
# Path setup
# =========================

root = Path(__file__).resolve().parent
detect_root = root / "detectModels"
default_weights = detect_root / "weights" / "best_0522.pt"

# YOLOv7 checkpoint 反序列化時會需要 models / utils
# 因此必須在 torch.load() 之前加入 sys.path
if str(detect_root) not in sys.path:
    sys.path.insert(0, str(detect_root))

import torch


# =========================
# Expected class mapping
# =========================

EXPECTED_NAMES = {
    0: "WIG-Wing In Grnd",
    1: "Hydrofoil",
    2: "Fishing",
    3: "Tug",
    4: "Dredger",
    5: "Naval Vessel",
    6: "Sailing Vessel",
    7: "Pleasure Craft",
    8: "Hovercraft",
    9: "Pilot Vessel",
    10: "Patrol Vessel",
    11: "Local Vessel",
    12: "Ferry",
    13: "Cruise Ship",
    14: "Container",
    15: "Bulk Carrier",
    16: "Tanker",
    17: "OtherTypeShip",
    18: "Ship",
    19: "sampan",
}


def print_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def get_name(names_obj, idx: int):
    if names_obj is None:
        return None

    if isinstance(names_obj, dict):
        if idx in names_obj:
            return names_obj[idx]
        if str(idx) in names_obj:
            return names_obj[str(idx)]
        return None

    if isinstance(names_obj, (list, tuple)):
        if 0 <= idx < len(names_obj):
            return names_obj[idx]
        return None

    return None


def normalize_names(names_obj):
    """
    將 model.names 轉成 0-based dict，方便檢查。
    支援：
    - list / tuple
    - dict[int, str]
    - dict[str, str]
    """
    if names_obj is None:
        return None

    if isinstance(names_obj, dict):
        output = {}
        for k, v in names_obj.items():
            try:
                output[int(k)] = v
            except Exception:
                output[k] = v
        return output

    if isinstance(names_obj, (list, tuple)):
        return {i: v for i, v in enumerate(names_obj)}

    return None


def inspect_model(model, title: str):
    print_section(f"[INFO] Inspecting {title}")

    nc = getattr(model, "nc", None)
    names = getattr(model, "names", None)

    print("model type        =", type(model))
    print("model.nc          =", nc)
    print("model.names type  =", type(names))

    if names is None:
        print("[ERROR] model.names not found.")
        return False

    names_dict = normalize_names(names)

    if names_dict is None:
        print("[ERROR] Unsupported model.names format:", type(names))
        print("raw names =", names)
        return False

    print_section(f"[INFO] Raw names from {title}")
    print(names)

    print_section(f"[INFO] Class index mapping from {title}")

    for idx in sorted(names_dict.keys()):
        print(f"{int(idx):2d}: {names_dict[idx]}")

    name_count = len(names_dict)

    print_section(f"[INFO] Consistency check for {title}")
    print("len(names) =", name_count)
    print("model.nc   =", nc)

    nc_ok = True

    if nc is not None:
        try:
            nc_ok = int(nc) == int(name_count)
        except Exception:
            nc_ok = False

        if nc_ok:
            print("[OK] model.nc matches len(names).")
        else:
            print("[ERROR] model.nc does NOT match len(names).")
            print(f"        model.nc = {nc}, len(names) = {name_count}")
    else:
        print("[WARN] model.nc is None. Cannot compare nc with len(names).")

    print_section(f"[INFO] Expected class check for {title}")

    all_ok = True

    for idx, exp_name in EXPECTED_NAMES.items():
        real_name = get_name(names, idx)
        ok = real_name == exp_name
        all_ok = all_ok and ok

        flag = "OK" if ok else "ERROR"
        print(f"[{flag}] {idx:2d}: weight='{real_name}' | expected='{exp_name}'")

    print_section(f"[INFO] Key class check for {title}")
    print("Expected:")
    print("  cls=2  should be Fishing")
    print("  cls=18 should be Ship")
    print("  cls=19 should be sampan")

    print()
    print("Actual:")
    print("  cls=2 :", get_name(names, 2))
    print("  cls=18:", get_name(names, 18))
    print("  cls=19:", get_name(names, 19))

    print_section(f"[INFO] Final result for {title}")

    if all_ok and nc_ok:
        print("[OK] This model's class names and nc look correct.")
        return True

    if not all_ok:
        print("[ERROR] This model's class names do NOT match expected names.")

    if not nc_ok:
        print("[ERROR] This model's nc does NOT match len(names).")

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help=f"YOLOv7 .pt weight path, default: {default_weights}",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()

    print_section("[INFO] Environment")
    print("script root  =", root)
    print("detect_root  =", detect_root)
    print("weights_path =", weights_path)
    print("python       =", sys.executable)
    print("torch        =", torch.__version__)

    print()
    print("[INFO] sys.path[0:5]")
    for p in sys.path[:5]:
        print(" ", p)

    print_section("[INFO] YOLOv7 source check")

    models_dir = detect_root / "models"
    utils_dir = detect_root / "utils"

    print("detectModels/models exists:", models_dir.exists(), models_dir)
    print("detectModels/utils  exists:", utils_dir.exists(), utils_dir)

    if not models_dir.exists():
        print("[ERROR] detectModels/models not found.")
        print("        This checkpoint needs YOLOv7 source files to load.")
        return

    if not utils_dir.exists():
        print("[ERROR] detectModels/utils not found.")
        print("        This checkpoint needs YOLOv7 source files to load.")
        return

    print_section("[INFO] Loading weight")
    print(weights_path)

    if not weights_path.exists():
        print(f"[ERROR] Weight file not found: {weights_path}")
        return

    try:
        ckpt = torch.load(
            str(weights_path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # for older PyTorch without weights_only argument
        print("[WARN] Current PyTorch does not support weights_only argument. Retrying without it.")
        ckpt = torch.load(
            str(weights_path),
            map_location="cpu",
        )
    except ModuleNotFoundError as e:
        print("[ERROR] ModuleNotFoundError while loading checkpoint:")
        print(e)
        print()
        print("Usually this means YOLOv7 source path is not correctly added to sys.path.")
        print("Current detect_root:", detect_root)
        print("Please confirm detectModels/models and detectModels/utils exist.")
        return
    except Exception as e:
        print("[ERROR] Failed to load checkpoint:")
        print(type(e).__name__, e)
        return

    print("[INFO] Checkpoint type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("[INFO] Checkpoint keys:")
        print(list(ckpt.keys()))
    else:
        print("[WARN] Checkpoint is not a dict.")

    model_results = []

    if isinstance(ckpt, dict):
        if "model" in ckpt and ckpt["model"] is not None:
            model_results.append(
                inspect_model(
                    ckpt["model"],
                    "ckpt['model']",
                )
            )
        else:
            print("[WARN] ckpt['model'] not found or is None.")

        if "ema" in ckpt and ckpt["ema"] is not None:
            model_results.append(
                inspect_model(
                    ckpt["ema"],
                    "ckpt['ema']",
                )
            )
        else:
            print("[WARN] ckpt['ema'] not found or is None.")

        if not model_results:
            print("[WARN] No model or ema found. Trying checkpoint itself.")
            model_results.append(
                inspect_model(
                    ckpt,
                    "checkpoint itself",
                )
            )
    else:
        model_results.append(
            inspect_model(
                ckpt,
                "loaded object",
            )
        )

    print_section("[INFO] Overall result")

    if any(model_results):
        print("[OK] At least one model object has correct class names.")
        print()
        print("Next check:")
        print("  If inference still needs cls + 1 to look correct,")
        print("  the problem is probably in your inference names list,")
        print("  merge-class mapping, or drawing/display logic.")
    else:
        print("[ERROR] No inspected model object fully matched expected class names.")
        print()
        print("Possible causes:")
        print("  1. This weight was trained with a different data.yaml.")
        print("  2. The checkpoint stores old names.")
        print("  3. The model was resumed from a wrong nc/names setting.")
        print("  4. Your expected class order is different from the actual training order.")


if __name__ == "__main__":
    main()