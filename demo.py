import cv2
from shipDetects import MainDetects
import os


if __name__ == '__main__':
    yoClass = MainDetects(
        weights="/Users/chou/Desktop/python/shipDetectspy_V1.0.5/pt/best_0511.pt",
        conf_thres=0.55,
        iou_thres=0.5,
        camIndex=1)

    cap = cv2.VideoCapture("/Users/chou/Desktop/python/shipDetectspy_V1.0.5/cam2.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locations = yoClass.run(frame)

        for obj in locations:
            x1 = int(obj["x"] - obj["w"] / 2)
            y1 = int(obj["y"] - obj["h"] / 2)
            x2 = int(obj["x"] + obj["w"] / 2)
            y2 = int(obj["y"] + obj["h"] / 2)

            cls_name = obj["class"]
            conf = obj.get("conf", 0.0)

            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()