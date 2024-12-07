import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import csv

# Load YOLO model and classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

translations = {
    "person": "Человек",
    "bicycle": "Велосипед",
    "car": "Машина",
    "motorbike": "Мотоцикл",
    "bus": "Автобус",
    "truck": "Грузовик",
    "tree": "Дерево",
}

net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Video path and output log file
video_path = "video.mp4"
log_file = "events_log.csv"

# Open video and prepare output
cap = cv2.VideoCapture(video_path)
resize_width, resize_height = 640, 360

# Initialize buffers and thresholds
object_positions = {}
stationary_objects = {}  # Tracks when objects became stationary
stationary_time_threshold = 10  # 10 seconds stationary to trigger an event
traffic_jam_threshold = 5
accident_threshold = 2
last_event_times = {"traffic_jam": 0, "accident": 0}
cooldown = 10  # Seconds to suppress duplicate events

# Initialize logging
with open(log_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Event", "Description"])

# Helper function: Draw Russian text
def draw_russian_text(image, text, position, font_size=20, color=(0, 255, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Update stationary objects
def update_stationary_objects(class_ids, boxes, classes, frame, current_time):
    global object_positions, stationary_objects
    current_frame_objects = {}

    for i, box in enumerate(boxes):
        x, y, w, h = box
        class_name = classes[class_ids[i]]
        if class_name in ["car", "bus", "truck"]:
            object_id = f"{class_name}_{x}_{y}_{w}_{h}"
            current_frame_objects[object_id] = (x, y, w, h)

            if object_id in object_positions:
                prev_x, prev_y, _, _ = object_positions[object_id]["position"]
                if abs(prev_x - x) < 5 and abs(prev_y - y) < 5:
                    if object_id not in stationary_objects:
                        stationary_objects[object_id] = current_time  # Mark stationary start time
                else:
                    stationary_objects.pop(object_id, None)

            object_positions[object_id] = {"position": (x, y, w, h), "time": current_time}

    object_positions = {
        key: value
        for key, value in object_positions.items()
        if current_frame_objects.get(key) is not None
    }

# Log event
def log_event(event_type, description):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, description])

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Видео завершено.")
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    filtered_boxes = [boxes[i] for i in indexes.flatten()]
    filtered_class_ids = [class_ids[i] for i in indexes.flatten()]

    current_time = time.time()
    update_stationary_objects(filtered_class_ids, filtered_boxes, classes, frame, current_time)

    # Detect traffic jam
    stationary_cars = [
        key
        for key, start_time in stationary_objects.items()
        if current_time - start_time >= stationary_time_threshold
    ]

    if len(stationary_cars) >= traffic_jam_threshold and current_time - last_event_times["traffic_jam"] > cooldown:
        draw_russian_text(frame, "Пробка: машины стоят!", (10, 30), font_size=24, color=(0, 0, 255))
        log_event("Traffic Jam", "Машины не двигаются.")
        last_event_times["traffic_jam"] = current_time

    # Detect accident
    if len(stationary_cars) >= accident_threshold and current_time - last_event_times["accident"] > cooldown:
        draw_russian_text(frame, "Авария: повреждены машины!", (10, 60), font_size=24, color=(255, 0, 0))
        log_event("Accident", "Замечены неподвижные повреждённые машины.")
        last_event_times["accident"] = current_time

    for i in range(len(filtered_boxes)):
        x, y, w, h = filtered_boxes[i]
        label = translations.get(classes[filtered_class_ids[i]], classes[filtered_class_ids[i]])
        frame = draw_russian_text(frame, label, (x, y - 10))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
