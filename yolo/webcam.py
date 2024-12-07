import cv2
import numpy as np

# Загрузка классов из файла coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Загрузка конфигурации и весов YOLOv4
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# Задание слоев для обработки
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Инициализация видеозахвата (камера 0 или путь к видеофайлу)
cap = cv2.VideoCapture(0)

while True:
    # Чтение каждого кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Получаем размеры кадра
    height, width, _ = frame.shape

    # Преобразование изображения в формат, пригодный для подачи в сеть
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Прямой проход через сеть
    outs = net.forward(output_layers)

    # Списки для хранения найденных объектов
    class_ids = []
    confidences = []
    boxes = []

    # Проход по результатам сети
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Порог уверенности
                # Координаты центра объекта
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Преобразуем в координаты верхнего левого угла
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Сохраняем данные об объекте
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем NMS для уменьшения количества перекрывающихся объектов
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображаем результаты
    for i in range(len(boxes)):
        if i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Зеленый цвет для рамки
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Отображаем кадр с использованием OpenCV
    cv2.imshow("YOLOv4 Object Detection", frame)

    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка
cap.release()
cv2.destroyAllWindows()  # Закрытие всех окон OpenCV
