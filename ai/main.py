import sqlite3
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder

# === Шаг 1: Подключение к базе данных и извлечение данных ===
def load_data_from_db(db_path):
    # Подключение к базе данных
    conn = sqlite3.connect(db_path)
    query = "SELECT street, event, resolution_code FROM incidents"  # Настройте запрос в зависимости от структуры вашей БД
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# === Шаг 2: Подготовка данных ===
class TrafficDataset(Dataset):
    def __init__(self, data):
        # One-hot encoding для улиц и происшествий
        self.street_encoder = OneHotEncoder(sparse=False)
        self.event_encoder = OneHotEncoder(sparse=False)

        self.street_encoded = self.street_encoder.fit_transform(data[['street']])
        self.event_encoded = self.event_encoder.fit_transform(data[['event']])
        
        self.labels = data['resolution_code'].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        street_features = self.street_encoded[idx]
        event_features = self.event_encoded[idx]
        combined_features = torch.tensor(
            street_features.tolist() + event_features.tolist(), dtype=torch.float32
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return combined_features, label

    def get_input_size(self):
        return self.street_encoded.shape[1] + self.event_encoded.shape[1]

# === Шаг 3: Архитектура нейросети ===
class TrafficResolutionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(TrafficResolutionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === Шаг 4: Обучение нейросети ===
def train_model(data, epochs=20, batch_size=16):
    dataset = TrafficDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = dataset.get_input_size()
    model = TrafficResolutionNet(input_size=input_size, output_size=3)  # 3 класса на выходе
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Эпоха {epoch+1}, Потери: {loss.item():.4f}")
    
    return model, dataset

# === Шаг 5: Система рекомендаций ===
def recommend_resolution(model, dataset, street, event):
    model.eval()
    with torch.no_grad():
        # Преобразование входных данных
        street_encoded = dataset.street_encoder.transform([[street]])[0]
        event_encoded = dataset.event_encoder.transform([[event]])[0]
        input_data = torch.tensor(
            street_encoded.tolist() + event_encoded.tolist(), dtype=torch.float32
        )
        prediction = model(input_data.unsqueeze(0))
        resolution_code = torch.argmax(prediction).item()
        
        recommendations = {
            0: "Увеличить длительность светофоров.",
            1: "Сообщить экстренным службам.",
            2: "Направить бригаду для устранения препятствия.",
        }
        return recommendations[resolution_code]

# === Шаг 6: Запуск и пример использования ===
if __name__ == "__main__":
    # Загрузка данных из базы данных
    db_path = "incident.db"
    data = load_data_from_db(db_path)

    # Обучение модели
    trained_model, dataset = train_model(data)

    # Пример входных данных: улица Ленина, происшествие — авария
    example_street = "Ленина"
    example_event = "Авария"
    recommendation = recommend_resolution(trained_model, dataset, example_street, example_event)
    print(f"Рекомендация: {recommendation}")
