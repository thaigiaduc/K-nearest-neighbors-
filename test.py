print("hello world");
train_data = [
    {'features': [1400, 3], 'label': 500},
    {'features': [1600, 3], 'label': 550},
    {'features': [1700, 2], 'label': 600},
    {'features': [1875, 3], 'label': 625},
    {'features': [1100, 2], 'label': 475},
    {'features': [1550, 4], 'label': 612.5},
    {'features': [2350, 4], 'label': 830},
    {'features': [2450, 5], 'label': 850},
    {'features': [1425, 3], 'label': 550},
    {'features': [1700, 4], 'label': 625},
]

# Tìm k điểm dữ liệu gần nhất
def find_k_nearest_neighbors(data, new_point, k):
    # Tính khoảng cách giữa new_point và tất cả các điểm dữ liệu trong tập data
    distances = []
    for item in data:
        distance = ((item['features'][0] - new_point['features'][0])**2 + (item['features'][1] - new_point['features'][1])**2)**0.5
        distances.append({'distance': distance, 'label': item['label']})

    # Sắp xếp khoảng cách tăng dần và chọn k điểm dữ liệu gần nhất
    sorted_distances = sorted(distances, key=lambda x: x['distance'])
    neighbors = [item['label'] for item in sorted_distances[:k]]
    return neighbors

# Hàm dự đoán giá nhà bằng thuật toán KNN
def predict_price(data, new_point, k):
    neighbors = find_k_nearest_neighbors(data, new_point, k)

    # Tính giá trung bình của các điểm dữ liệu gần nhất
    price = sum(neighbors) / len(neighbors)
    return price

# Dữ liệu mới cần dự đoán
new_point = {'features': [1500, 3]}

# Dự đoán giá nhà với k = 3
price = predict_price(train_data, new_point, 3)
print(f'Predicted price: {price}')