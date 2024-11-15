import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# ------------------ Load Data ------------------
iris_df = pd.read_csv('iris.csv')
X_iris = iris_df.drop(columns=['Id', 'Species']).values
y_iris = iris_df['Species'].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_iris = label_encoder.fit_transform(y_iris)

# ------------------ KNN ------------------
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Measure KNN time
start_time_knn = time.time()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_predictions = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_time = time.time() - start_time_knn  # Time taken for KNN

# ------------------ SVM ------------------
start_time_svm = time.time()
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
svm_predictions = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_time = time.time() - start_time_svm  # Time taken for SVM

# ------------------ ANN ------------------
class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        self.learning_rate = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

y_iris_one_hot = np.zeros((y_iris.shape[0], 3))
y_iris_one_hot[np.arange(y_iris.shape[0]), y_iris] = 1

X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X_iris, y_iris_one_hot, test_size=0.3, random_state=42)
X_train_ann_scaled = scaler.fit_transform(X_train_ann)
X_test_ann_scaled = scaler.transform(X_test_ann)

# Measure ANN time
start_time_ann = time.time()
ann = ANN(input_size=4, hidden_size=5, output_size=3)
ann.train(X_train_ann_scaled, y_train_ann, epochs=10000)
ann_predictions = ann.predict(X_test_ann_scaled)
ann_accuracy = accuracy_score(np.argmax(y_test_ann, axis=1), ann_predictions)
ann_time = time.time() - start_time_ann  # Time taken for ANN

# ------------------ User Input ------------------

def predict_class(input_data):
    # Standardize the input data
    input_scaled = scaler.transform([input_data])
    
    # KNN prediction and accuracy
    knn_prediction = knn.predict(input_scaled)
    knn_predicted_class = label_encoder.inverse_transform(knn_prediction)
    
    # SVM prediction and accuracy
    svm_prediction = svm.predict(input_scaled)
    svm_predicted_class = label_encoder.inverse_transform(svm_prediction)
    
    # ANN prediction and accuracy
    ann_prediction = ann.predict(input_scaled)
    ann_predicted_class = label_encoder.inverse_transform(ann_prediction)

    return knn_predicted_class[0], svm_predicted_class[0], ann_predicted_class[0]

# ------------------ Calculate Accuracy for ANN ------------------

def get_user_input():
    # Nhập dữ liệu từ người dùng
    print("Enter flower features (sepal length, sepal width, petal length, petal width):")
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))
    
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Dự đoán cho các mô hình
    knn_class, svm_class, ann_class = predict_class(input_data)
    
    # In kết quả ra màn hình
    print(f"\nKNN predicted class: {knn_class}, Accuracy: {knn_accuracy:.4f}, Time: {knn_time:.4f} seconds")
    print(f"SVM predicted class: {svm_class}, Accuracy: {svm_accuracy:.4f}, Time: {svm_time:.4f} seconds")
    print(f"ANN predicted class: {ann_class}, Accuracy: {ann_accuracy:.4f}, Time: {ann_time:.4f} seconds")
    
    # Lưu kết quả vào file ketqua.txt
    with open('ketqua_vs_du_lieu_so.txt', 'a') as file:
        file.write(f"Input: {input_data}\n")
        file.write(f"KNN predicted class: {knn_class}, Accuracy: {knn_accuracy:.4f}, Time: {knn_time:.4f} seconds\n")
        file.write(f"SVM predicted class: {svm_class}, Accuracy: {svm_accuracy:.4f}, Time: {svm_time:.4f} seconds\n")
        file.write(f"ANN predicted class: {ann_class}, Accuracy: {ann_accuracy:.4f}, Time: {ann_time:.4f} seconds\n")
        file.write("\n" + "="*50 + "\n")  # Dấu phân cách giữa các lần dự đoán

get_user_input()
