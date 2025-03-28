import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = np.array(data_dict['labels'], dtype=np.int32)

# Ensure all samples have the same length (Padding or Truncation)
fixed_length = 42  # 21 hand landmarks * 2 (x, y)
for i in range(len(data)):
    if len(data[i]) < fixed_length:
        data[i] += [0] * (fixed_length - len(data[i]))  # Padding
    else:
        data[i] = data[i][:fixed_length]  # Truncation

# Convert to NumPy array
data = np.array(data, dtype=np.float32)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize RandomForest with class balancing
model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

# Train model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
score = accuracy_score(y_pred, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully! ✅")
