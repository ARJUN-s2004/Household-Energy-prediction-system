import pandas as pd

df = pd.read_csv("code/household_power_consumption.txt", sep=";", low_memory=False, na_values="?")

import pandas as pd

# Strip column names of extra spaces
df.columns = df.columns.str.strip()

# Combine 'Date' and 'Time' into a new 'DateTime' column
df["DateTime"] = pd.to_datetime(df["Date"] + ' ' + df["Time"], format="%d/%m/%Y %H:%M:%S", errors='coerce')

# Drop the old Date and Time columns if not needed
df.drop(columns=["Date", "Time"], inplace=True)

# Convert numeric columns from object to float
cols_to_convert = df.columns.difference(["DateTime"])  # Exclude DateTime from conversion
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Show the updated data types
print(df.dtypes)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Define Features and Target ---
# Drop rows with NaN (if not already done)
df.dropna(inplace=True)

# Target variable
y = df['Global_active_power']

# Features (drop target and DateTime)
X = df[['Global_reactive_power', 'Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']]

# --- Step 2: Split into Train & Test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train Linear Regression ---
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred = lr_pipeline.predict(X_test)

# --- Step 4: Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# Preprocessing and feature selection
# Model creation for predicting bmi using preg,bp,st
x = df[['Global_reactive_power', 'Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']] #features
y = df['Global_active_power'] #target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50) # Test_size = 20% for testing and 80% for training

# Model training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train) # training xtrain and ytrain
print(model.coef_)
print(model.intercept_)

# Model evaluation using testing split
from sklearn import metrics
import numpy as np
y_pred=model.predict(x_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Prepare Features & Target ---
x = df[['Global_reactive_power', 'Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']] #features
y = df['Global_active_power'] 

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Simple Random Forest Model ---
rf = RandomForestRegressor(n_estimators=5,max_depth=10,random_state=42,n_jobs=-1)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# --- 4. Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Fast Random Forest:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Prepare Features & Target ---
x = df[['Global_reactive_power', 'Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']] #features
y = df['Global_active_power'] 

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Gradient Boosting Regressor (Lightweight Settings) ---
gbr = GradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=3, random_state=42)

gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

# --- 4. Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🚀 Gradient Boosting Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. Prepare Features & Target ---
x = df[['Global_reactive_power', 'Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']] #features
y = df['Global_active_power'] 

# --- 2. Split and Scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Convert to Tensors ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# --- 4. Create DataLoader ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- 5. Define Model ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

model = Net()

# --- 6. Training Setup ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 7. Train ---
for epoch in range(20):  # small number of epochs to keep it fast
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# --- 8. Predict & Evaluate ---
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n🧠 PyTorch Neural Net Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
