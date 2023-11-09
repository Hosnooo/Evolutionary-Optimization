from utils import *

# Generate data for function 1
x1_data = np.linspace(0, 2, 100)  # Input data for function 1
y1_data = function_1(x1_data)     # Corresponding output data

# Generate data for function 2
x2_data = np.random.uniform(-1, 1, size=(100, 2))  # Input data for function 2
y2_data = np.array([function_2(x[0], x[1]) for x in x2_data])  # Corresponding output data

# Split data into training, validation, and test sets for function 1
x1_train, x1_test, y1_train, y1_test = train_test_split(x1_data, y1_data, test_size=0.2, random_state=42)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, test_size=0.2, random_state=42)

# Split data into training, validation, and test sets for function 2
x2_train, x2_test, y2_train, y2_test = train_test_split(x2_data, y2_data, test_size=0.2, random_state=42)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, test_size=0.2, random_state=42)