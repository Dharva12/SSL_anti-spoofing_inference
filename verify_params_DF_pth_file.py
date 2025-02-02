import torch
from model import Model

# Step 1: Load the model file
model_path = "C:/Users/Legion 5I 72IN/Desktop/SUMMER internship/SSL/SSL_Anti-spoofing/Best_LA_model_for_DF.pth"
state_dict = torch.load(model_path, map_location='cpu')

print("\nKeys in the state dictionary:")
model_file_keys = set(state_dict.keys())
for key in model_file_keys:
    print(key)

# Step 2: Initialize the Model class and extract its expected keys
args = None  # Replace with required arguments if necessary
device = "cpu"  # Use "cuda" if you want to run on a GPU
model = Model(args, device)

expected_model_keys = set(model.state_dict().keys())

print("\nKeys expected by the model:")
for key in expected_model_keys:
    print(key)

# Step 3: Compare the keys
missing_keys = expected_model_keys - model_file_keys
unexpected_keys = model_file_keys - expected_model_keys

print("\nMissing keys (expected but not found in the state dictionary):")
print(missing_keys)

print("\nUnexpected keys (found in the state dictionary but not expected by the model):")
print(unexpected_keys)

# Step 4: Handle discrepancies
if unexpected_keys:
    print("\nFixing unexpected keys...")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    torch.save(new_state_dict, "Best_LA_model_for_DF_cleaned.pth")
    print("Unexpected keys fixed. Saved cleaned state dictionary as 'Best_LA_model_for_DF_cleaned.pth'.")
