import argparse
import os
import torch
from torch.utils.data import DataLoader
from model import Model
from data_utils_SSL import Dataset_SimpleTest


def produce_evaluation_file(dataset, model, device, save_path):
    """
    Runs inference on the dataset and saves evaluation scores to a file.
    Args:
        dataset: PyTorch Dataset containing the test data.
        model: Loaded PyTorch model for inference.
        device: Device to run inference on ('cuda' or 'cpu').
        save_path: Path to save the evaluation scores.
    """
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()

    fname_list = []
    score_list = []
    
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():  # Disable gradients for inference
            batch_out = model(batch_x)  # Raw logits or probabilities

        # Print or save raw logits/probabilities for inspection
        print("Raw batch_out Tensor:", batch_out)

        batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    with open(save_path, 'w') as fh:
        for f, cm in zip(fname_list, score_list):
            fh.write(f"{f} {cm}\n")
    print(f"Scores saved to {save_path}")



if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='ASVspoof2021 Pretrained Model Inference')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to the folder containing test files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--eval_output', type=str, required=True, help='Path to save evaluation results')

    args = parser.parse_args()

    # Check if test folder exists
    if not os.path.isdir(args.test_folder):
        raise FileNotFoundError(f"Test folder not found: {args.test_folder}")

    # Check if model path exists
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model = Model(args, device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded: {args.model_path}")
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model. Ensure the architecture matches the checkpoint.\n{e}")
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_set = Dataset_SimpleTest(test_folder=args.test_folder)
    print(f"Test dataset loaded from {args.test_folder}")

    # Run inference
    produce_evaluation_file(test_set, model, device, args.eval_output)
