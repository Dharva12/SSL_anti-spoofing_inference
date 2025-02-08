# SSL_anti-spoofing_inference

link to the original Repo:- https://github.com/TakHemlata/SSL_Anti-spoofing?tab=readme-ov-file

## Inference pipeline for replication

Make sure to replace paths to relevant files/folders in the code files, the inference snippet and where required.

### 1. Setup Instructions
#### 1.1 Clone the Repository

####  1.2 Install Dependencies

pip install -r requirements.txt

#### 1.3 Download Pretrained Models
The required models can be downloaded from Google Drive:

https://drive.google.com/drive/folders/1izeSTiCesyJTPttOYTbmRtPw24o6zjSr?usp=sharing

## Model Checkpoints & Feature Extractor

This Google drive link contains two key files required for testing the pretrained anti-spoofing model. 

#### a. Best_LA_model_for_DF_cleaned.pth

What is this file?

This is a pretrained model checkpoint trained for Logical Access (LA) attacks but adapted for DeepFake (DF) testing.
It is part of the SSL Anti-Spoofing model pipeline.
The original file had key mismatches when loading into our testing script, so it was cleaned to ensure compatibility.

How was it cleaned?

The prefix "module." was removed from certain model keys to fix mismatches.
The cleaned model was saved as Best_LA_model_for_DF_cleaned.pth.

How to Use it?

The inference script automatically loads this checkpoint when running tests.

#### b. xlsr2_300m.pt

What is this file?

This is a pretrained Wav2Vec 2.0 XLSR-300M model.
XLSR (Cross-Lingual Speech Representations) is a self-supervised feature extractor trained on multilingual speech data.
This file is likely used to extract deep audio embeddings before passing them through the anti-spoofing classifier.

How to Use it?

It will be loaded in the SSL Anti-Spoofing model pipeline as a feature extractor.
If the model doesn't detect it, manually specify its path in inference.py.

### 2. Expected Input Format

The test dataset should contain WAV audio files.

### 3. Running Inference
Once the setup is complete, run inference on your test dataset.

#### 3.1 Logical Access (LA) Model Inference

python main_SSL_LA.py --test_folder "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\SSL_test_interpretation_check" --model_path "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\SSL_Anti-spoofing\LA_model.pth" --eval_output "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\output_scores_LA.txt"

What this does:

Loads the LA model from models/LA_model.pth

Runs inference on the test samples

Saves results in results/scores_LA.txt

#### 3.2 DeepFake (DF) Model Inference

python main_SSL_DF.py --test_folder "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\SSL_test" --model_path "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\SSL_Anti-spoofing\Best_LA_model_for_DF_cleaned.pth" --eval_output "C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\output_scores_DF.txt"

What this does:

Loads the DF model from models/Best_LA_model_for_DF_cleaned.pth

Runs inference on the test samples

Saves results in results/scores_DF.txt

### 4. Understanding the Output Scores

The model generates a confidence score for each input file in the output file (results/scores_LA.txt or results/scores_DF.txt).

#### 4.1 How to Interpret the Scores

Score	Prediction

Negative (< 0)	Likely Bonafide (Genuine Speech)

Positive (≥ 0)	Likely Spoofed (Fake Speech)

Higher magnitude (e.g., -5.4 or +5.4) indicates stronger confidence.

### 5. Evaluating Model Performance
   
You can compute the Equal Error Rate (EER) and t-DCF to set a proper threshold.
