 SSL_anti-spoofing_inference

link to the original Repo:- https://github.com/TakHemlata/SSL_Anti-spoofing?tab=readme-ov-file

Inference pipeline for replication


Model Checkpoints & Feature Extractor

https://drive.google.com/drive/folders/1izeSTiCesyJTPttOYTbmRtPw24o6zjSr?usp=sharing

This Google drive link contains two key files required for testing the pretrained anti-spoofing model. 

1. Best_LA_model_for_DF_cleaned.pth

What is this file?

This is a pretrained model checkpoint trained for Logical Access (LA) attacks but adapted for DeepFake (DF) testing.
It is part of the SSL Anti-Spoofing model pipeline.
The original file had key mismatches when loading into our testing script, so it was cleaned to ensure compatibility.

How was it cleaned?

The prefix "module." was removed from certain model keys to fix mismatches.
The cleaned model was saved as Best_LA_model_for_DF_cleaned.pth.

How to Use it?

The inference script automatically loads this checkpoint when running tests.

2. xlsr2_300m.pt

What is this file?

This is a pretrained Wav2Vec 2.0 XLSR-300M model.
XLSR (Cross-Lingual Speech Representations) is a self-supervised feature extractor trained on multilingual speech data.
This file is likely used to extract deep audio embeddings before passing them through the anti-spoofing classifier.

How to Use it?

It will be loaded in the SSL Anti-Spoofing model pipeline as a feature extractor.
If the model doesn't detect it, manually specify its path in inference.py.


