# Basic Captcha Recognition

This is a simple text based CAPTCHA Recognition project where we build, train and experiment with different types of Convolutional Recurrent Neural Network (CRNN) models and report accuracy across different metrics. A CRNN model with bidirectional GRU layers and Attention mechanism gave the best performance of 99.5% cosine similarity on the test set. A detailed information about different models experimented with and accuracy metrics are given in the [paper](https://github.com/harikiran17/BasicCaptchaRecognition/blob/main/report/Captcha_recognition_report.pdf).

# Dataset

The dataset used is a simple [kaggle dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images) consisting of different CAPTCHA images, where each image is a combination of 5 random alphanumeric characters. 90% of the data was used for training, while the rest 10% was used for testing and reporting accuracies. A sample of the dataset is given below:

![sample training set](https://github.com/harikiran17/BasicCaptchaRecognition/blob/main/imgs/training_sample.png)

# Training

In [train_models.py](https://github.com/harikiran17/BasicCaptchaRecognition/blob/main/train_models.py), in the main() function, edit the following paths to your respective locations of the dataset and the locations where you want to save the model to start training.
```
input_images_path = "./captcha_1_train/"
ckpt_path = "./captcha_models/ckpt_crnn_full_captcha_1_32_200_50_gray_gru_attn"
logs_path = "./captcha_models_logs/logs_crnn_full_captcha_1_32_200_50_gray_gru_attn"
model_save_path = "./captcha_models/model_crnn_full_captcha_1_32_200_50_gray_gru_attn"
```
CRNN [model](https://github.com/harikiran17/BasicCaptchaRecognition/tree/main/models) with GRU and attention mechanishm was trained for 200 epochs. The graph below shows epochs vs CTC loss.

![epochs vs CTC loss](https://github.com/harikiran17/BasicCaptchaRecognition/blob/main/imgs/model5.PNG)

