# ASL-Recognition

This project is an **American Sign Language (ASL) recognition system** built using deep learning.  
It can recognize ASL hand signs in **real-time** using a webcam via **OpenCV** and a pre-trained Keras model.

## Features
- Detects and recognizes ASL hand signs from live camera feed
- Uses a pre-trained deep learning model 
- Integrates **OpenCV** for real-time video capture
- Predicts letters from `A` to `Z` 
  
## Model Details
- **Framework:** TensorFlow / Keras  
- **Input Shape:** `(128, 128, 3)`  
- **Output:** Softmax probabilities for each ASL alphabet class  
- **Dataset:** [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
- **My Kaggle Notebook:** [ASL Recognition Project](https://www.kaggle.com/code/notsu66/asl-recognition-project/notebook)  
- **Training Data:** Images of ASL alphabets from `A` to `Z`  
- **Purpose:** Real-time ASL letter recognition using webcam input  

## ⚙️ Installation & Setup

Follow these steps to set up and run the ASL Recognition project locally:

	1. Clone the Repository
	2. Create a virtual environment
	3. Activate the virtual environment
	4. Install dependencies
	5. Create a .env file to add model path
	5. Run the Script
After that, the webcam will be opened and the model will start recognizing ASL letters in real-time.
