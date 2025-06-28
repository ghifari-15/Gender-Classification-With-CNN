# Gender Classification with CNN

A deep learning application for classifying gender (male/female) from face images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.  
Supports both batch image prediction and real-time webcam detection via a simple web app.

---

## Features
- Image preprocessing and normalization
- Data augmentation for robust training
- CNN model for binary classification (male/female)
- Early stopping to prevent overfitting
- Real-time prediction support (webcam & web app)
- User-friendly mini app (upload image & real-time camera)

---

## Project Structure
```
Gender_Classification/
├── data/
│   ├── Training/
│   │   ├── male/
│   │   └── female/
│   └── Validation/
│       ├── male/
│       └── female/
├── test/                       # Test images for inference
├── my_model.h5                 # Saved trained model
├── index.ipynb                 # Main Jupyter notebook (training & evaluation)
├── app.py                      # Streamlit mini app (upload & real-time camera)
├── realtime_gender_classification.py # Standalone real-time webcam script
├── readme.md                   # Project documentation
```

---

## How to Run

### 1. Prepare Data
- Place training and validation images in the respective folders as shown above.
- Images should be face photos, ideally in JPG/PNG format.

### 2. Train the Model
- Open `index.ipynb` in Jupyter or VS Code.
- Run all cells to preprocess data, train, and evaluate the model.
- The trained model will be saved as `my_model.h5`.

### 3. Test the Model (Batch Prediction)
- Place test images in the `test/` folder.
- Run the prediction cells in the notebook to see gender classification results.

### 4. Real-Time Prediction (Standalone Script)
- Run `realtime_gender_classification.py` for real-time gender detection from your webcam:
  ```
  python realtime_gender_classification.py
  ```

### 5. Web App (Upload & Real-Time Camera)
- Install Streamlit and dependencies:
  ```
  pip install streamlit streamlit-webrtc av tensorflow opencv-python numpy matplotlib pillow
  ```
- Run the app:
  ```
  streamlit run app.py
  ```
- Use the sidebar to choose between uploading images or using the real-time camera.

---

## Requirements
- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Streamlit
- streamlit-webrtc
- av

---

## Notes
- All images are normalized to 0-1 before training/prediction.
- Data augmentation is used to improve generalization.
- Early stopping is applied to avoid overfitting.
- For real-time camera in the web app, make sure your browser allows webcam access.

---

## Example Prediction Output
```
Image 1: Female. Confidence: 87.23%
Image 2: Male. Confidence: 91.12%
```
Or, in real-time mode, results and bounding boxes will appear live on your webcam feed.

---

![image](https://github.com/user-attachments/assets/501aacfb-8b4c-4af5-9f93-59e448642c15)



**Author:** Gathan Ghifari Rachwiyono
