# Gender Classification with CNN

This project is a deep learning application for classifying gender (male/female) from face images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Features
- Image preprocessing and normalization
- Data augmentation for robust training
- CNN model for binary classification
- Early stopping to prevent overfitting
- Real-time prediction support (webcam)

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
├── test/           # Test images for inference
├── my_model.h5     # Saved trained model
├── index.ipynb     # Main Jupyter notebook
├── readme.md       # Project documentation
```

## How to Run
1. **Prepare Data:**
   - Place training and validation images in the respective folders as shown above.
   - Images should be face photos, ideally in JPG/PNG format.

2. **Train the Model:**
   - Open `index.ipynb` in Jupyter or VS Code.
   - Run all cells to preprocess data, train, and evaluate the model.

3. **Test the Model:**
   - Place test images in the `test/` folder.
   - Run the prediction cells to see gender classification results.

4. **Real-Time Prediction (Optional):**
   - Use the provided script (see notebook) to run real-time gender prediction from webcam.

## Requirements
- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

Install dependencies:
```
pip install tensorflow opencv-python numpy matplotlib
```

## Notes
- Normalize all images to 0-1 before training/prediction.
- Data augmentation is used to improve generalization.
- Early stopping is applied to avoid overfitting.

## Example Prediction Output
```
Image 1: Female. Confidence: 87.23%
Image 2: Male. Confidence: 91.12%
```

---

**Author:** Gathan Ghifari Rachwiyono
