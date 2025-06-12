# EfficientNet Breast Cancer Detection

This project uses a deep learning model to classify breast ultrasound images as Normal, Benign, or Malignant.

## How to Run the Pipeline

Follow these steps from the root directory of the project.

### 1. Download the Dataset

The model is trained on the Breast Ultrasound Images Dataset.
- **Download Link:** [https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download)

After downloading, unzip the folder and place its contents into a directory named `dataset` in the root of this project. The structure should look like this:
```
/enhanced-cnn-breast-cancer-detection
|-- /dataset
|   |-- /benign
|   |-- /malignant
|   |-- /normal
|-- /src
...
```

### 2. Install Dependencies
Install all the required Python packages.
```bash
pip install -r requirements.txt
```

### 3. Prepare and Split the Dataset
This script splits the raw images from the `dataset` folder into `train`, `val`, and `test` sets, placing them in a new `data_processed` directory.
```bash
python prepare_dataset.py
```

### 4. Train the Model
Run the training script. This will train the model using the settings in `src/config.py` and save the final model to `src/models/cnn_model.pth`.
```bash
python src/train.py
```

### 5. Evaluate the Model (Optional)
To see the model's performance on the test set, run the test script.
```bash
python src/test.py
```

### 6. Run the Web Application
Launch the interactive Streamlit app to make predictions on your own images.
```bash
streamlit run streamlit/app.py
```
