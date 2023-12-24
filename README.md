
# Object-Detection


This is a Streamlit web application that uses a pre-trained Faster R-CNN model to detect and draw bounding boxes around people in images.



## Setup
### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/object-detection.git
```
### 2.Navigate to the Project Directory
Change your current directory to the project directory
```bash
cd object-detection
```
### 3. Install Dependencies
Make sure you have Python installed. Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```
### 4.Run the App
To run the app, execute the following command in your terminal:
``` bash
streamlit run main.py
```
## Usage
   1.Visit the deployed Streamlit app or run it locally.

   2.Upload an image using the provided file uploader.

   3.The app will display the uploaded image with bounding boxes
     drawn around detected people.
## Dependencies
    Streamlit
    Pillow
    NumPy
    Matplotlib
    PyTorch
    Torchvision
## Model

The app uses a Faster R-CNN model with a ResNet50 backbone for person detection. The model is pre-trained and comes with default weights.
