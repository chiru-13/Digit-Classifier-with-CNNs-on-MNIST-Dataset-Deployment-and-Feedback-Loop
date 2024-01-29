# Digit Classifier with CNNs on MNIST Dataset: Deployment and Feedback Loop

This assignment submission includes a Streamlit web application for classifying handwritten digits using a pre-trained convolutional neural network (CNN) model on the MNIST dataset. The application allows users to upload images for classification, provide feedback, and trigger immediate fine-tuning. Additionally, a system-scheduled job runs periodically to fine-tune the model based on user feedback, and the dynamically updated model replaces the initial pre-trained model (`model.h5`).

## Folder Contents:

- **app.py**: The Streamlit web application code that includes image classification, user feedback, and options for immediate model fine-tuning.
- **finetuning.py**: Module containing functions for scheduled fine-tuning of the pre-trained model based on user feedback.
- **model.h5**: The dynamically updated model, initially pre-trained and later fine-tuned based on user feedback.
- **train.csv**: CSV file containing training data for model evaluation during fine-tuning.
- **num.png**: Image displayed on the Streamlit app interface.
- **modelbuilding.ipynb**: Jupyter Notebook containing the code used to build the initial pre-trained model.
- **requirements.txt**:The file lists necessary Python packages required to run the model and the Streamlit app

## Instructions

1. **Environment Setup**: Ensure you have the required Python packages installed. You can install them using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Model Building**: Explore the Jupyter Notebook `modelbuilding.ipynb` to understand the process of building the initial pre-trained model (`model.h5`). Execute the notebook if you wish to retrain or modify the model architecture.

3. **Run the App**: Execute the Streamlit app by running the following command:

   ```bash
   streamlit run app.py
   ```

4. **Usage**: Visit the provided URL(after running the app.py file)in your web browser. Upload images, provide feedback with the help of the feedback interface to indicate the correctness of model predictions and if it is incorrectly predicted then the enter actual value which will automatically stored in the "incorrect_images" folder and incorrect_predictions.csv which will be generated automatically and also explore options for immediate fine-tuning.

5. **Immediate Fine-tuning**: Users have the option to trigger immediate fine-tuning by clicking the "Run  Fine-tuning" button on the app interface. The fine-tuned model will replace the initial pre-trained model (`model.h5`), ensuring continuous improvement.

6. **System-Scheduled Fine-tuning**:
For this part I have automated the fine-tuning process using a cron job on my laptop. With the scheduled cron job, the finetuning.py script will run automatically at the specified intervals, retrieving the stored incorrect prediction, using the weights of the previous model and train them with the incoorect predictions, and fine-tuning the model weights accordingly. Consequently, the the model (`model.h5`) is dynamically updated.
## Notes

- The app uses the model (`model.h5`) for predictions which will be is dynamically updated for digit classification based on user feedback.
- Incorrect predictions are stored in the `incorrect_images/` folder, and details are logged in `incorrect_predictions.csv`. This files are automatically by the app.py code.
- The model is evaluated on a subset of the training data (`train.csv`) during fine-tuning.
