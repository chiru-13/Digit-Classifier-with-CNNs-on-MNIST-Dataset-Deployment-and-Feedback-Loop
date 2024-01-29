import tensorflow as tf
import numpy as np
import os
import pandas as pd
# Function for fine-tuning
def fine_tune_model(incorrect_predictions):
    model = tf.keras.models.load_model('model.h5')
    if incorrect_predictions is not None:
        incorrect_data_paths = incorrect_predictions['Image_Path'].tolist()
        incorrect_labels = incorrect_predictions['Actual_Label'].tolist()
        incorrect_images = load_data_from_paths(incorrect_data_paths)
        incorrect_images = preprocess_data(incorrect_images)
        incorrect_images = np.squeeze(incorrect_images, axis=1)
        incorrect_images = np.expand_dims(incorrect_images, axis=-1)
        incorrect_labels = tf.keras.utils.to_categorical(incorrect_labels, num_classes=10)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(incorrect_images, incorrect_labels, epochs=5)  

        model.save('model.h5')

# Function to evaluate the fine-tuned model
def evaluate_fine_tuned_model():
    train_data = pd.read_csv('train.csv')
    test_data = train_data.sample(frac=0.2, random_state=42)
    test_labels = test_data['label'].tolist()
    test_pixels = test_data.iloc[:, 1:].values 
    test_images = test_pixels.reshape(-1, 28, 28, 1)
    test_images = preprocess_data(test_images)
    test_images = np.squeeze(test_images, axis=1)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    fine_tuned_model = tf.keras.models.load_model('model.h5')
    evaluation_result = fine_tuned_model.evaluate(test_images, test_labels)
    print("Fine-tuned Model Evaluation:", evaluation_result)

# Function to load images and labels from paths
def load_data_from_paths(paths):
    images = []
    for path in paths:
        img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Function to preprocess and predict image
def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.mean(img_array, axis=-1, keepdims=True) if img_array.shape[-1] == 3 else img_array
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = tf.image.resize(img_array, (28, 28))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to preprocess a batch of images
def preprocess_data(images):
    preprocessed_images = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
    preprocessed_images = np.array(preprocessed_images)
    preprocessed_images = tf.squeeze(preprocessed_images, axis=-1)
    return  preprocessed_images

# Function to retrieve incorrect predictions
def retrieve_incorrect_predictions():
    incorrect_predictions_file = 'incorrect_predictions.csv'
    image_folder = 'incorrect_images'
    if os.path.exists(incorrect_predictions_file):
        incorrect_data = pd.read_csv(incorrect_predictions_file)
        incorrect_data['Image_Path'] = [os.path.join(image_folder, f'incorrect_image_{i+1}.png') for i in range(len(incorrect_data))]
        return incorrect_data
    else:
        return None

if __name__ == "__main__":
    incorrect_predictions = retrieve_incorrect_predictions()
    if incorrect_predictions is not None:
        fine_tune_model(incorrect_predictions)
    evaluate_fine_tuned_model()
