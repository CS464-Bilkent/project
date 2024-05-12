from sklearn.model_selection import GridSearchCV
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imutils import paths

def import_images_data(samples_directory):
    image_pixels = []
    image_labels = []

    for i, image_path in enumerate(samples_directory):
        sample_image = cv2.imread(image_path)
        sample_label = image_path.split(os.path.sep)[-2]

        sample_image = cv2.resize(sample_image, (32, 32), interpolation=cv2.INTER_AREA)

        image_pixels.append(sample_image)
        image_labels.append(sample_label)

    return (np.array(image_pixels), np.array(image_labels))

# Dosyaların bulunduğu dizin
path_to_sample_images = 'raw-img'


param_grid = {
    'C': [0,1,1,10],  # Penalty parameter C
    'kernel': ['linear'] ,  # Kernel type
}


samples_directory = list(paths.list_images(path_to_sample_images))

samples_directory_half = samples_directory[:len(samples_directory)]

# Separating and saving image pixels and image labels
(image_pixels_half, image_labels_half) = import_images_data(samples_directory_half)

# Reshaping image pixels
image_pixels_half = image_pixels_half.reshape((image_pixels_half.shape[0], 3072))

# Every image label will be encoded to a single digit: Cats: 0, Dogs: 1
encod_image_labels_half = LabelEncoder()
image_labels_half = encod_image_labels_half.fit_transform(image_labels_half)

# Split the dataset into 75% training data and 25% test data
X_train_half, X_test_half, y_train_half, y_test_half = train_test_split(image_pixels_half, image_labels_half, test_size=0.25, random_state=42)

print("Starting Classification...")  

# Train the SVM classifier
# Create an SVM classifier
svm_classifier = SVC()


grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, n_jobs=-1)


grid_search.fit(X_train_half, y_train_half)


best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM classifier with best hyperparameters
svm_classifier_best = SVC(**best_params)
svm_classifier_best.fit(X_train_half, y_train_half)


y_pred_half_best = svm_classifier_best.predict(X_test_half)


accuracy_half_best = accuracy_score(y_test_half, y_pred_half_best)
print(f'Accuracy (best model): {accuracy_half_best}')

conf_matrix_half_best = confusion_matrix(y_test_half, y_pred_half_best)


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_half_best, annot=True, fmt='d', cmap='Blues', xticklabels=encod_image_labels_half.classes_, yticklabels=encod_image_labels_half.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap (Best Model)')
plt.show()
