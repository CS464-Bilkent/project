from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Importing the dataset
print("Importing Dataset...")
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


path_to_sample_images = 'raw-img'
# Directories for every single image
samples_directory = list(paths.list_images(path_to_sample_images))

# Seperating and saving image pixels and image labels
(image_pixels, image_labels) = import_images_data(samples_directory)

# Reshaping image pixels
image_pixels = image_pixels.reshape((image_pixels.shape[0], 3072))

# Every image label will be incoded to a single digit: Cats: 0, Dogs: 1
encod_image_labels = LabelEncoder()
image_labels = encod_image_labels.fit_transform(image_labels)


# Spliting dataset into 75% training data and 25% test data. It can be changed.
print("Split Data...")
(trainig_image_pixels, test_image_pixels, training_image_labels, test_image_labels) = train_test_split(image_pixels, image_labels, test_size=0.25, random_state=42)


print("Starting Classification...")   

best_n_neighbors = 0
best_accuracy = 0
best_conf_matrix = None

for n_neighbors in range(1, 11):
    print(f"Trying n_neighbors = {n_neighbors}...")
    
   
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

   
    knn_classifier.fit(trainig_image_pixels, training_image_labels)

  
    prediction = knn_classifier.predict(test_image_pixels)

    # Calculate accuracy
    accuracy = accuracy_score(test_image_labels, prediction)
    print(f"Accuracy: {accuracy}")

    # Update best parameters if this model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_neighbors = n_neighbors
        best_conf_matrix = confusion_matrix(test_image_labels, prediction)  
# Print the best parameters and corresponding score
print(f"Best n_neighbors: {best_n_neighbors}")
print(f"Best accuracy: {best_accuracy}")

plt.figure(figsize=(10, 7))
sns.heatmap(best_conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=encod_image_labels.classes_, yticklabels=encod_image_labels.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Best n_neighbors')
plt.show()

  
n_neighbors_values = []
accuracy_values = []


for n_neighbors in range(1, 11):
    print(f"Trying n_neighbors = {n_neighbors}...")
    
    
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

   
    knn_classifier.fit(trainig_image_pixels, training_image_labels)

    prediction = knn_classifier.predict(test_image_pixels)

   
    accuracy = accuracy_score(test_image_labels, prediction)
    print(f"Accuracy: {accuracy}")

    # Store n_neighbors and accuracy values
    n_neighbors_values.append(n_neighbors)
    accuracy_values.append(accuracy)


plt.figure(figsize=(10, 7))
plt.plot(n_neighbors_values, accuracy_values, marker='o')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('n_neighbors vs Accuracy')
plt.xticks(n_neighbors_values)
plt.grid(True)
plt.show()
