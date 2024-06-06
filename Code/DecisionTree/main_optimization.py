  from typing_extensions import Self
  import os
  import numpy as np
  from sklearn import tree
  from sklearn.preprocessing import LabelEncoder
  from PIL import Image
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
  from sklearn.model_selection import GridSearchCV

  class MyDecisionTree:

      def load_images(self, base_path, classes, image_size=(256, 256)):

          # Initialize lists to store images and labels
          images = []
          labels = []
          for class_folder in classes:
              print(class_folder)
              folder_path = os.path.join(base_path, class_folder)
              image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

              for image_file in image_files:
                  image_path = os.path.join(folder_path, image_file)
                  image = Image.open(image_path).convert('RGB')
                  image = image.resize(image_size)
                  image_array = np.array(image)
                  # print(image_array.shape)
                  images.append(image_array)
                  labels.append(class_folder)  # Use the folder name as the label

          # Convert images to a numpy array and normalize(# Normalize pixel values to [0, 1])
          images = np.array(images, dtype=np.float32) / 255.0
          # print(images.shape)

          # Flatten the images (samples x features) to convert array to 1D array
          images_flattened = images.reshape(len(images), -1)

          # Encode class labels to numerical values
          label_encoder = LabelEncoder()
          labels = label_encoder.fit_transform(labels)

          # return an array of features(image pixel)
          # each row of this array present extracted features from one train images
          # and labels array present an array of class labels
          return images_flattened, labels

      def plot_tree(self, dtc, feature_names,classes):
          # print a nicer tree using graphviz
          import graphviz
          dot_data = tree.export_graphviz(dtc, out_file=None,
                                          feature_names=feature_names,
                                          class_names=classes,
                                          filled=True, rounded=True)
          graph = graphviz.Source(dot_data)
          graph.render("DecisionTree")  # the DecisionTree will save in a pdf file

      def decisiontree_evaluate(self, y_true, y_pred):

          accuracy = accuracy_score(y_true, y_pred)
          precision = precision_score(y_true, y_pred, average='macro')
          recall = recall_score(y_true, y_pred, average='macro')
          f1 = f1_score(y_true, y_pred, average='macro')
          conf_matrix = confusion_matrix(y_true, y_pred)
          class_report = classification_report(y_true, y_pred)

          # Print the metrics
          print("Accuracy:", accuracy)
          print("Precision:", precision)
          print("Recall:", recall)
          print("F1 Score:", f1)
          print("Confusion Matrix:\n", conf_matrix)
          print("Classification Report:\n", class_report)


      def decisiontree_optimization(self,X_train,Y_train,param_grid):

          from sklearn.model_selection import GridSearchCV
          # Train Decision Tree Classifier
          dtc = tree.DecisionTreeClassifier(criterion="entropy")
          dtc.fit(X_train, Y_train)
          # Instantiate GridSearchCV
          grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5)

          # Fit the GridSearchCV object to the training data
          grid_search.fit(X_train, Y_train)

          # Print the best hyperparameters found
          print("Best hyperparameters:", grid_search.best_params_)

          # Get the best model
          best_model = grid_search.best_estimator_

          return(best_model)


      def decisiontree_optimization(self, X_train, Y_train, X_val, Y_val, param_grid):
        """
        Optimize decision tree hyperparameters using GridSearchCV and log performance metrics.

        :param X_train: Training features.
        :param Y_train: Training labels.
        :param X_val: Validation features.
        :param Y_val: Validation labels.
        :param param_grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
        :return: Best model after grid search, performance log.
        """
        dtc = tree.DecisionTreeClassifier(criterion="entropy")
        grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, return_train_score=True)
        grid_search.fit(X_train, Y_train)

        print("Best hyperparameters:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        # Log performance metrics
        results = grid_search.cv_results_
        performance_log = {
            'params': results['params'],
            'mean_train_score': results['mean_train_score'],
            'mean_test_score': results['mean_test_score']
        }

        return best_model, performance_log

      def plot_performance(self, performance_log):

        import matplotlib.pyplot as plt
        """
        Plot performance improvement per hyperparameter combination.

        :param performance_log: Dictionary containing parameters and their corresponding mean train and test scores.
        """
        params = performance_log['params']
        mean_train_score = performance_log['mean_train_score']
        mean_test_score = performance_log['mean_test_score']

        # Extract hyperparameters names and values for plotting
        param_names = list(params[0].keys())
        for param_name in param_names:
            plt.figure(figsize=(10, 6))
            param_values = [param[param_name] for param in params]
            plt.plot(param_values, mean_train_score, label='Train Score', marker='o')
            plt.plot(param_values, mean_test_score, label='Val Score', marker='o')
            plt.xlabel(param_name)
            plt.ylabel('Score')
            plt.title(f'Performance vs {param_name}')
            plt.legend()
            plt.grid(True)
            plt.show()



  # Update this with the correct path to load val and train data
  base_path = "/content/drive/MyDrive/ColabNotebooks/DataSet/"
  classes = ["airport_terminal", "market", "movie_theater", "museum", "restaurant"]
  image_size = (256, 256)
  X_train = []
  Y_train = []

  X_train, Y_train = odt.load_images(os.path.join(base_path, "train"), classes)
  X_val = []
  Y_val = []
  X_val,Y_val =odt.load_images(os.path.join(base_path, "val"),classes)



 
  # create an object of the MyDecisionTree class
  odt = MyDecisionTree()

  # Hyperparameter optimization min_samples_split
  best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, { 'min_samples_leaf': [4,8,11,13]})

  # Evaluate the optimized model on the validation set
  y_val_pred_best=[]
  y_val_pred_best = best_model.predict(X_val)
  print("Optimized Model Evaluation on Validation Set:")
  odt.decisiontree_evaluate(Y_val, y_val_pred_best)
  # Plot performance comparison
  odt.plot_performance(performance_log)


  # Hyperparameter optimization min_samples_leaf
  best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, {'min_samples_split': [15,20,30]})
  # Evaluate the optimized model on the validation set
  y_val_pred_best=[]
  y_val_pred_best = best_model.predict(X_val)
  print("Optimized Model Evaluation on Validation Set:")
  odt.decisiontree_evaluate(Y_val, y_val_pred_best)
  # Plot performance comparison
  odt.plot_performance(performance_log)

  # Hyperparameter optimization
  best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, {'max_Depth': [4,7,8,10,12,14]})
  # Evaluate the optimized model on the validation set
  y_val_pred_best=[]
  y_val_pred_best = best_model.predict(X_val)
  print("Optimized Model Evaluation on Validation Set:")
  odt.decisiontree_evaluate(Y_val, y_val_pred_best)
  # Plot performance comparison
  odt.plot_performance(performance_log)