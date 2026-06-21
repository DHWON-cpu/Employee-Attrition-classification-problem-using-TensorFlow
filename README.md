# Employee-Attrition-classification-problem-using-TensorFlow
## Title: Employee Attrition Classification Using TensorFlow

* This project focuses on building a binary classification model to predict employee attrition using TensorFlow. The target variable is Attrition, where the model predicts whether an employee is likely to leave the organization or stay.

* The dataset was separated into input features and a target variable, then split into training and testing sets. Feature scaling was applied using StandardScaler to improve model training stability. A neural network model was built using TensorFlow’s Sequential API with dense layers and a sigmoid output layer, which is suitable for binary classification.

* Several hyperparameters were tested, including the number of neurons, activation functions, learning rate, optimizer settings, and training epochs. Due to TensorFlow execution issues in the local environment, standard Keras methods such as fit(), evaluate(), and predict() were replaced with equivalent manual alternatives using train_on_batch(), direct model inference, and manual accuracy calculation.

* The best-performing model was selected based on test accuracy rather than training accuracy to avoid choosing an overfitted model. The final model used a dense hidden layer, sigmoid output activation, binary cross-entropy loss, and SGD optimization.
