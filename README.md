# generalizable_robotic_manipulation

~7/24/24
Algorithmic approach to robotic generalization

#### Replication Instructions
Run the cells of simulation.ipynb(make sure the initial ones with functions are all run first)  
The URDFs will load and the CNN model will work with the given file structure, and you can choose whether to use the 4.8mm error model or the 2.7mm error model, indicated by their name  

For retraining the CNN:
* Data regathering: run 'data_collection.ipynb' and adjust how many data points and the train-test split
* Model retraining: edit 'cnn/training.ipynb' or 'cnn/cnn.py', and make sure all changes are replicated to 'cnn.py'

### Key features

#### Convolutional Neural Network for object detection
*  Takes a 135x135 RGBA array of an image as input
*  Gives x and y coordinate of object in picture as output
*  Uses a regression model to find the values
*   Difficulties emerged because CNN archictures are usually used with classification tasks, hence little documentation
*  Training specifications:
*   run cells of 'cnn/training.ipynb' or run 'cnn/cnn.py'
*   Trained on Intel Core i7-1335U processor
*   480 epochs
*   MSE: 5e-6
*   Average distance error: 2.7 mm

#### Rapidly-exploring Random Tree for obstacle avoidance
* Takes in the obstacles in the simulation environment and a start and end joint configuration of the robot as input
* Returns a path for the robot to follow that will stay a specified distance away from all obstacles(set to a default of 15 cm)
* Follows S. M. LaValle's original conception of the RRT algorithm

Read 'report.pdf' for more info

