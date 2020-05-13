# Dino Run
## With Reinforcement Learning

A Deep Convolutional Neural Network to play Google Chrome's offline Dino Run game by learning action patterns from visual input using a model-less Reinforcement Learning Algorithm

<br><br/>

### Installation 
Start by cloning the repository
<br>
<br>
`$ git clone https://github.com/Paperspace/DinoRunTutorial.git`
<br>
You need to initialize the file system to save progress and resume from last step.<br/>
Invoke `init_cache()` for the first time to do this <br/>


Dependencies can be installed using pip install or conda install for Anaconda environment<br><br>

- Python 3.6 Environment with ML libraries installed (numpy,pandas,keras,tensorflow etc)
- Selenium
- OpenCV

<br/>

### This project contains following files:

-  `run.py`: Main Driver Code
-  `model.py`:  Neural network implementation (Model Architecture)
-  `model.h5`:  Trained & Saved model
-  `progress_viz.py`: Visualize Training Progress
-  `chromedriver.exe`: Chrome Driver for Windows
