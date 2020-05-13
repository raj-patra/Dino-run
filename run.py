# -*- coding: utf-8 -*-
"""
@author: Raj Kishore Patra

"""
# In[ ]: Imports

import numpy as np
from PIL import Image
import cv2
import io
import time
import pandas as pd
import numpy as np
from IPython.display import clear_output
from random import randint
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
from io import BytesIO
import base64
import json

# In[ ]: Path Variables and Scripts

game_url = "chrome://dino"
chrome_driver_path = "./objects/chromedriver.exe"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

# Create ID for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"


# In[ ]: Game Class

'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the broswer window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''

class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path, chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(game_url)
        self._driver.execute_script("Runner.config.ACCELERATION=0.005")
        self._driver.execute_script(init_script)
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self._driver.close()
        

# In[ ]: Dinosaur Class
        

class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()
        
        
# In[ ]: Game State Class
        
        
class Game_state:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img() # Displays Processed image using openCV
        self._display.__next__() # Initializes the next co-routine
    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1] # Storing actions in a DF
        score = self._game.get_score()
        reward = 0.1
        is_over = False # Game Over
        if actions[1] == 1:
            self._agent.jump()
        elif actions[2] == 1:
            self._agent.duck()
        image = grab_screen(self._game._driver)
        self._display.send(image) # Displays the image on screen
        
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score # Logs score when game's over
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over # Experience Tuple
    
    
# In[ ]: Functions
        

def save_obj(obj, name):
    with open('objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open('objects/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:300, :500]
    image = cv2.resize(image, (80,80))
    return image

def show_img(graphs = False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
        
        
# In[ ]: Log Structures

       
loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(scores_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns = ['actions'])
q_values_df =pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns = ['qvalues'])


# In[ ]: Game Parameters

ACTIONS = 3             # Possible actions: jump, do nothing
GAMMA = 0.99            # Decay rate
OBSERVATION = 100.0     # Timesteps to observe before training
EXPLORE = 100000        # Frames over which to anneal epsilon
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000   # Number of previous transitions to remember
BATCH = 16              # Size of mini batch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows, img_cols = 80,80
img_channels = 4        # 4 Image stacks to be passed to model


# In[ ]: Training Variables saved as checkpoints


def init_cache():
    """
    initial variable caching , done only once
    """
    save_obj(INITIAL_EPSILON, "epsilon")
    t=0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")


# In[ ]: Model Building
    

def buildmodel():
    print("Creating Neural Network...")
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_cols,img_rows,img_channels), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
        
    model.add(Dense(ACTIONS))
        
    adam = Adam(lr = LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    
    #if not os.path.isfile(loss_file_path):
    model.save_weights('model.h5')
    print("Model Built...")
    return model


# In[ ]: Training Module


''' 
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''

def trainNetwrok(model, game_state, observe=False):
    last_time = time.time() # Stores previous obs in replay memory
    D = load_obj("D")
    
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 2 # 0-> Do nothing; 1-> Jump: 2->Duck
    
    x_t, r_0, terminal = game_state.get_state(do_nothing)
    # Get next step after performing the action
    
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # Stack 4 images for input
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*20*40*4
    initial_state = s_t
    
    if observe:
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        
        print("Loading Weights...")
        model.load_weights("model.h5")
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weights Loaded Successfully...")
        
    else:
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")
        model.load_weights("model.h5")
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        
    t = load_obj("time")
    
    while True:
        
        loss = 0; Q_sa = 0; action_index = 0; r_t = 0;
        a_t = np.zeros([ACTIONS])
        
        # Choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon: #randomly explore an action
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t) # Input the stack and get prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1
                
        # Reducing the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        # Running the selected action and observe next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        # Measuring Frame Rates
        print('fps: {0}'.format(1 / (time.time()-last_time)))
        last_time = time.time()
        
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        # Appends the new image to the i/p stack and removes the 1st one
        
        # Storing the Transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
            
        # Only train if done observing
        if t > OBSERVE:
            
            # Sampling a mini-batch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            # 32, 20, 40, 4
            
            targets = np.zeros((inputs.shape[0], ACTIONS))
            
            
            # Experience Replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]   # 4D Stack of images
                action_t = minibatch[i][1]  # Action Index
                reward_t = minibatch[i][2]  # Reward at state_t for action_t
                state_t1 = minibatch[i][3]  # Next State
                terminal = minibatch[i][4]  # Agent DoA due to the action
                
                
                inputs[i:i + 1] = state_t
                targets[i] = model.predict(state_t) # Predicted q values
                Q_sa = model.predict(state_t1) # Predicted q values for next
                
                if terminal:
                    # If terminated, Only reward
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA*np.max(Q_sa)
                    
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        
        s_t = initial_state if terminal else s_t1
        # Reset game to initial frame if terminated
        t = t + 1 
        
        # Saving Progress every 1000 Iteration
        if t % 1000 == 0:
            print("Saving Model...")
            game_state._game.pause() # Pause while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_obj(D, "D") # Saving Episodes
            save_obj(t, "time") # Caching Time steps
            save_obj(epsilon, "epsilon") # Caching Epsilon to avoid randomness
            
            loss_df.to_csv("./objects/loss_df.csv",index=False)
            scores_df.to_csv("./objects/scores_df.csv",index=False)
            actions_df.to_csv("./objects/actions_df.csv",index=False)
            q_values_df.to_csv(q_value_file_path,index=False)
            
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
            
        # Printing Information
            
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
            
        print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")
    
    
# In[ ]: Main function
    
   
def playGame(observe = False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)
    model = buildmodel()

    try:
        trainNetwrok(model, game_state, observe=observe)
    except StopIteration:
        game.end()
        
# In[ ]: Driver Code
               
#init_cache()
playGame(observe = False);