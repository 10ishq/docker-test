# %% [markdown]
# # Fire Detection in Images
# 
# **Description about the dataset:**
# 
# Data was collected to train a model to distinguish between the images that contain fire (fire images) and regular images (non-fire images). Data is divided into 2 folders, fireimages folder contains 755 outdoor-fire images some of them contains heavy smoke, the other one is non-fireimages which contain 244 nature images (eg: forest, tree, grass, river, people, foggy forest, lake, animal, road, and waterfall).
# 
# **Objective: To create a classification model that can detect fire in images**
# 
# **Models used: Sequential CNN from scratch, Pretrained Xception with modifications**

# %% [markdown]
# ## Exploratory Data Analysis

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

sns.set_style('darkgrid')

# %% [markdown]
# **Let's first create a dataframe that contains the path to each picture and its corresponding label (fire or non fire).**
# 
# **Reading Paths**

# %%
#create an empty DataFrame
column_names=['path','label']
df = pd.DataFrame(columns=column_names,dtype=object)

#loop over fire images and label them 1
for dirname, _, filenames in os.walk('fire_dataset/fire_images'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        df = df.append(pd.DataFrame([[os.path.join(dirname, filename),'fire']],columns=['path','label']))

#loop over non fire images and label them 0
for dirname, _, filenames in os.walk('fire_dataset/non_fire_images'):
    for filename in filenames:
        df = df.append(pd.DataFrame([[os.path.join(dirname, filename),'non_fire']],columns=['path','label']))
        #print(os.path.join(dirname, filename))

#shuffle the dataset for redistribute the labels
df = df.sample(frac=1).reset_index(drop=True)
df.head(10)
# %% [markdown]
# **The height and width of images vary too much. We have to reshape them to a fixed shape before training**
# 
# ## Image Generation or Augmentation

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
generator = ImageDataGenerator(
    rotation_range= 20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range = 2,
    zoom_range=0.2,
    rescale = 1/255,
    validation_split=0.2,
)

# %% [markdown]
# **Creating the training and test generator**
# 
# **We will use the flow_from_dataframe method of the ImageDataGenerator class. It will take the path of the images from the dataframe along with their labels. We construct two generators, one for training and the other for validation.**
# 
# **Note: Our labels are strings 'fire ' and 'non_fire'. Image generator will automatically encode them to integer labels.**

# %%
train_gen = generator.flow_from_dataframe(df,x_col='path',y_col='label',images_size=(256,256),class_mode='binary',subset='training')
val_gen = generator.flow_from_dataframe(df,x_col='path',y_col='label',images_size=(256,256),class_mode='binary',subset='validation')

# %% [markdown]
# #### Class indices assigned by the Image generator

# %%
class_indices = {}
for key in train_gen.class_indices.keys():
    class_indices[train_gen.class_indices[key]] = key
    
print(class_indices)

# %% [markdown]
# **Hence an image predicted 0 will contain fire and 1 won't.**

# %% [markdown]
# ## Model creation by transfer learning

# %%
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.metrics import Recall,AUC
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stoppping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5)

# %%
xception = Xception(include_top = False,input_shape = (256,256,3))
input_to_model = xception.input
#turn off training
xception.trainable = False

x = Flatten()(xception.output)
x = Dense(64,activation = 'relu')(x)
output_to_model = Dense(1,activation = 'sigmoid')(x)
model2 = Model(inputs = input_to_model,outputs = output_to_model)

# %% [markdown]
# **Compiling the model**

# %%
model2.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',Recall(),AUC()])

# %% [markdown]
# **Fitting the model**

# %%
history2 = model2.fit(x = train_gen,batch_size=32,epochs=1,callbacks = [early_stoppping,reduce_lr_on_plateau],validation_data = val_gen)

# %% [markdown]
# ### Model Evaluation
# %%
eval_list = model2.evaluate(val_gen,return_dict=True)
for metric in eval_list.keys():
    print(metric+f": {eval_list[metric]:.2f}")




# %%
#loading the image
img = image.load_img('predict.jpg')
img

# %% [markdown]
# **Resizing the image and expanding its dimension to include the batch size - 1**

# %%
img = image.img_to_array(img)/255
img = tf.image.resize(img,(256,256))
img = tf.expand_dims(img,axis=0)

print("Image Shape",img.shape)

# %% [markdown]
# **Prediction**

# %%
prediction = int(tf.round(model2.predict(x=img)).numpy()[0][0])
print("The predicted value is: ",prediction,"and the predicted label is:",class_indices[prediction])


# %%
tf.saved_model.save(model2, 'new_saved_model')

# %%
converter = tf.lite.TFLiteConverter.from_saved_model('new_saved_model') # path to the SavedModel directory
tflite_model = converter.convert()


# %%
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# %% [markdown]
# # Thank You


