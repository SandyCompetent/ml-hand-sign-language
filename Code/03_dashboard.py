# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:37:36 2021

@author: sheet
"""

import sqlite3
import tkinter  as tk 
from tkinter import * 
import time
import numpy as np

import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 
from PIL import Image , ImageTk  

root = tk.Tk()

#------------------------------------------------------

root.configure(background="seashell2")


w = 1000 # width for the Tk root
h = 600 # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

# set the dimensions of the screen 
# and where it is placed
root.geometry('%dx%d+%d+%d' % (w, h, x, y))

root.title("BE Project - ML Based Sign Language")
#------------------Frame----------------------



#-------function------------------------

def create_gesture():
##### tkinter window ######
    from subprocess import call
    call(["python", "04_create_dataset.py"])   



def train_cnn():
##### tkinter window ######
    from subprocess import call
    call(["python", "05_train_cnn.py"])   
    

def test_cam():
##### tkinter window ######
    from subprocess import call
    call(["python", "06_test_cam.py"])   
    


#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('asset/bg_dashboard.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)


lbl = tk.Label(root, text="ML Based Sign Language ", font=('Roboto', 30,' bold '), height=1, width=50,bg="blue",fg="white")
lbl.place(x=0, y=0, width=1000)

framed = tk.LabelFrame(root, text=" --WELCOME-- ",width=900, height=350, bd=5, font=('Roboto', 14, ' bold '),bg="pink")
framed.grid(row=0, column=0, sticky='nw')
framed.place(x=50, y=100)
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
button1 = tk.Button(framed, text='Create Gesture',width=15,height=3,bg='dark blue',fg='white',command=create_gesture,font='bold').place(x=50,y=50, width=250)
button1 = tk.Button(framed, text='Train Model using CNN Algorithm',width=17,height=3,bg='dark blue',fg='white',command=train_cnn,font='bold').place(x=350,y=50, width=400)
button1 = tk.Button(framed, text='Run Test CAM',width=17,height=3,bg='dark blue',fg='white',command=test_cam,font='bold').place(x=50,y=200, width=250)

lb2 = tk.Label(root, text="ML Based Sign Language ", font=('Roboto', 15,' bold '), height=1, width=50,bg="orange",fg="white")
lb2.place(x=0, y=550, width=1000)

root.mainloop()
