import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re


##############################################+=============================================================
root = tk.Tk()
root.configure(background="black")

w = 800 # width for the Tk root
h = 650 # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

# set the dimensions of the screen 
# and where it is placed
root.geometry('%dx%d+%d+%d' % (w, h, x, y))

root.title("Login Form")


username = tk.StringVar()
password = tk.StringVar()
        

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('asset/bg_login.jpg')
image2 = image2.resize((ws,hs), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)


def registration():
    from subprocess import call
    call(["python","02_registration.py"])
    root.destroy()

def login():
        # Establish Connection

    with sqlite3.connect('ml_db.db') as db:
         c = db.cursor()

        # Find user If there is any take proper action
         db = sqlite3.connect('ml_db.db')
         cursor = db.cursor()
         cursor.execute("CREATE TABLE IF NOT EXISTS admin_registration"
                           "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
         db.commit()
         find_entry = ('SELECT * FROM admin_registration WHERE username = ? and password = ?')
         c.execute(find_entry, [(username.get()), (password.get())])
         result = c.fetchall()

         if result:
            msg = ""
            print(msg)
            ms.showinfo("messege", "LogIn sucessfully")
            # ===========================================
            root.destroy()
            from subprocess import call
            call(['python','03_dashboard.py'])
            # ================================================
         else:
           ms.showerror('Oops!', 'Username Or Password Did Not Found/Match.')


title=tk.Label(root, text="Login Here", font=("Roboto", 28, "bold"),bd=5,bg="black",fg="white")
title.place(x=275,y=100,width=250)
        
Login_frame=tk.Frame(root,bg="white")
Login_frame.place(x=150,y=225)
        
logolbl=tk.Label(Login_frame,bd=0).grid(row=0,columnspan=2,pady=20)
        
lbluser=tk.Label(Login_frame,text="Username",compound=LEFT,font=("Roboto", 18, "bold"),bg="white").grid(row=1,column=0,padx=20,pady=10)
txtuser=tk.Entry(Login_frame,bd=5,textvariable=username,font=("",15))
txtuser.grid(row=1,column=1,padx=20)
        
lblpass=tk.Label(Login_frame,text="Password",compound=LEFT,font=("Roboto", 18, "bold"),bg="white").grid(row=2,column=0,padx=50,pady=10)
txtpass=tk.Entry(Login_frame,bd=5,textvariable=password,show="*",font=("",15))
txtpass.grid(row=2,column=1,padx=20)
        
btn_log=tk.Button(Login_frame,text="Login",command=login,width=15,font=("Roboto", 14, "bold"),bg="Green",fg="white")
btn_log.grid(row=3,column=1,pady=10)
btn_reg=tk.Button(Login_frame,text="Create Account",command=registration,width=15,font=("Roboto", 14, "bold"),bg="purple",fg="white")
btn_reg.grid(row=3,column=0,pady=60)
        
root.mainloop()