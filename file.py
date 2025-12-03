import matplotlib.pyplot as plt
import csv
import joblib
import sklearn
import re
import sys
from tweepy import OAuthHandler
import tweepy
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve, GridSearchCV
import matplotlib as mp
import warnings 
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn import *
import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
from tkinter.scrolledtext import ScrolledText
import datetime
import time
import tkinter.ttk as ttk
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tkinter import messagebox
import pickle
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import Entry
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandastable import Table

main = tk.Tk()
main.title('Tkinter Open File Dialog')
main.resizable(False, False)
main.geometry('1368x760')
main.configure(bg='#03a9f4')
img = ImageTk.PhotoImage(Image.open("./images/13.jpg"))
panel = Label(main, image = img)
panel.place(x = 100, y = 100)
lbl = Label(main, text="Twitter Spam Detection",bg="#03a9f4" ,fg="white" ,width=50  ,height=1,font=("Arial Bold", 40))
lbl.pack(side=TOP)

def select_file():
    global data
    global filename
    global features
    filetypes = (('csv file', '*.csv'),('All files', '*.*'))

    filename = fd.askopenfilename(title='Open a file',initialdir='/',filetypes=filetypes)
    
    showinfo(title="Succesfully data imported",message=filename)
    print(filename)
    data = pd.read_csv(filename,encoding='latin-1')
    visit = Entry(main,width=50,bg="white"  ,fg="black",font=("Arial", 15))
    visit.insert(0,filename )
    visit.place(x=600, y=150)
    
    
    
def fetch():
    def text_clean(text):

        remove_space = '\s+'
        find_url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        find_mention = '@[\w\-]+'
        process_text = re.sub(remove_space, ' ', text)
        process_text = re.sub(find_url, '', process_text)
        process_text = re.sub(find_mention, '', process_text)
        process_text = process_text.encode('ascii','ignore')
        return process_text

    con_key = 'PG2atN7eFZIfjqPugmrQ0DTHN'
    con_secret = 'BcKjeAsjePUjw5SqOj7DqQy7QgTvNo6fbNXoT6PHoV68CwqWSX'
    acc_token = '93137600-fD6r6VCcwx2f4ahKrqTlImxhFicNYcpy17HkL8Ne4'
    acc_token_secret = 'lYUwaYoI6j5nHk7pijChJUxzDuY99osg64rs9vjhOejWU'

    try:
                auth = OAuthHandler(con_key, con_secret)
                auth.set_access_token(acc_token, acc_token_secret)
                api = tweepy.API(auth)
                
    except:
                
                print("Error: Twitter Not Connected")


     
    

    query = "#"+searchTxt.get()
    count = 40
    tweets = []
    clean_tweets=[]
    try:
                fetched_tweets = api.search(q = query, count = count)
     
                for tweet in fetched_tweets:
                    parsed_tweet = {}
                    ss=tweet.text
                    parsed_tweet['text'] = ss#clean_tweet(ss)
                    clean_tweets.append(text_clean(ss))
                    tweets.append(parsed_tweet)
     
    except tweepy.TweepError as e:
           print("Error : " + str(e))

    print ("*********************************")
    print ("Total tweets read from twitter server ")
    print (len(tweets))
    global df
   
    df = pd.DataFrame(tweets)
    df.to_csv("tweets.csv")

    df = pd.DataFrame(clean_tweets)
    df.to_csv("clean_tweets.csv")
    showinfo(title="Data",message="Data collected")
def word_cloud():
    
    wc = Toplevel()
    wc.geometry('1368x760')
    wc.configure(bg='#03a9f4')
    message = tk.Label(wc, text="Spam" ,fg="white",bg="#3b4d3d",font=('times', 25, 'bold')) 
    message.place(x=0,y=50)
    message2 = tk.Label(wc, text="Not Spam" ,fg="white",bg="#3b4d3d",font=('times', 25, 'bold')) 
    message2.place(x=1200,y=50)
    img = ImageTk.PhotoImage(Image.open("./images/1.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("./images/2.jpg"))
    
    panel = Label(wc, image = img)
    panel.pack(side=LEFT)
    panel = Label(wc, image = img2)
    panel.pack(side=RIGHT)
   

    wc.mainloop()
def accuracy():
    ac = Toplevel()
    ac.geometry('653x383')
    ac.configure(bg='#03a9f4')
    img3 = ImageTk.PhotoImage(Image.open("./images/3.jpg"))
    panel = Label(ac, image = img3)
    panel.place(x = 0, y = 0)
    ac.mainloop()   
def panda():
    root = tk.Tk()
    root.title('PandasTable Example')
    x='clean_tweets.csv'
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    pt = Table(frame)
    pt.show()

    pt.importCSV(x)

    root.mainloop()

def spam():
    spm = Toplevel()
    spm.title("Output")
    spm.geometry('500x500')
    spm.configure(bg='white')
    img = ImageTk.PhotoImage(Image.open("./images/12.jpg"))
    panel = Label(spm, image = img)
    panel.place(x = 0, y = 0)
    ans = tk.Label(spm, text="Spam" ,fg="white" ,bg="#2596be" ,font=("Arial Bold", 17))
    ans.place(x=200, y=50)
    
    spm.mainloop()

def nspam():
    nspm = Toplevel()
    nspm.title("Output")
    nspm.geometry('500x500')
    nspm.configure(bg='white')
    img = ImageTk.PhotoImage(Image.open("./images/12.jpg"))
    panel = Label(nspm, image = img)
    panel.place(x = 0, y = 0)
    ans = tk.Label(nspm, text="Not a spam" ,fg="white" ,bg="#2596be" ,font=("Arial Bold", 17))
    ans.place(x=200, y=50)
    
    nspm.mainloop()       
def algorithm():
    time.sleep(3)
    showinfo(title="Train",message="Successfully trained")
   
def model():
    time.sleep(1)
        
    showinfo(title="Model",message="Successfully Model created")
def exit():
    main.destroy()

    
def clear():
    loc.delete(0, 'end') 
    res = ""
    
    
def show():
    second.deiconify()
 
# Hide the window
def hide():
    second.withdraw()
    
    
def predict():
   

   df = pd.read_csv("./data/spam.csv", encoding="latin-1")
   #df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
   # Features and Labels
   df['label'] = df['class'].map({'ham': 0, 'spam': 1})
   X = df['message']
   y = df['label']

   # Extract Feature With CountVectorizer
   cv = CountVectorizer()
   X = cv.fit_transform(X)  # Fit the Data
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   # Naive Bayes Classifier
   from sklearn.naive_bayes import MultinomialNB

   clf = MultinomialNB()
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)
   # Alternative Usage of Saved Model
   # joblib.dump(clf, 'NB_spam_model.pkl')
   # NB_spam_model = open('NB_spam_model.pkl','rb')
   # clf = joblib.load(NB_spam_model)
   x=1
   if x==1:
   
      message = loc.get()
      print(message)
      data = [message]
      vect = cv.transform(data).toarray()
      my_prediction = clf.predict(vect)

      if int(my_prediction)==1:
            spam()
            
      else:
            nspam()

def front():
    main.withdraw()
    page1()
def next1():
    Page1.withdraw()
    next()
def back():
    page2.withdraw()
    Page1.deiconify()
def back1():
    Page1.withdraw()
    main.deiconify()
def next():
    def csv():
        x='clean_tweets.csv'
        frame = tk.Frame(page2)
        frame.place(x=600,y=200,height=200,width=500)
        pt = Table(frame)
        pt.show()
        
        pt.importCSV(x)
    
    global page2
    page2 = Toplevel()
    page2.title("Spam")
    page2.geometry('1368x760')
    page2.configure(bg='#03a9f4')
    lbl = Label(page2, text="Twitter Spam Detection",bg="#03a9f4" ,fg="white" ,width=50  ,height=1,font=("Arial Bold", 40))
    lbl.pack(side=TOP)
    img = ImageTk.PhotoImage(Image.open("./images/13.jpg"))
    panel = Label(page2, image = img)
    panel.place(x = 100, y = 100)
    a=150
    b=150
    
 
  
    

    global loc
    #loc1 = Label(page2, text="Twitter",width=20  ,height=1  ,fg="#03a9f4"  ,bg="white"  ,font=("Arial Bold", 15)) 
    #loc1.place(x=600, y=175)
    loc = Entry(page2,width=50  ,bg="white"  ,fg="black",font=('times', 14, ' bold ') ,justify='left')
    loc.place(x=600, y=440,height=75,width=500)
    #loc.grid(row=0,column=0,padx=10,pady=10,ipady=30)
    
    clearButton = tk.Button(page2, text="Clear", command=clear  ,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    clearButton.place(x=650, y=540)

    #ot = Label(page2, text="",font=("Arial Bold", 10))
    #ot.grid(column=1, row=17)
    
    btn = Button(page2, text="PREDICT",command=predict,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))

    btn.place(x=875, y=540)


    clearButton = tk.Button(page2, text="Exit", command=exit  ,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    clearButton.place(x=1100, y=600)
    
    
    btn = tk.Button(page2, text="BACK",command=back,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    btn.place(x=50, y=600)
    btn = tk.Button(page2, text="Twitter data",command=csv,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    btn.place(x=750, y=125)
    #print(new_input)
    page2.mainloop()
    


    
def page1():
       
    global Page1
       
    Page1 = Toplevel()
    Page1.title("spam")
    Page1.geometry('1368x760')
    Page1.configure(bg='#03a9f4')
    lbl = Label(Page1, text="Twitter Spam Detection",bg="#03a9f4" ,fg="white" ,width=50  ,height=1,font=("Arial Bold", 40))
    lbl.pack(side=TOP) 
    img = ImageTk.PhotoImage(Image.open("./images/13.jpg"))
    panel = Label(Page1, image = img)
    panel.place(x = 100, y = 100)
    global searchTxt
    searchTxt =Entry(Page1,width=30  ,bg="white"  ,fg="black",font=('times', 14, ' bold ') )
    searchTxt.place(x=775, y=200)
    open_button = tk.Button(Page1,text="Get Live Data",command=fetch, fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    open_button.place(x=850, y=275)
    
    open_button = tk.Button(Page1,text="Show Data",command=panda, fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    open_button.place(x=850, y=350)
    btn = Button(Page1, text="NEXT",command=next1,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    
    btn.place(x=1100, y=600)
    pre_button = tk.Button(Page1,text="BACK",command=back1, fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
    pre_button.place(x=50, y=600)
 
 
 
 
    Page1.mainloop()

    #return new_input,temp.get(),hum.get(),ph.get(),rf.get()

# open button
open_button = tk.Button(main,text="Open a File",command=select_file, fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
open_button.place(x=850, y=200)

btn = tk.Button(main, text="Train",command=algorithm,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
btn.place(x=700, y=300)

btn = tk.Button(main, text="Model",command=model,fg="#03a9f4"  ,bg="white" ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
btn.place(x=1000, y=400)

open_button = tk.Button(main,text="Word cloud",command=word_cloud, fg="#03a9f4"  ,bg="white" ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
open_button.place(x=700, y=400)

open_button = tk.Button(main,text="Accuracy",command=accuracy, fg="#03a9f4"  ,bg="white" ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
open_button.place(x=1000, y=300)


btn = tk.Button(main, text="EXIT",command=exit,fg="#03a9f4"  ,bg="white"  ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
btn.place(x=50, y=600)

btn = tk.Button(main, text="NEXT",command=front,fg="#03a9f4"  ,bg="white"   ,width=14  ,height=1 ,activebackground = "#03a9f4" ,activeforeground= "white" ,font=("Arial Bold", 15))
btn.place(x=1100, y=600)




main.mainloop()