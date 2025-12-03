from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
Debug=True
window = Tk()
window.title("Twitter spam detection")
window.geometry('1368x768')
window.configure(bg='#03a9f4')
img = ImageTk.PhotoImage(Image.open("./images/13.jpg"))
panel = Label(window, image = img)
panel.place(x = 100, y = 100)
lbl = Label(window, text="Twitter Spam Detection",bg="#03a9f4" ,fg="white" ,width=50  ,height=1,font=("Arial Bold", 40))
lbl.pack(side=TOP)


def Nextpage():
        
        window.destroy()
        import file
        
def login():
    
    if user.get() =="Admin" and pw.get()=="Admin":
        Nextpage()
    else:
        messagebox.showinfo('sample 3', 'Invalid username or password')
        
        
        
      
   
lbl = Label(window, text="User id",width=20  ,height=1  ,fg="#1e81b0"  ,bg="white" ,font=('times', 14, ' bold '))

lbl.place(x=650, y=250)

userid=StringVar()
user = Entry(window,width=20,textvariable=userid  ,bg="white" ,fg="black",font=('times', 14, ' bold '))
user.place(x=900, y=250)

passw=StringVar()
lbl = Label(window, text="Password",width=20  ,height=1 ,fg="#1e81b0"  ,bg="white" ,font=('times', 14, ' bold '))

lbl.place(x=650, y=300)

pw = Entry(window,width=20 ,textvariable=passw ,show= "*", bg="white" ,fg="black",font=('times', 14, ' bold '))

pw.place(x=900, y=300)

button=Button(window, text="Login",command=login,fg="white"  ,bg="#1e81b0"  ,width=10  ,height=1 ,activebackground = "#6687a8" ,font=('times', 15, ' bold '))
button.place(x=800,y=365)

window.mainloop()
