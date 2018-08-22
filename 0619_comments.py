import Tkinter as tk
import tkFileDialog,ttk
import tkMessageBox as mb
import os,glob
import cv2,csv
import configparser
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import pandas as pd
import io

# root window
root = tk.Tk()
root.title("Control Panel")

# add tab
tabControl=ttk.Notebook(root)
tab1=ttk.Frame(tabControl)
tab2=ttk.Frame(tabControl)
tabControl.add(tab1,text="Video Process")
tabControl.add(tab2,text="Data Analysis")
tabControl.pack(expand=1,fill="both")

# add label
ttk.Label(tab1,text="Choose model:").grid(column=0,row=0,sticky="W")

modelvariable=tk.StringVar()
modelChoosen=ttk.Combobox(tab1,width=15,textvariable=modelvariable,state="readonly")
modelChoosen['values']=("BMW","TESLA","MB","HOO60")
modelChoosen.grid(column=0,row=1,sticky="WE")
modelChoosen.current(0)

# global variable
model=" "
videoPath=" "
videoName=" "

def getFile():
    global model,videoPath,videoName
    model=modelvariable.get()
    videoPath = tkFileDialog.askopenfilename()
    videoName=os.path.basename(videoPath)

# add button
videoButton=ttk.Button(tab1,text="Select Video",width=15,command=getFile)
videoButton.grid(column=1,row=1,columnspan=3)

# flagDict={"ACC":False,"HTOR":False,"LSA":False}

# read parameter
config=configparser.ConfigParser()
config.read("parameter.ini")

thresholdDict=config["THRESHOLD"]
flagDict=config["FLAG"]
funcBarDict={0:"ACC",1:"LSA",2:"HTOR"}

# callback function for trackbar
def callback_func(self):
    pass
# callback function for trackbar
def callback_threshold(self):
    global funcBarDict,thresholdDict
    threshold=cv2.getTrackbarPos("threshold","matching:Press ESC to close")
    func=funcBarDict[cv2.getTrackbarPos("functions","matching:Press ESC to close")]
    thresholdDict[func]=str(threshold/10.0)

def matchIcon():
    
    global model,videoPath,thresholdDict,config
    locDict={}
    
    cv2.namedWindow("matching:Press ESC to close")
    cv2.createTrackbar("functions","matching:Press ESC to close",0,2,callback_func)
    cv2.createTrackbar("threshold","matching:Press ESC to close",9,10,callback_threshold)

    video=cv2.VideoCapture(videoPath)


    while True:
        ret,img=video.read()
        if ret==True:
            for func in flagDict:
                if int(flagDict[func])==0:
#                     process image
                    image_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    iw,ih=image_gray.shape[::-1]
                    
#                   default search area top right 1/4 frame
                    image_gray=image_gray[:ih/2,iw/2:]
#                     process template
                    templatePath=model+"_"+func+".jpg"
                    template=cv2.imread(templatePath)
                    template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
                    tw,th=template_gray.shape[::-1]
#                     template edge detection
                    high_canny_t,_=cv2.threshold(template_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    low_canny_t=0.5 * high_canny_t
                    template_edge=cv2.Canny(template_gray,low_canny_t,high_canny_t)
#                     image edge detection
                    high_canny_i,_=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    low_canny_i=0.5 * high_canny_i
                    image_edge=cv2.Canny(image_gray,low_canny_i,high_canny_i)
#                     match template
                    res=cv2.matchTemplate(image_edge,template_edge,cv2.TM_CCORR_NORMED)
                    _,max_val,_,max_loc=cv2.minMaxLoc(res)
                    top_left = (max_loc[0]+iw/2,max_loc[1])
                    bot_right = (top_left[0]+tw,top_left[1]+th)
        
                    if max_val<float(thresholdDict[func]):
                        cv2.putText(img,func+":"+str(max_val)[:4],top_left,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA,False)

                    else:
                        flagDict[func]=str(1)
#                         top_left = (max_loc[0]+iw/2,max_loc[1])
#                         bot_right = (top_left[0]+tw,top_left[1]+th)
                        locDict[func]=(top_left,bot_right)

                else:
#                     if icon detected then
                    pass
#             diplay image with detected icon and undected max_val
            for func,loc in locDict.iteritems():
#                 print func,locDict[func]
                cv2.rectangle(img,loc[0],loc[1],(0,0,255),2)
                cv2.putText(img,func,loc[1],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,False)
                
            
            cv2.imshow("matching:Press ESC to close",img)
            cv2.waitKey(1)
        
        else:
            break
            
        if cv2.waitKey(1) & 0xFF == 27:
            break

            
    video.release()
    cv2.destroyAllWindows()
    
    for func,loc in locDict.iteritems():
        config.set(model,func,str(loc))
        
    with open("parameter.ini","w") as configfile:
        config.write(configfile)
    
matchButton=ttk.Button(tab1,text="Auto Match",width=15,command=matchIcon)
matchButton.grid(column=4,row=1)

ttk.Label(tab1,text="Hour:  Min:  Sec").grid(column=1,row=2,columnspan=3,sticky="W")

hourvariable=tk.IntVar()
hourEntered=ttk.Entry(tab1,width=4,textvariable=hourvariable)
hourEntered.grid(column=1,row=3)

minvariable=tk.IntVar()
minEntered=ttk.Entry(tab1,width=4,textvariable=minvariable)
minEntered.grid(column=2,row=3)

secvariable=tk.IntVar()
secEntered=ttk.Entry(tab1,width=4,textvariable=secvariable)
secEntered.grid(column=3,row=3)

flag=False
left_top=(0,0)
right_bot=(0,0)
frame=""

def onMouse(event,x,y,flags,para):
    global flag,left_top,right_bot,posvariable
#     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    
    if event==cv2.EVENT_LBUTTONDOWN:
        flag=True
        left_top=(x,y)

    elif event==cv2.EVENT_MOUSEMOVE:
        if flag:
#             cv2.rectangle(frame,left_top,(x,y),(0,0,255),-1)
            pass
# -1 means filled rectangle
    elif event==cv2.EVENT_LBUTTONUP:
        flag=False
        right_bot=(x,y)
        cv2.rectangle(frame,left_top,right_bot,(0,0,255),2)
        posvariable.set(str(left_top)+","+str(right_bot))

def getFrame():
    global videoPath,frame
    video=cv2.VideoCapture(videoPath)
    # print video.isOpened()
    total_frames=video.get(7)
#     print total_frames
    fps=video.get(5)
#     print fps
    hour=hourvariable.get()
    minute=minvariable.get()
    sec=secvariable.get()
    mstime=(3600*hour+60*minute+sec)*1000
    second=3600*hour+60*minute+sec
    frame_pos=second*fps
#     print frame_pos
    if frame_pos>total_frames-1:
        mb.showerror("error","Frame not found!\nPlease check what you input")
    else:
#         video.set(1,frame_pos)
        video.set(0,mstime)
        ret,frame=video.read()
#         print frame.shape
        # print frame
#         print type(frame)

        cv2.namedWindow("select icon:Press ESC to close")
        cv2.setMouseCallback("select icon:Press ESC to close", onMouse, 0)
#         cv2.rectangle(frame,left_top,right_bot,(0,0,255),1)
        while True:
        
            cv2.imshow("select icon:Press ESC to close",frame)
    
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cv2.destroyAllWindows()
    
frameButton=ttk.Button(tab1,text="Extract Frame",width=15,command=getFrame)
frameButton.grid(column=4,row=3)

ttk.Label(tab1,text="Choose function:").grid(column=0,row=2,sticky="W")

funcvariable=tk.StringVar()
funcChoosen=ttk.Combobox(tab1,width=15,textvariable=funcvariable,state="readonly")
funcChoosen['values']=("ACC","LSA","HTOR","ALL","WHEEL")
funcChoosen.grid(column=0,row=3,sticky="WE")
funcChoosen.current(0)

ttk.Label(tab1,text="Position:").grid(column=0,row=4,sticky="W")

posvariable=tk.StringVar()
posEntered=ttk.Entry(tab1,width=20,textvariable=posvariable)
posEntered.grid(column=0,row=5,sticky="W")
posvariable.set("(0,0),(0,0)")


def updateIni():
    global config,left_top,right_bot,model,funcvariable
    func=funcvariable.get()
    pos=(left_top,right_bot)
#     print model
    config.set(model,func,str(pos))
    with open("parameter.ini","w") as configfile:
        config.write(configfile)

writeButton=ttk.Button(tab1,text="UPDATE",width=15,command=updateIni)
writeButton.grid(column=1,row=5,columnspan=3)

def initialize():

    config=configparser.ConfigParser()

    config["THRESHOLD"]={"ACC":0.9,"LSA":0.9,"HTOR":0.9}
    config["FLAG"]={"ACC":0,"LSA":0,"HTOR":0}
    config["BMW"]={"ACC":((0,0),(0,0)),"LSA":((0,0),(0,0)),"HTOR":((0,0),(0,0))}
    config["TESLA"]={"ACC":((0,0),(0,0)),"LSA":((0,0),(0,0))}
    config["MB"]={"ACC":((0,0),(0,0)),"LSA":((0,0),(0,0))}
    config["HOO60"]={"ALL":((0,0),(0,0)),"WHEEL":((0,0),(0,0))}

    with open("parameter.ini","w") as configfile:
        config.write(configfile)
    # if add new function, take all below into considerartion:
    # flagDict, conifg["threshold"], config[model], colorDict, createTrackbar and funcbarDict
    # all possible colors should be added in colorDict (not only what you want to be marked)
    # flagDict func should be in alphabetical order (same with config[model])


initializeButton=ttk.Button(tab1,text="Initialize",width=15,command=initialize)
initializeButton.grid(column=4,row=5)

def lefts(n):
    return str.isdigit(n) or n==","

colorDict={"BMW":{"ACC":{"G":"ON","W":"STANDBY","N":"OFF","B":"ERROR"},
                  "LSA":{"G":"ON","W":"STANDBY","N":"OFF","B":"ERROR"},
                  "HTOR":{"Y":"HOR","R":"TOR","G":"G","W":"W"}},
           "TESLA":{"ACC":{"B":"ON","W":"STANDBY","N":"OFF","G":"ERROR"},
                    "LSA":{"B":"ON","W":"STANDBY","N":"OFF","G":"ERROR"}},
           "MB":{"ACC":{"G":"ON","W":"STANDBY","N":"OFF","Y":"Y"},
                 "LSA":{"G":"ON","W":"STANDBY","N":"OFF","Y":"Y"}},
           "HOO60":{"WHEEL":{"G":"ON","Y":"ON","R":"TOR","W":"STANDBY"}}}

Gmin=np.array([40,20,40],np.uint8)
Gmax=np.array([80,255,255],np.uint8)
Bmin=np.array([90,70,120],np.uint8)
Bmax=np.array([130,255,220],np.uint8)
Rmin1=np.array([160,100,100],np.uint8)
Rmax1=np.array([180,255,255],np.uint8)
Rmin2=np.array([0,100,100],np.uint8)
Rmax2=np.array([10,255,255],np.uint8)
Ymin=np.array([20,20,160],np.uint8)
Ymax=np.array([40,255,255],np.uint8)
# hsv h-range(0,179) in opencv

def detectColor(img_bgr,model):

    color=""
    img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    if model=="BMW"or model=="MB" or model=="HOO60":
        Gmask=cv2.inRange(img_hsv,Gmin,Gmax)
        Gcnt=cv2.countNonZero(Gmask)
        Rmask1=cv2.inRange(img_hsv,Rmin1,Rmax1)
        Rmask2=cv2.inRange(img_hsv,Rmin2,Rmax2)
        Rcnt1=cv2.countNonZero(Rmask1)
        Rcnt2=cv2.countNonZero(Rmask2)
        Rcnt=Rcnt1+Rcnt2
        Ymask=cv2.inRange(img_hsv,Ymin,Ymax)
        Ycnt=cv2.countNonZero(Ymask)
        
        if Gcnt>50:
            color="G"
        elif Ycnt>100:
            color="Y"
        elif Rcnt>100:
            color="R"
        else:
            img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
            thresh=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)[1]
            _,contour,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contour:
                color="W"
            else:
                color="N"
    elif model=="TESLA":
        Bmask=cv2.inRange(img_hsv,Bmin,Bmax)
        Bcnt=cv2.countNonZero(Bmask)
        if Bcnt:
            color="B"
        else:
            img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
            thresh=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)[1]
            _,contour,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contour:
                color="W"
            else:
                color="N"
        
    return color

# mark wrong frame
def doubleClick(event,x,y,flags,para):
    global video,textname,img
    ms=video.get(0)
    fr=video.get(1)
    datetime=dt.timedelta(milliseconds=ms)      
    if event==cv2.EVENT_LBUTTONDBLCLK:
        mb.showinfo("alert","wrong frame marked!")
#         print datetime
        cv2.imwrite(str(fr)+".jpg",img)
        with open(textname,"a") as text:
            text.write(str(datetime)+" "+str(fr)+"\n")

framevariable=tk.IntVar()
frameEntered=ttk.Entry(tab1,textvariable=framevariable)
frameEntered.grid(column=0,row=6,pady=20)

hooFunc=["ACC","LSA","PLUS"]

def batchProcess():
    global videoPath,video,model,i,textname,img
    locDict={}
    ans=mb.askquestion("warning","Are you sure to continue?")
    if ans=="yes":
        config.read("parameter.ini")
#         convert unicode to dict
        for func,preloc in config.items(model):
            func=func.encode("utf-8")
            func=str.upper(func)
            filtered=filter(lefts,preloc.encode("utf-8"))
            loclist=filtered.split(",")
            locDict[func]=((int(loclist[0]),int(loclist[1])),(int(loclist[2]),int(loclist[3])))

        path=os.path.dirname(videoPath)
                
        cv2.namedWindow("batch processing:Press ESC to close")
        cv2.setMouseCallback("batch processing:Press ESC to close", doubleClick, 0)
        
        for itervideo in glob.glob(os.path.join(path, '*.mp4')):
            csvname=os.path.splitext(itervideo)[0]+".csv"
            csvfile=open(csvname,"ab")
            csvwriter=csv.writer(csvfile)
            
            csvwriter.writerow([str(func) for func in locDict])
            textname=os.path.splitext(itervideo)[0]+".txt"
            video=cv2.VideoCapture(itervideo)
            total_frames=video.get(7)
            
            t=int(total_frames)
#             fps=video.get(5)
            i=0
    
            frame=framevariable.get()
            video.set(1,frame)
            
            while True:
           
                ret,img=video.read()
#                 print ret
                if ret==True:
                    i+=1
                    statusDict={}
                    if model=="HOO60":
#                       detect icon
                        max_score={}
                        image_crop=img[locDict["ALL"][0][1]:locDict["ALL"][1][1],locDict["ALL"][0][0]:locDict["ALL"][1][0]]
                        image_gray=cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
                        high_canny_i,_=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        low_canny_i=0.5 * high_canny_i
                        image_edge=cv2.Canny(image_gray,low_canny_i,high_canny_i)
                        for func in hooFunc:
                            templatePath=func+".jpg"
                            template=cv2.imread(templatePath)
                            template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
                            high_canny_t,_=cv2.threshold(template_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            low_canny_t=0.5 * high_canny_t
                            template_edge=cv2.Canny(template_gray,low_canny_t,high_canny_t)
                            res=cv2.matchTemplate(image_edge,template_edge,cv2.TM_CCORR_NORMED)
                            _,max_val,_,_=cv2.minMaxLoc(res)
                            max_score[func]=max_val
#                         print max_score
                        max_func=max(max_score,key=max_score.get)
#                       detect icon color
                        colori=detectColor(image_crop,model)
#                       detect wheel color
                        wheel_crop=img[locDict["WHEEL"][0][1]:locDict["WHEEL"][1][1],locDict["WHEEL"][0][0]:locDict["WHEEL"][1][0]]
                        color=detectColor(wheel_crop,model)
#                       mark icon_area
                        cv2.rectangle(img,locDict["ALL"][0],locDict["ALL"][1],(0,0,255),2)
                        cv2.putText(img,max_func+" "+colori,locDict["ALL"][1],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA,False) 
#                       mark wheel_area
                        cv2.rectangle(img,locDict["WHEEL"][0],locDict["WHEEL"][1],(0,0,255),2)
                        cv2.putText(img,color,locDict["WHEEL"][1],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA,False) 
                        csvwriter.writerow([max_func+" "+colori,color])
                    else:               
                        for func in locDict:
                            img_crop=img[locDict[func][0][1]:locDict[func][1][1],locDict[func][0][0]:locDict[func][1][0]]
                            color=detectColor(img_crop,model)
    #                         print func,color
                            statusDict[func]=colorDict[model][func][color]
                            cv2.rectangle(img,locDict[func][0],locDict[func][1],(0,0,255),2)
                            cv2.putText(img,func+" "+statusDict[func],locDict[func][1],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA,False)
                            csvwriter.writerow([statusDict[func] for func in statusDict])
                    progress="{0:.2f}%".format(float(i)/t * 100)
                    cv2.putText(img,progress,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA,False)
                    cv2.imshow("batch processing:Press ESC to close",img)
                    cv2.waitKey(1)

                else:
                    break
            
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            video.release()
            cv2.destroyAllWindows()
            csvfile.close()
#         pass
    else:
        pass
    

batchButton=ttk.Button(tab1,text="Start batch process",width=30,command=batchProcess)
batchButton.grid(column=1,row=6,columnspan=4,pady=20)

def loadCsv():
    global csvPath
    csvPath = tkFileDialog.askopenfilename()
    
#     print df.head()

loadButton=ttk.Button(tab2,text="load",command=loadCsv)
loadButton.grid(column=0,row=0,padx=5,pady=10)

def statsummary():
    global df,model
    model=modelvariable.get()
    df=pd.read_csv(csvPath)
    if model=="HOO60":
        def label_status(row):
            if row["ALL"]=="ACC G":
                return "ACC ON"
            elif row['ALL']=='ACC W':
                return 'ACC STANDBY'
            elif "PLUS" in row['ALL']:
                if row['WHEEL']=="R":
                    return "HOO60 TOR"
                else:
                    return "HOO60 ON"
            elif 'LSA' in row['ALL']:
                if row['WHEEL']=='G':
                    return 'LSA ON ACC ON'
                elif row['WHEEL']=='Y':
                    return 'HOR LSA ON ACC ON'
                elif row['WHEEL']=="W":
                    if row['ALL']=='LSA G':
                        return 'LSA STANDBY ACC ON'
                    elif row['ALL']=='LSA W':
                        return 'LSA STANDBY ACC STANDBY'
                elif row['WHEEL']=='R':
                    if row['ALL']=='LSA G':
                        return 'TOR LSA STANDBY ACC ON'
                    elif row['ALL']=='LSA W':
                        return 'TOR LSA STANDBY ACC STANDBY'
                

        df['status']=df.apply(label_status,axis=1)
        dfc=100.*df['status'].value_counts(normalize=True)
        ttk.Label(tab2,text=dfc).grid(column=0,row=1,columnspan=3,padx=10)
    else: 
        cnt=0
        for column in df:
            dfc=100.*df[column].value_counts(normalize=True)
            ttk.Label(tab2,text=dfc).grid(column=cnt,row=1,padx=10)
            cnt+=1


plotButton=ttk.Button(tab2,text="summary",command=statsummary)
plotButton.grid(column=1,row=0,padx=5,pady=10)

def exportsummary():
    basename=os.path.basename(csvPath)
    tocsvPath=os.path.dirname(csvPath)+'/exports/'+os.path.splitext(basename)[0]+".xlsx"
#     print tocsvPath
    writer=pd.ExcelWriter(tocsvPath,engine='openpyxl')
    for column in df:
        dfc=100.*df[column].value_counts(normalize=True)
        dfc.to_excel(writer,sheet_name=column)
        writer.save()
    mb.showinfo("alert","exporting completed!")

exportButton=ttk.Button(tab2,text="export",command=exportsummary)
exportButton.grid(column=2,row=0,padx=5,pady=10)

def close():
    root.destroy()
    
root.protocol("WM_DELETE_WINDOW",close)

root.mainloop()

