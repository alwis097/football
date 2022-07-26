
import os
# from re import X
# from tkinter import Frame

from flask import Flask, flash, jsonify, request, redirect, url_for, render_template,Response
# from matplotlib import colors
# from nbformat import read
# from pyrsistent import b
from werkzeug.utils import secure_filename
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import face_recognition
import numpy as np
import json


from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route('/',methods = ['GET','POST'])
def upload_video():
    global text1
    global text2
    print("wrong")
    if request.method == "POST":
        t1 = request.form['text1']
        t2 = request.form['text2']
        text1 = int(t1)
        text2 = int(t2)
        print(type(text1))
        print(text2)
        file = request.files['file']

        if file:
            filename = secure_filename(request.files['file'].filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		# filename = secure_filename(request.files['file'].filename)
		# request.files['file'].save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("upload.html",filename=filename)







@app.route('/video/<filename>',methods=['GET'])
def video(filename):

    Video_FILE = "static/uploads/"+filename
    print(filename)
    vidcap = cv2.VideoCapture(Video_FILE,0)


    def gen_frames(text3,text4):  
            # success,frame = vidcap.read()
            count = 0
            # success = True
            idx = 0

            count1 = 0
            #Read the video frame by frame
            while True:
                count1 = count1 + 1
                # print(count1)
                success,frame = vidcap.read()
                # count = 0
                # success = True
                # idx = 0
                # ret, image = vidcap.read()
                if not success:
                    break
                else:
                    # cv2.COLOR_BGR2HSV
                #converting into hsv image
                #hsv = cv2.imread(sys.path[0]+"/image.jpg", 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    #cv2.imshow('Image', image)
                    #cv2.imshow('HSV', hsv)

                    #green range
                    lower_green = np.array([40,40, 40])
                    upper_green = np.array([70, 255, 255])
                    #blue range
                    lower_blue = np.array([110,50,50])
                    upper_blue = np.array([130,255,255])

                    #yellow  
                    lower_yellow = np.array([181,166,66])
                    upper_yellow = np.array([255,255,240])

                    #Red range
                    lower_red = np.array([0,31,255])
                    upper_red = np.array([176,255,255])

                    #white range
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([0,0,255])


                    lower_black = np.array([0,0,0])
                    upper_black = np.array([170,150,50])


                    #Define a mask ranging from lower to uppper
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    #cv2.imshow('Mask', mask)
                    #Do masking
                    res = cv2.bitwise_and(frame, frame, mask=mask)
                    #convert to hsv to gray
                    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
                    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

                    #Defining a kernel to do morphological operation in threshold image to 
                    #get better output.
                    kernel = np.ones((13,13),np.uint8)
                    thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    

                    #find contours in threshold image     
                    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    

                    prev = 0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    for c in contours:
                        x,y,w,h = cv2.boundingRect(c)
                        
                        #Detect players
                        if(h>=(1.5)*w):
                            if(w>15 and h>= 15):
                                idx = idx+1
                                player_img = frame[y:y+h,x:x+w]
                                player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)


                                #If player has blue jersy
                                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                                nzCount = cv2.countNonZero(res1)


                                #If player has red jersy
                                mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                                nzCountred = cv2.countNonZero(res2)

                                 #If player has white jersy
                                mask3 = cv2.inRange(player_hsv, lower_white, upper_white)
                                res3 = cv2.bitwise_and(player_img, player_img, mask=mask3)
                                res3 = cv2.cvtColor(res3,cv2.COLOR_HSV2BGR)
                                res3 = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
                                nzCountred1 = cv2.countNonZero(res3)


                                mask4 = cv2.inRange(player_hsv, lower_yellow, upper_yellow)
                                res4 = cv2.bitwise_and(player_img, player_img, mask=mask4)
                                res4 = cv2.cvtColor(res4,cv2.COLOR_HSV2BGR)
                                res4 = cv2.cvtColor(res4,cv2.COLOR_BGR2GRAY)
                                nzCountred2 = cv2.countNonZero(res4)

                                mask5 = cv2.inRange(player_hsv, lower_black, upper_black)
                                res5 = cv2.bitwise_and(player_img, player_img, mask=mask5)
                                res5 = cv2.cvtColor(res5,cv2.COLOR_HSV2BGR)
                                res5 = cv2.cvtColor(res5,cv2.COLOR_BGR2GRAY)
                                nzCountred3 = cv2.countNonZero(res5)

                                
                                if (text3 == 1 and text4 == 2) or (text4 == 1 and text1 == 2):

                                    if(nzCount >= 20):
 
                                        cv2.putText(frame, 'Blue', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                                        

                                        a =[]
                                        b =[]
                                        a.append(x)
                                        a.append(y)
                                        b.append(a)
                                        clr1 = "red"
                                        clr2 = "Blue"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,0,0", 'RGB_color_code_for_color_1': "0,0,255"}
                                        
                                        team1['frame x,y values_color1'] = {'frame x,y values': b}
                                        team1['Last_countNonZero_values_color1'] = {'countNonZero_values_color1':nzCount }
                                        if(nzCountred>=20):
                                            team1['Last_countNonZero_values_color2'] = {'countNonZero_values_color2':nzCountred }
                                        else:
                                            pass
                                        with open('red_blue.json', 'w') as f:
                                            json.dump(team1, f)

                                    else:
                                        pass
                                    if(nzCountred>=20):
                                        #Mark red jersy players as belgium
                                        cv2.putText(frame, 'Red', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                                    else:
                                        pass

                                elif (text3 == 1 and text4 == 3) or (text4 == 1 and text3 == 3):
                                        if(nzCountred1>=20):
                                            #Mark White jersy players as belgium
                                            cv2.putText(frame, 'White', (x-2, y-2), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                                            a =[]
                                            b =[]
                      
                                            a.append(x)
                                            a.append(y)

                                            b.append(a)
                                            clr1 = "White"
                                            clr2 = "Red"
                                            team1 ={}
                                            team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                            team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,255,255", 'RGB_color_code_for_color_1': "0,0,255"}
                                            team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                            team1['Last_countNonZero_values_color1'] = {'Last_countNonZero_values_color1':nzCountred1 }
                                            if(nzCountred>=20):
                                                team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred }
                                            else:
                                                pass
                                            with open('red_white.json', 'w') as f:
                                                json.dump(team1, f)
                                        else:
                                            pass
                                        if(nzCountred>=20):
                                            #Mark red jersy players as belgium
                                            cv2.putText(frame, 'Red', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                                        else:
                                            pass
                                elif (text3 == 1 and text4 == 4) or (text4 == 1 and text3 == 4):
                                    if(nzCountred2>=20):
                                        #Mark yellow jersy players as belgium
                                        cv2.putText(frame, 'yellow', (x-2, y-2), font, 0.8, (252, 245, 95), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(252, 245, 95),3)
                                        a =[]
                                        b =[]
 
                                        a.append(x)
                                        a.append(y)

                                        b.append(a)
                                        clr1 = "Yellow"
                                        clr2 = "Red"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "252, 245, 95", 'RGB_color_code_for_color_1': "0,0,255"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['Last_countNonZero_values_color1'] = {'Last_countNonZero_values_color1':nzCountred2 }
                                        if(nzCountred>=20):
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred }
                                        else:
                                                pass
                                        with open('red_yellow.json', 'w') as f:
                                            json.dump(team1, f)
                                    else:
                                        pass

                                    if(nzCountred>=20):
                                        #Mark red jersy players as belgium
                                        cv2.putText(frame, 'Red', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                                    else:
                                        pass

                                elif (text3 == 1 and text4 == 5) or (text4 == 1 and text3 == 5):
                                    if(nzCountred3>=20):
                                        #Mark black jersy players as belgium

                                        cv2.putText(frame, 'Black', (x-2, y-2), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
                                        a =[]
                                        b =[]
          
                                        a.append(x)
                                        a.append(y)

                                        b.append(a)
                                        clr1 = "Black"
                                        clr2 = "Red"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "0,0,0", 'RGB_color_code_for_color_1': "0,0,255"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred3 }
                                        if(nzCountred>=20):
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred }
                                        else:
                                                pass
                                        with open('red_black.json', 'w') as f:
                                            json.dump(team1, f)
                                    else:
                                        pass

                                    if(nzCountred>=20):
                                        #Mark black jersy players as belgium
                                        cv2.putText(frame, 'Red', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

                                    else:
                                        pass  
                                    # second level
                                elif (text3 == 2 and text4 == 3) or (text4 == 2 and text3 == 3):
                                        if(nzCountred1>=20):
                                            #Mark White jersy players as belgium
                                            cv2.putText(frame, 'White', (x-2, y-2), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                                            a =[]
                                            b =[]

                                            a.append(x)
                                            a.append(y)

                                            b.append(a)
                                            clr1 = "White"
                                            clr2 = "Blue"
                                            team1 ={}
                                            team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                            team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,255,255", 'RGB_color_code_for_color_1': "255,0,0"}
                                            team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred1 }
                                            if(nzCount>=20):
                                                team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCount }
                                            else:
                                                    pass
                                            with open('White_Blue.json', 'w') as f:
                                                json.dump(team1, f)
                                        else:
                                            pass
                                        if(nzCount >= 20):
                                            # print("hello")
                                            #Mark Blue jersy players as france
                                            cv2.putText(frame, 'Blue', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                                        else:
                                            pass
                                elif (text3 == 2 and text4 == 4) or (text4 == 2 and text3 == 4):
                                    if(nzCount >= 20):
                                        # print("hello")
                                        #Mark Blue jersy players as france
                                        cv2.putText(frame, 'Blue', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                                        a =[]
                                        b =[]

                                        a.append(x)
                                        a.append(y)
                                        b.append(a)
                                        clr1 = "Yellow"
                                        clr2 = "Blue"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,0,0", 'RGB_color_code_for_color_1': "252, 245, 95"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred1 }
                                        if(nzCountred2>=20):
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred2 }
                                        else:
                                                pass
                                        with open('Yellow_Blue.json', 'w') as f:
                                            json.dump(team1, f)
                                    else:
                                        pass
                                    if(nzCountred2>=20):
                                        #Mark yellow jersy players as belgium
                                        cv2.putText(frame, 'yellow', (x-2, y-2), font, 0.8, (252, 245, 95), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(252, 245, 95),3)
                                    else:
                                        pass

                                elif (text3 == 2 and text4 == 5) or (text4 == 2 and text3 == 5):
                                    if(nzCount >= 20):
                                        # print("hello")
                                        #Mark Blue jersy players as france
                                        cv2.putText(frame, 'Blue', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                                        a =[]
                                        b =[]

                                        a.append(x)
                                        a.append(y)
                                        b.append(a)
                                        clr1 = "Black"
                                        clr2 = "Blue"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,0,0", 'RGB_color_code_for_color_1': "0,0,0"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCount }
                                        if(nzCountred3>=20):
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred3 }
                                        else:
                                                pass
                                        with open('Black_Blue.json', 'w') as f:
                                            json.dump(team1, f)
                                    else:
                                        pass
                                    if(nzCountred3>=20):
                                        #Mark black jersy players as belgium
                                        cv2.putText(frame, 'Black', (x-2, y-2), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
                                    else:
                                        pass    

                                #3rd level
                                elif (text3 == 3 and text4 == 4) or (text4 == 2 and text3 == 4):
                                    if(nzCountred1>=20):
                                         #Mark White jersy players as belgium
                                        cv2.putText(frame, 'White', (x-2, y-2), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                                        a =[]
                                        b =[]
      
                                        a.append(x)
                                        a.append(y)
                                        b.append(a)
                                        clr1 = "White"
                                        clr2 = "Yellow"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,255,255", 'RGB_color_code_for_color_1': "252, 245, 95"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['countNonZero_values_color2'] = {'countNonZero_values_color2':nzCountred1}
                                        if(nzCountred2>=20):
                                            team1['countNonZero_values_color2'] = {'countNonZero_values_color2':nzCountred2 }
                                        else:
                                                pass
                                        with open('Yellow_White.json', 'w') as f:
                                            json.dump(team1, f)                                        
                                    else:
                                        pass
                                    if(nzCountred2>=20):
                                        #Mark yellow jersy players as belgium
                                        cv2.putText(frame, 'yellow', (x-2, y-2), font, 0.8, (252, 245, 95), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(252, 245, 95),3)
                                    else:
                                        pass

                                elif (text3 == 3 and text4 == 5) or (text4 == 3 and text3 == 5):
                                    if(nzCountred1>=20):
                                         #Mark White jersy players as belgium
                                        cv2.putText(frame, 'White', (x-2, y-2), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                                        a =[]
                                        b =[]

                                        a.append(x)
                                        a.append(y)
                                        b.append(a)
                                        clr1 = "White"
                                        clr2 = "Black"
                                        team1 ={}
                                        team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                        team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "255,255,255", 'RGB_color_code_for_color_1': "0,0,0"}
                                        team1['frame x,y values_color1'] = {'frame x,y values_color1': b}
                                        team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred1}
                                        if(nzCountred3>=20):
                                            team1['Last_countNonZero_values_color2'] = {'Last_countNonZero_values_color2':nzCountred3}
                                        else:
                                                pass
                                        with open('Black_White.json', 'w') as f:
                                            json.dump(team1, f) 
                                    else:
                                        pass
                                    if(nzCountred3>=20):
                                        #Mark black jersy players as belgium
                                        cv2.putText(frame, 'Black', (x-2, y-2), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
                                    else:
                                        pass 
                                           
                                # 4th level 
                                elif (text3 == 4 and text4 == 5) or (text4 == 4 and text3 == 5):
                                        if(nzCountred2>=20):
                                            #Mark yellow jersy players as belgium
                                            cv2.putText(frame, 'yellow', (x-2, y-2), font, 0.8, (252, 245, 95), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(252, 245, 95),3)
                                            a =[]
                                            b =[]

                                            a.append(x)
                                            a.append(y)
                                            b.append(a)
                                            clr1 = "Yellow"
                                            clr2 = "Black"
                                            team1 ={}
                                            team1['Colors'] = {'team_1_color': clr1, 'team_2_color': clr2}
                                            team1['rgb_code_for_colors'] = {'RGB_color_code_for_color_1': "252, 245, 95", 'RGB_color_code_for_color_1': "0,0,0"}
                                            team1['frame x,y values_color1_color1'] = {'frame x,y values_color1': b}
                                            team1['countNonZero_values_color2'] = {'countNonZero_values_color2':nzCountred2}
                                            if(nzCountred3>=20):
                                                team1['countNonZero_values_color2'] = {'countNonZero_values_color2':nzCountred3}
                                            else:
                                                    pass
                                            with open('Yellow_Black.json', 'w') as f:
                                                json.dump(team1, f) 
                                        else:
                                            pass
                                        if(nzCountred3>=20):
                                            #Mark black jersy players as belgium
                                            cv2.putText(frame, 'Black', (x-2, y-2), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
                                        else:
                                            pass 
                                else:
                                    x = "Choose two differnet colors"
                                    return render_template("upload.html",x=x)
                                
                                # print("nzCount")
                                # print(nzCount)
                                # print("x")
                                # print(x)
                                # print("y")
                                # print(y)
                                team = {}



                                                                
                        # if((h>=1 and w>=1) and (h<=30 and w<=30)):
                        #     player_img = frame[y:y+h,x:x+w]
                        
                        #     player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                        #     #white ball  detection
                        #     mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                        #     res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                        #     res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                        #     res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                        #     nzCount = cv2.countNonZero(res1)
                    

                        #     if(nzCount >= 3):
                        #         # detect football
                        #         cv2.putText(frame, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

                    cv2.imwrite("./Cropped/frame%d.jpg" % count, res)
                    # print ('Read a new frame: '), success     # save frame as JPEG file	
                    # count += 1
                # cv2.imshow('HSV', hsv)
                    cv2.imshow('Match Detection',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    success,image = vidcap.read()
                
             
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                # print(frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(text1,text2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display/<filename>')
def display_video(filename):

	return redirect(url_for('static',filename='uploads/' + filename), code=301)


x=[]
@app.route('/red_blue',methods=['GET'])
def js1():
    f= open('red_blue.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})




@app.route('/red_white',methods=['GET'])
def js2():
    f= open('red_white.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})

@app.route('/red_yellow',methods=['GET'])
def js3():
    f= open('red_yellow.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})



@app.route('/red_black',methods=['GET'])
def js5():
    f= open('red_black.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})








@app.route('/White_Blue',methods=['GET'])
def js6():
    f= open('White_Blue.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})



@app.route('/Yellow_Blue',methods=['GET'])
def js10():
    f= open('Yellow_Blue.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})



@app.route('/Black_Blue',methods=['GET'])
def js7():
    f= open('Black_Blue.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})


@app.route('/Yellow_White',methods=['GET'])
def js8():
    f= open('Yellow_White.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})

@app.route('/Black_White',methods=['GET'])
def js9():
    f= open('Black_White.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})




@app.route('/Yellow_Black',methods=['GET'])
def js4():
    f= open('Yellow_Black.json')
    data = json.load(f)
    for i in data['Colors']:
        x.append(i)
        print(i)
    f.close()
    return jsonify({"results " :data})






import os

from flask import Flask, flash, request, redirect, url_for, render_template,Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import face_recognition
import numpy as np
import cv2
import numpy as np
from PIL import ImageGrab

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/upload1')
def upload_form1():
	return render_template('upload1.html')


@app.route('/upload1', methods=['POST'])
def upload_video1():
		
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)
		else:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#print('upload_video filename: ' + filename)
			# Video_FILE = '/uploads/' + filename
			Video_FILE = "static/uploads/"+filename

			return render_template('upload1.html', filename=filename)          

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

@app.route('/upload1/video1/<filename>')
def video1(filename):
    Video_FILE = "static/uploads/"+filename
    # cap = cv2.VideoCapture(Video_FILE)
    # previous_frame = None


    def gen_frames():  
        # Video_FILE = "static/uploads/"+filename
        cap = cv2.VideoCapture(Video_FILE,0)
        previous_frame = None
        while True:

            # 1. Load image; convert to RGB
            ret, frame = cap.read()
            frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)


            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # 2. Calculate the difference
            if (previous_frame is None):
            # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # 3. Set previous frame and continue if there is None
            if (previous_frame is None):
            # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

            # 6. Find and optionally draw contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # Comment below to stop drawing contours
            cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # Uncomment 6 lines below to stop drawing rectangles
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    # too small: skip!
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Motion detector', frame)
            # cv2.imwrite("./Cropped/frame%d.jpg" % frame)



        # Cleanup
        # cv2.destroyAllWindows()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
                        # print(frame)
            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload1/display/<filename>')
def display_video1(filename):

	return redirect(url_for('static',filename='uploads/' + filename), code=301)


app.run(debug=True)