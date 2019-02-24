# USAGE
# python fps_demo.py
# python fps_demo.py --display 1 --file \path\to\file -roi
# Options:
#	--display 1 or 0
#		weither to show the mask and orginal memory
#	--file <path>
#		path to a file to run againist instead of a using the video0 webcam
#	--roi
#		this option is to set an area of intrest or load the myRoi without
#		this option it will read the whole image




# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import time
import datetime
import signal
import numpy as np
from selectRoi import RoiInit
import os
import re

###########################################################
#
# Variable to be set/changed
#
###########################################################
roi = False #set to true to make roi the default mode
frame_history=60 #set the history for the mask level


###########################################################
interrupted = False #set it so crtl+c will exit the loop not kill whole script

#################################################
# construct the argument parse and parse the arguments
#################################################
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,
	help="Whether or not frames should be displayed")
ap.add_argument("-f", "--file", default='cam',
	help="File path to video file")
ap.add_argument("-area", "--min-area", type=int, default=1, help="minimum area size")
ap.add_argument("-roi", "--roi", action='store_true', help="use if you want to use roi or not")
args = vars(ap.parse_args())
##############################################################
##############################################################

############################################################
#set it so crtl+c will exit the loop not kill whole script
############################################################
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
interrupted = False 
################################################################
################################################################


################################################################
#prep the multithreaded loading of frames from webcam/file
################################################################



# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
#This checks if a file is to be opened if not -f is used then
# it will set the video source to webcam on port 0
if args['file'] == 'cam':
	source =WebcamVideoStream(src=0).start()
else:
	source = WebcamVideoStream(args['file']).start()

vs = source
time.sleep(2.0)

frame = vs.read()
##############################################################
################################################################

#############################################################
#read the RoI file if there is none then run selectRoi app
#############################################################

if args["roi"]==1:
	# Selected Coordinates to be saved into myRoi
	refPt = []
	print("[INFO] Using area of intrest cropping")

	def is_non_zero_file(fpath):
		return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


	def setRoi(frame):
		global refPt
		if is_non_zero_file("myRoi"):
			print("[INFO] Found myRoi file to use")
			with open("myRoi") as f:
				for line in f:
					x1, y1, x2, y2 = map(int, re.match(r"\((.*), (.*)\) \((.*), (.*)\)", line).groups())
					print("[INFO] Found Roi using following cordinates", x1, y1,"to", x2, y2)
					refPt = [(x1, y1), (x2, y2)]

		else:
			print("[INFO] No myRoi file found please creating new one")
			refPt = RoiInit(frame)
			with open('myRoi', 'a') as fi:
				fi.write(str(refPt[0])+' '+str(refPt[1]))        
  
#load the myRoi and select if non avalialable 
if frame is not None:
	if args["roi"]==True:
		setRoi(frame)
##########################################################
##########################################################		


fps = FPS().start() #start counter for fps counter

# initialize the first frame in the video stream
first_frame = vs.read()
#create a maask settings based on setting below
subtractor = cv2.createBackgroundSubtractorMOG2(history=frame_history, varThreshold=40, detectShadows=True)
 
print("[INFO] Starting tracking.....")
# loop over some frames...this time using the threaded stream


while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of Imagesize variable
	
	frame = cv2.UMat(vs.read())
	
	if args["roi"]==True:
		#change frame to be the referance points in myRoi
		frame=frame.get()
		frame = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		frame=cv2.UMat(frame)
		
	# update the FPS counter
	fps.update()
	
	#break the loop on ctrl+c interrupt
	if interrupted:
		print("exit requested")
		break
		
	# apply the initial MOG2 setting to see frame.
	thresh = subtractor.apply(cv2.UMat(frame))
	
	contours, hier = cv2.findContours(cv2.UMat(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# with each contour, draw boundingRect in green
	# a minAreaRect in red and
	# a minEnclosingCircle in blue
	

	for c in contours:
		# get the bounding rect
		x, y, w, h = cv2.boundingRect(c)
		if(cv2.contourArea(c) > 1):
			# draw a green rectangle to visualize the bounding rect
			cv2.rectangle(cv2.UMat(frame), (x, y), (x+w, y+h), (0, 255, 0), 2)
		
			# get the min area rect
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
		
			# convert all coordinates floating point values to int
			box = np.int0(box)
			
			# draw a red 'nghien'/rotating rectangle
			cv2.drawContours(cv2.UMat(frame), [box], 0, (0, 0, 255))
			print(cv2.contourArea(c))

			#cv2.drawContours(frame, contours, -1, (255, 255, 0), 1)
	
		
		
	
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", cv2.UMat(frame))
		cv2.imshow("mask", cv2.UMat(thresh))

		k = cv2.waitKey(1) & 0xFF
		if k==27:    # Esc key to stop
			break




# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
vs.stop()




