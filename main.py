import cv2
import datetime
from threading import Thread
import threading
import imutils
from imutils.video import VideoStream
import time
from astral import LocationInfo
from astral.sun import sun
from astral.geocoder import database, lookup
import numpy as np
from flask import Flask
#from flask import Response
#from flask import Flask
#from flask import render_template

db = database()
city = lookup("Los Angeles", db)

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

#def update_light_status():
#	s = sun(city.observer, date=datetime.date.today(), tzinfo=city.timezone)
#	dawn = s['dawn'].replace(tzinfo=None)
#	dusk = s['dusk'].replace(tzinfo=None)
#	current = datetime.datetime.now()
#	return not dawn<current<dusk

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

class FPS:
	def __init__(self):
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		self._end = datetime.datetime.now()
	def update(self):
		self._numFrames += 1
	def elapsed(self):
		return (self._end - self._start).total_seconds()
	def fps(self):
		return self._numFrames / self.elapsed()

class WebcamVideoStream:
	def __init__(self, src=0):
		self.first_frame = None
		self.last_seen_at = 0.0
		self.min_area = 700
		self.threshold = 100
		self.camera_val = src
		self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,2592)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
		self.recording = False
		frame_width = int(self.stream.get(3))
		frame_height = int(self.stream.get(4))
		self.size = (frame_width, frame_height)
		self.detect_x, self.detect_y, self.detect_w, self.detect_h = 0, 0, frame_width-1, frame_height-1
		print(self.detect_x, self.detect_y, self.detect_w, self.detect_h)
		print(self.size)
		self.writer = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, self.size)
		(self.grabbed, self.frame) = self.stream.read()
		self.writer.write(self.frame)
		self.stopped = False
	def start(self):
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()
			self.track_motion()
			self.add_text()
			if self.recording:
				self.writer.write(self.frame)
			
	def read(self):
		return self.frame
	def stop(self):
		self.stopped = True
	def add_text(self):
		timestamp = datetime.datetime.now()
		timestamp_string = timestamp.strftime("%d/%m/%Y %H:%M:%S")
		self.frame = cv2.putText(self.frame, timestamp_string, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), thickness=3)
		self.frame = cv2.putText(self.frame, ('Camera #: ' + str(vs.camera_val)), (10,170), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), thickness=3)
		if self.recording:
			self.frame = cv2.putText(self.frame, 'Recording', (2000, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=3)
	def track_motion(self):
		gray_frame = cv2.cvtColor(self.frame[self.detect_y:self.detect_y+self.detect_h, self.detect_x:self.detect_x+self.detect_w], cv2.COLOR_BGR2GRAY)
		blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
		if self.first_frame is None:
			print("first frame captured")
			self.first_frame = blurred_frame
		frame_difference = cv2.absdiff(self.first_frame, blurred_frame)
		thresh = cv2.threshold(frame_difference, self.threshold, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		found_person = False
		for contour in contours:
			if cv2.contourArea(contour) < self.min_area:
				continue
			found_person = True
			self.recording = True
			self.last_seen_at = time.time()
			#(x, y, w, h) = cv2.boundingRect(contour)
			#cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		if (not found_person) and (time.time()-self.last_seen_at > 10):
			self.recording = False


print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
time.sleep(10)

light_status = False
checked_light_at = 0
"""
while True:
	if time.time() - checked_light_at > 60:
		light_status = update_light_status()
		checked_light_at = time.time()
	cv2.imshow("Frame", vs.frame)
	#cv2.imshow("BW", vs.first_frame)
	with lock:
		outputFrame = vs.read().copy()
	if cv2.waitKey(1) & 0xFF == ord('s'):
			break
	if cv2.waitKey(1) & 0xFF == ord('a'):
			vs.recording = not vs.recording
			print(vs.recording)
	fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()"""

def stream_video():
	global vs, outputFrame, lock, light_status, checked_light_at

	while True:
		if time.time() - checked_light_at > 60:
			light_status = update_light_status()
			checked_light_at = time.time()
		with lock:
			outputFrame = vs.read().copy()


@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	# start a thread that will perform motion detection
	t = threading.Thread(target=stream_video)
	t.daemon = True
	t.start()
	# start the flask app
# release the video stream pointer
vs.stop()