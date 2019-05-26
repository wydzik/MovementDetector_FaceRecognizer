# USAGE
# python recognize_video.py  

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from email.message import EmailMessage
import Contact_details as cd
import numpy as np
import smtplib
import imghdr
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
img_counter = 0; #licznik do numerowania snapshotów
msg = EmailMessage()
msg['Subject'] = 'Nieautoryzowany dostep do urzadzenia!'
msg['From'] = cd.EMAIL_SENDER
msg['To'] = cd.EMAIL_RECEIVER
msg.set_content('Kamera wykryla ruch przy Twoim stanowisku, w zalaczniku przesylamy Snapshota z tej sesji.\nZachowaj ostroznosc i zabezpiecz stanowisko.')
localtime = time.strftime("%d_%m_%Y__%H_%M_%S", time.localtime())

czy_rozpoznano = 1
czas_od_wyslania = 1
start = 0

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	k = cv2.waitKey(1)

	if k % 256 == 32: # jeżeli kliknięty klawisz to spacja
			localtime = time.strftime("%d_%m_%Y__%H_%M_%S", time.localtime())
			img_name = "Snapshot_{}_".format(img_counter) + localtime + ".jpg"  # nazwanie naszego snapshota
			#img_name = "Snapshot_{}.jpg".format(img_counter)
			#print(img_name)
			# format nie ma znaczenia - moze byc dowolny
			cv2.imwrite(img_name, frame) # zapisanie
			print("{} written!".format(img_name))  # potwierdza że Snapshot został wykonany i zapisany
			img_counter += 1  # licznik do nazwy



	if k % 256 == 27:
		# dodanie załącznika
		new_string = str(img_counter-1)+"_"+str(localtime)
		if img_counter > 0:
			with open("Snapshot_{}.jpg".format(new_string), 'rb') as f:
				file_data = f.read()
				file_type = imghdr.what(f.name)
				file_name = f.name

			msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

			# wysyłanie wiaomości
			# with smtplib.SMTP('smtp.gmail.com', 465) as smtp:

			with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
				smtp.ehlo()
				smtp.starttls()
				smtp.ehlo()
				smtp.login(cd.EMAIL_SENDER, cd.PASSWORD)
				smtp.send_message(msg)
				smtp.quit()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]


			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("MovementFetector_FaceRecognizer", frame)
	key = cv2.waitKey(1) & 0xFF




	if name != "unknown" :
		czy_rozpoznano = 1

	if name == "unknown":
		if start == 0 :
			start = time.time()

	if time.time() - start > 30  and start > 0 :
			czy_rozpoznano = 0


	if czas_od_wyslania == 0 :
		if czy_rozpoznano == 0 :
			print("Wysylam wiadomosc")
			czas_od_wyslania = time.time()

			localtime = time.strftime("%d_%m_%Y__%H_%M_%S", time.localtime())
			img_name = "Snapshot_{}_".format(img_counter) + localtime + ".jpg"  # nazwanie naszego snapshota
			# img_name = "Snapshot_{}.jpg".format(img_counter)
			# print(img_name)
			# format nie ma znaczenia - moze byc dowolny
			cv2.imwrite(img_name, frame)  # zapisanie
			print("{} written!".format(img_name))  # potwierdza że Snapshot został wykonany i zapisany
			img_counter += 1  # licznik do nazwy
			new_string = str(img_counter - 1) + "_" + str(localtime)
			if img_counter > 0:
				with open("Snapshot_{}.jpg".format(new_string), 'rb') as f:
					file_data = f.read()
					file_type = imghdr.what(f.name)
					file_name = f.name

				msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

				# wysyłanie wiaomości
				# with smtplib.SMTP('smtp.gmail.com', 465) as smtp:

				with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
					smtp.ehlo()
					smtp.starttls()
					smtp.ehlo()
					smtp.login(cd.EMAIL_SENDER, cd.PASSWORD)
					smtp.send_message(msg)
					smtp.quit()

	if time.time() - czas_od_wyslania > 100000:
		czas_od_wyslania = 0;

	# if the `q` key was pressed, break from the loop
	if k == ord("q"):
		break




# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()