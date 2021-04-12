from absl import flags
import sys
FLAGS = flags.FLAGS #commands
FLAGS(sys.argv)

import time #to calculate the fps
import numpy as np
import cv2 #open cv to visualize de tracking
import matplotlib.pyplot as plt #call a map

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5 #that's consider if the object is the same or not
nn_budget = None #used to creat librearies and store features vector, is 100 instead of None
nms_max_overlap = 1

model_filename = 'model_data/mars-small128.pb' #pretrained cnn for tracking pedestrians
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('/content/Single-Multiple-Custom-Object-Detection-and-Tracking/data/video/cars.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID') #that's guarantee using AVI file format
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi',codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen = 30) for _ in range(1000)]

counter = []

#capture all the frames from the video
while True:
	_, img = vid.read()
	if img is None:
		print('Completed')
		break

	#Transform the image obtained to yolo prediction
	img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_in = tf.expand_dims(img_in, 0) #expands because the original image 3D arrays shapes (haight, width n' channel) so we need add a one more dimension that's the batch one mean the batch size
	img_in = transform_images(img_in, 416)
	#Now, we're ready for the yolo predictions
	t1 = time.time() #start the timer

	boxes, scores, classes, nums = yolo.predict(img_in) #pass images and predict
	# The yolo predictions return numpy empty arrays, includes the boxes, scores, classes and nums, the boxes per images are limited

	# boxes, 3D shape (1, 100, 4)
	# scores, 2D shape (1, 100)
	# classes, 2D shape (1, 100)
	# nums, 1D shape (1,)

	classes = classes[0]

	names = []
	for i in range(len(classes)):
		names.append(class_names[int(classes[i])])
	names = np.array(names)
	converted_boxes = convert_boxes(img, boxes[0])
	features = encoder(img, converted_boxes) #generate the features vector
	
	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

	cmap =  plt.get_cmap('tab20b')
	colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

 # run non-maxima suppresion
	boxs = np.array([d.tlwh for d in detections])
	scores = np.array([d.confidence for d in detections])
	classes = np.array([d.class_name for d in detections])
	indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap,scores) #witch boxes thas be gotten
	detections = [detections[i] for i in indices] # ready for deep_sort
	
	tracker.predict()
	tracker.update(detections)

	for track in tracker.tracks:
		#print(track.is_confirmed())
		if not track.is_confirmed() or track.time_since_update > 1:
			continue # if khalman filter couldn't assign a track or not updates in the track
		bbox = track.to_tlbr() # min_x, max_x, min_y, max_y
		class_name = track.get_class()
		color = colors[int(track.track_id) % len(colors)]
		color = [i * 255 for i in color]
		cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
		cv2.rectangle(img, (int(bbox[0]),int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17,int(bbox[1])), color, -1)
		cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]),int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)
		
		#Center points of objects
		center = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)) #Center point of the object
		pts[track.track_id].append(center) #Store the center points

		for j in range(1, len(pts[track.track_id])):
			if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
				continue
			thickness = int(np.sqrt(64/float(j+1))*2)
			cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

		#Gate counter
		height, width, _ = img.shape
		cv2.line(img, (0, int(3*height/6)), (width, int(3*height/6)), (0, 255, 0), thickness=2) #line in the middle of the screen

		center_y = int((bbox[1] + bbox[3])/2) #Just track the center_y

		if center_y <= int(3 * height/6 + height/30) and center_y >= int(3 * height/6 - height/30):
			if class_name == 'car' or class_name == 'truck':
				counter.append(int(track.track_id))
	total_count = len(set(counter))
	#							     top left,font face, fscale, color, thickness
	cv2.putText(img, "Contador de vehiculos: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

	fps = 1./(time.time()-t1)
	cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
	result = np.asarray(img)
	#result = cv2.cvtColor(img,)
	#cv2.resizeWindow('output', 1024, 768)
	#cv2.imshow('output',img)
	out.write(result)

	if cv2.waitKey(1)==ord('q'):
		break
#vid.release()
#out.release()
cv2.destroyAllWindows()
