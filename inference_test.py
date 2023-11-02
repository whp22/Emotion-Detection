from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import timeit
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from preprocessing import preprocessing_factory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####################################### Processing Flags ######################################

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('preprocessing_name', 'mobilenet_v1', 'The name of the preprocessing to use.')

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1', 'The name of the architecture to use')

tf.app.flags.DEFINE_string('image_path', 'lady.jpg', 'The path of the images to be inferenced')

tf.app.flags.DEFINE_string('face_classifier',
                           'haarcascade_frontalface_default.xml',
                           'The path where the face detection classifier located')

tf.app.flags.DEFINE_integer('image_size', 48, 'Image size of preprocessing')

###############################################################################################

# Emotion labels list:
#list = ['Angry','Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# Read from Pictures:
#frame = cv2.imread(FLAGS.image_path, cv2.IMREAD_COLOR)

# Read from Camera:
cap = cv2.VideoCapture(0)

# Define the face detector:
face = cv2.CascadeClassifier(FLAGS.face_classifier)

#####################################
# Select the preprocessing function #
#####################################

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

image_size = FLAGS.image_size

######################################
# Define the face detection function #
######################################

def face_detection(frame):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_start = timeit.default_timer()
    faces = face.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detect_time = timeit.default_timer() - detect_start
    print('detection_time: {0}'.format(detect_time))

    face_length = len(faces)
    print('Found {0} faces!'.format(face_length))

    face_data = []

    start = timeit.default_timer()
    for (x, y, w, h) in faces:
        crop_img = frame
        crop_img = crop_img[y:y + h, x:x + w]

        #crop_img = frame[y:y+h, x:x+w]
        #color_image = np.stack((crop_img,)*3, -1)

        image = tf.image.convert_image_dtype(crop_img, dtype=tf.float32)

        m = image_preprocessing_fn(image, image_size, image_size)

        face_data.append(m)

    time = timeit.default_timer() - start
    print('preprocessing time: {0}'.format(time))

    return faces, face_data, face_length

###################################
# Loading and importing the graph #
###################################
graph_def = None
graph = None

print('Loading graph definition ...')
try:
    with tf.gfile.GFile('../output_frozen_graph/6large/rate0.94_decay15_lr0.045_step8000_dropout0.5_momentum0.9.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
except BaseException as e:
    print (e.message)
    print('Error loading the graph definition...')
    #parser.exit(2, 'Error loading the graph definition: {}'.format(str(e)))

print('Importing graph ...')
try:
    assert graph_def is not None
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
except BaseException as e:
    print (e.message)
    print('Error importing the graph...')
    #parser.exit(2, 'Error importing the graph: {}'.format(str(e)))

assert graph is not None


#####################
# Configure the GPU #
#####################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config = tf.ConfigProto(device_count={'GPU': 0})


def main(_):

    start = timeit.default_timer()
    # Read from images:
    '''
    with tf.Session(graph=graph, config=config) as sess:
        # Get the input and output of the computing graph:
        input = sess.graph.get_tensor_by_name('input:0')
        output = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')

        # Get the faces in a frame:
        face_start = timeit.default_timer()
        faces, face_data, face_length = face_detection(frame)
        face_time = timeit.default_timer() - face_start
        print('face_time: {0}'.format(face_time))
        face_data = sess.run(face_data)

        # Do the inference:

        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()

        #result = sess.run(output, feed_dict={input: face_data},options=options, run_metadata=run_metadata)

        inf_start = timeit.default_timer()

        result = sess.run(output, feed_dict={input: face_data})
        print('results: ', result)

        inf_end = timeit.default_timer()
        inf_time = inf_end - inf_start
        print('inf_time: {0}'.format(inf_time))

        processing_time = inf_end - start
        print('processing_time: {0}'.format(processing_time))

        # print('results: ', result)
        # Create the Timeline object, and write it to a json file:
        #fretched_timeline = timeline.Timeline(run_metadata.step_stats)
        #chrome_trace = fretched_timeline.generate_chrome_trace_format()

        #with open('timeline_inference_abba.json', 'w') as f:
        #    f.write(chrome_trace)


        for i, (x, y, w, h) in enumerate(faces):
            print('face_%d' % i)
            ind = np.where(result[i, :] == np.max(result[i, :]))
            index = int(ind[0][0])
            print (list[index])
            print ('\n')

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, list[index], (x + 3, y + 10), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
    
        
            

    '''
    # Read from camera:
    with tf.Session(graph=graph, config=config) as sess:

        while (True):
            ret, frame = cap.read()
            #frame = cv2.equalizeHist(frame)
            faces, face_data, face_length = face_detection(frame)

            face_data = sess.run(face_data)

            input = sess.graph.get_tensor_by_name('input:0')
            output = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')

            inf_start = timeit.default_timer()

            result = sess.run(output, feed_dict={input: face_data})

            inf_end = timeit.default_timer()
            inf_time = inf_end - inf_start
            print('inf_time: {0}'.format(inf_time))

            for i, (x, y, w, h) in enumerate(faces):
                print('face_%d' % i)
                print(result[i, :])
                probability = np.max(result[i, :])
                print('probability: ', probability)
                ind = np.where(result[i, :] == np.max(result[i, :]))
                index = int(ind[0][0])
                emotion_label = list[index]
                print('emotion: ', emotion_label)
                print('\n')

                if emotion_label == 'Angry':
                    color = probability * np.asarray((255, 0, 0))
                elif emotion_label == 'Sad':
                    color = probability * np.asarray((0, 0, 255))
                elif emotion_label == 'Happy':
                    color = probability * np.asarray((255, 255, 0))
                elif emotion_label == 'Surprise':
                    color = probability * np.asarray((0, 255, 255))
                else:
                    color = probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()


                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, emotion_label, (x + 3, y + 10), font, 3, color, 2, cv2.LINE_AA)


            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
