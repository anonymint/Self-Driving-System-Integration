from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_site = False):
        #TODO load classifier
        self.is_site = is_site
        if self.is_site :
            model_path = "light_classification/model/frozen_inference_graph_real.pb"
        else:
            model_path = "light_classification/model/frozen_inference_graph_sim.pb"

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

             # TF Graph is ready
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        # pass graph session to rest of program
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # # FOR TESTING STATIC VALUE USE THIS BLOCK - START
        # return TrafficLight.UNKNOWN
        # # FOR TESTING STATIC VALUE USE THIS BLOCK - END

        # # FOR TF CLASSIFIED USE THIS BLOCK - START
        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            start_clock = datetime.datetime.now()
            (boxes, scores, classes, num) = self.sess.run(
                [self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
            stop_clock = datetime.datetime.now()
            print("Classified in", (stop_clock-start_clock).total_seconds())

        # squeeze here from shape (1,10) to (,10)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        print('SCORES', scores[0], 'CLASSES', classes[0])

        # color code
        # {
        #  1: {'id': 1, 'name': 'Green'},
        #  2: {'id': 2, 'name': 'Red'},
        #  3: {'id': 3, 'name': 'Yellow'},
        #  4: {'id': 4, 'name': 'off'}
        # }
        if scores[0] >= self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[1] == 3:
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
        # # FOR TF CLASSIFIED USE THIS BLOCK - END