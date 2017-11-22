from styx_msgs.msg import TrafficLight
from keras.models import model_from_json
import json
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import tensorflow as tf
import time


class TLClassifier(object):
    def __init__(self):

        self._work_dir ='./light_classification/'
        self._model_name = 'model_light'
        
        # load the model
        with open(self._work_dir+self._model_name+'.json', 'r') as jfile:
          json_str = json.loads(jfile.read())
          self._model = model_from_json(json_str)
    
        #use default hyper parameters when creating new network
        Adam_Optimizer = Adam(lr=0.00001)
        self.model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        # load weights
        self.model.load_weights(self._work_dir+self._model_name+'.h5')

        # freeze layers to avoid being trainable
        for l in self.model.layers:
        	l.trainable = False

        # compile forward pass
        self.fp = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        
        pass
    
    
    @property
    def model(self):
      return self._model
    
    
    def slide_window(self, img_shape, xy_overlap=(0.5, 0.5)):
        '''
        Takes an image, start and stop positions in both x and y, window size in x and y directions
        and overlap fraction in both x and y directions and returns list of window coordinates.
        img_shape: subject image shape
        x_start_stop: start and end position in x direction (array of size 2)
        y_start_stop: start and end position in y direction (array of size 2)
        xy_overlap: amount of overlap between windows in both x and y directions (tuple)
        '''
        
        # size of window in x and y directions
        xy_window=(64, 64)
        
        # If x and/or y start/stop positions not defined, set to image size
        x_start_stop = (0, img_shape[1])
        y_start_stop = (0, img_shape[0])
        
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    
    def normalize(self, image, norm_factor=0.5):
        '''
        Normalize the piexel values of the images
        '''
        norm_img = np.copy(image).astype('float32')
        norm_img -= 127.5
        norm_img /= 127.5
        norm_img *= norm_factor
        
        return norm_img
    

    def slide_imgs(self, image):
    	windows = self.slide_window(image.shape)

    	img_windows = []
        for window in windows:
            x1 = window[0][0]
            y1 = window[0][1]
            x2 = window[1][0]
            y2 = window[1][1]
            img_box = image[y1:y2, x1:x2, :]
            img_windows.append(self.normalize(img_box))
        img_windows = np.asarray(img_windows)

        return img_windows


    def interpret_pred(self, predictions):
    	pred_indexes = [np.argmax(x) for x in predictions]
        unique, counts = np.unique(pred_indexes, return_counts=True)
        final_prediction = unique[np.argmax(counts[:3])]

        return final_prediction


    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
    	# obtain sliding images 64x64x3
      	img_windows = self.slide_imgs(image)
        
        # make predictions
        t0 = time.time()
    	predictions = self.fp([img_windows, 0])[0]

    	# interpret prediction results
        final_prediction = self.interpret_pred(predictions)
        
       
        if final_prediction == 0:
            return TrafficLight.GREEN
        elif final_prediction == 1:
            return TrafficLight.YELLOW
        elif final_prediction == 2:
            return TrafficLight.RED
        else: 
            return TrafficLight.UNKNOWN
        
