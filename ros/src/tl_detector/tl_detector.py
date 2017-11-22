#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import scipy
import time

import numpy as np
from scipy.misc import imresize

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # AK debug:
        self.save_images_for_debugging = False
        self.counter = 0
        self.test_img_dir = './test_imgs/'#'/home/student/Desktop/shared/'
        #

        self.pose = None
        self.waypoints = []
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Add coefficients from regression
        self.y_estimator = np.poly1d(np.array([-8603.39023981, 572.75655132]))
        self.height_estimator = np.poly1d(np.array([2978.60361045, 24.61502163]))

        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints


    def traffic_cb(self, msg):
        self.lights = msg.lights
        #print('here in traffic_cb')
        #print(msg)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # AK rospy.loginfo('light-waypoint: {}, state: {}'.format(light_wp, state))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            # state change, but we still need to publish the last wp
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_wp_index = 0
        closest_wp_dist = 100000.0
        
        for i in range(len(self.waypoints)):
            x1 = pose.position.x
            y1 = pose.position.y
            x2 = self.waypoints[i].pose.pose.position.x
            y2 = self.waypoints[i].pose.pose.position.y
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < closest_wp_dist:
                closest_wp_dist = dist
                closest_wp_index = i

        return closest_wp_index


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        
        #Get classification
        distance = self.distance_to_light(light)
        if distance < 60:
            # light could be visible in the camera image
            cropped_img = self.crop_window(distance, cv_image)
            #cropped_img = cv_image[cc[0][0]:cc[1][0],cc[0][1]:cc[1][1],:]
            
            # TODO: 
            # once crop_window is implemented properly, pass the properly cropped and resized image to the classifier
            actual_pred = light.state
            pred =  self.light_classifier.get_classification(cropped_img)
            
            # AK debug
            if self.save_images_for_debugging:
                scipy.misc.imsave(self.test_img_dir+'{:04d}-{}-{:.0f}'.format(self.counter,pred,distance)+'.png', cropped_img)
                self.counter += 1



            return pred
        else:
            # light is not visible in the camera image
            pred = TrafficLight.UNKNOWN



    def print_all_lights(self, light):
    	'''
    	AK: for debugging purposes. 
    	write the position of the car and all traffic lights to a text file
    	'''
    	f = open(self.test_img_dir+'{:04d}'.format(self.counter)+'.txt','w')

    	if (self.pose):
    		car_wp_index = self.get_closest_waypoint(self.pose.pose)
    		x = self.pose.pose.position.x
    		y = self.pose.pose.position.y
    		z = self.pose.pose.position.z
    		other_z = self.pose.pose.orientation.z
    		w = self.pose.pose.orientation.w
    		f.write('car_waypoint: {}, x: {}, y: {}, z: {}, other_z: {}, w: {}, '.format(car_wp_index, x, y, z, other_z, w))
    	for l in self.lights:
    		light_wp_index = self.get_closest_waypoint(l.pose.pose)
    		x = l.pose.pose.position.x
    		y = l.pose.pose.position.y
    		z = l.pose.pose.position.z
    		state = l.state
    		f.write('light_waypoint: {}, state: {}, x: {}, y: {}, z:{}, '.format(light_wp_index, state, x, y, z))
    	l_wp = self.get_closest_waypoint(light.pose.pose)
    	l_x = light.pose.pose.position.x
    	l_y = light.pose.pose.position.y
    	l_z = light.pose.pose.position.z
    	l_state = light.state
    	f.write('l_wp: {}, l_x: {}, l_y: {}, l_z: {}, state: {}'.format(l_wp, l_x, l_y, l_z, l_state))

    	f.close()

    
    def distance_to_light(self, light):
        if (self.pose):
            l_x = light.pose.pose.position.x
            l_y = light.pose.pose.position.y
            car_x = self.pose.pose.position.x
            car_y = self.pose.pose.position.y
            distance = ((l_x-car_x)**2+(l_y-car_y)**2)**0.5
            return distance
        else:
            return 1000


    def crop_window(self, distance, cv_image):
        '''
        TODO: This is a placeholder only.
        properly identify the crop window around the next traffic lisght 
        '''
        height, width = cv_image.shape[:2]
        if distance < 15.:
            return cv_image[:height//2, :, :]
        else:
            y_est, h_est = int(self.y_estimator(1./distance)), int(self.height_estimator(1./distance))
            y_min = int(min(height, max(0, y_est - int(h_est*.8))*0.9))
            y_max = int(min(height, max(0, y_est + int(h_est*1.3))*1.1))
            sliced_img = cv_image[y_min:y_max, :, :]
            delta_y = y_max - y_min
            # scale = 2 if delta_y < 128 else 1
            scale =1 
            ratio = 1.*width/delta_y
            resized_img = imresize(sliced_img, (64*scale, int(64*ratio*scale)), "nearest")
            return resized_img

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        car_wp_index = 0
        if(self.pose):
            car_wp_index = self.get_closest_waypoint(self.pose.pose)

        light = None
        stop_line_wp_index = 0
        for i in range(len(stop_line_positions)):
            pose = Pose()
            pose.position.x = stop_line_positions[i][0]
            pose.position.y = stop_line_positions[i][1]
            stop_line_wp_index = self.get_closest_waypoint(pose)
            index_diff = stop_line_wp_index - car_wp_index
            #rospy.loginfo("LightID: %d = StopLine WP: %d, Car WP: %d", i, stop_line_wp_index, car_wp_index)
            # light is ahead of car
            if index_diff > 0:# and index_diff < 200:
                light = self.lights[i]
                #rospy.loginfo("Breaking at LightID: %d, WP: %d", i, self.get_closest_waypoint(light.pose.pose))
                break

        if light:
            #Debug, to see how much time the classifie took to process one image
            #t0 = time.time()
            state = self.get_light_state(light)
            #rospy.loginfo("Light detection took {:.2f}s".format(time.time()-t0))
            return stop_line_wp_index, state
        #self.waypoints = []
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
