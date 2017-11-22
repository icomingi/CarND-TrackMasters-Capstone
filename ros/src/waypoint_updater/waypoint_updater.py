#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

from std_msgs.msg import Int32

import numpy as np

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    
    def __init__(self):
        
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # Subscriber for /traffic_waypoint
        # assuming msg type is Int64
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.base_waypoints = None
        self.waypoints_to_pub = None # store the index of waypoints to publish
        self.next_wp_index = 0 # index of the closest waypoint ahead of the car

        # Get maximum speed in m/s
        self.max_velocity = rospy.get_param('/waypoint_loader/velocity')*1000./3600.

        # Add a member variable to store index of next waypoints
        #self.next_wp_index = None

        self.last_stop_id = -1
        self.deceleration_set = False
        self.acceleration_set = False

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        
        # Set current pose
        self.current_pose = msg.pose

        # Get next closest waypoint that is ahead of the car
        self.next_wp_index = self.get_next_waypoint(self.current_pose)
        #rospy.loginfo('car x=%s, y=%s, next wp_x=%s, y=%s',
        #    self.current_pose.position.x, self.current_pose.position.y,
        #    self.base_waypoints[self.next_wp_index].pose.pose.position.x,
        #    self.base_waypoints[self.next_wp_index].pose.pose.position.y)
                    
        # Get indexes of waypoints that are ahead of car position to waypoints_to_pub list
        self.waypoints_to_pub = self.base_waypoints[self.next_wp_index:self.next_wp_index+LOOKAHEAD_WPS]
        #for i in range(0, LOOKAHEAD_WPS):
        #    self.waypoints_to_pub.append(self.next_wp_index + i)
        
        # Publish waypoints to final_waypoints topic
        msg = Lane()
        msg.waypoints = self.waypoints_to_pub
        self.final_waypoints_pub.publish(msg)

    def get_next_waypoint(self, pose):
        closest_wp_index = 0
        closest_wp_dist = 100000
        
        # car's position
        x_car = pose.position.x
        y_car = pose.position.y

        if self.base_waypoints != None:
            for i in range(len(self.base_waypoints)):
                # current waypoint's position
                x_wp = self.base_waypoints[i].pose.pose.position.x
                y_wp = self.base_waypoints[i].pose.pose.position.y
                # compute distance
                dist = math.hypot(x_car - x_wp, y_car - y_wp)
                # compare and set
                if dist < closest_wp_dist:
                    closest_wp_dist = dist
                    closest_wp_index = i

            # get car's current orientation
            yaw = self.yaw_from_quaternion(self.current_pose)
            # get car's heading if going to the closest waypoint
            closest_wp = self.base_waypoints[closest_wp_index]
            heading = math.atan2(closest_wp.pose.pose.position.y - y_car, 
                closest_wp.pose.pose.position.x - x_car) # btw (-pi, pi)
            if heading < 0:
                heading = heading + 2*math.pi; # set to btw (0, 2pi) as yaw is btw (0, 2pi)
            # get difference on heading, if diff is bigger than pi/4, take the next waypoint
            diff_heading = abs(yaw - heading)
            if diff_heading > math.pi/4:
                closest_wp_index = closest_wp_index + 1
        # Fix bug closest_wp_index out of index
        return closest_wp_index % len(self.base_waypoints)

    def yaw_from_quaternion(self, pose):
        return tf.transformations.euler_from_quaternion((pose.orientation.x, 
            pose.orientation.y, pose.orientation.z, pose.orientation.w))[2]

    def waypoints_cb(self, msg):
        # TODO: Implement
        # Store base waypoints to a list as it's only published once
        self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        stop_id = msg.data
        # AK rospy.loginfo('stop_id: {} - last_stop_id: {}'.format(stop_id, self.last_stop_id))
        if (stop_id > self.last_stop_id):
            # Change to red light
            self.last_stop_id = stop_id
            self.deceleration_set = False
            rospy.loginfo("Red light detected, stop line at {}".format(stop_id))
        elif (stop_id < self.last_stop_id):
            # Change to no red light
            self.last_stop_id = stop_id
            self.acceleration_set = False
            rospy.loginfo("No red light ahead, ready to accelerate")
        if stop_id < 0:
            # Get no stop_id
            starting_v = self.get_waypoint_velocity(self.base_waypoints[self.next_wp_index])
            # Set acceleration trajectory when if there's no red light ahead
            if self.max_velocity - starting_v > 0.1 and not self.acceleration_set:
                avg_accel = 2.
                fn_s, fn_v, T = self.generate_acceleration_trajectory(0., self.max_velocity, avg_accel)
                wps_id = self.next_wp_index
                d = 0.
                while wps_id <= self.base_waypoints[-1]:
                    d += self.distance(self.base_waypoints, wps_id, wps_id+1)
                    t = newton_solve(fn_s, fn_v, d, T)
                    target_v = max(0., min(fn_v(t), self.max_velocity))
                    if self.max_velocity - target_v > 0.1:
                        self.set_waypoint_velocity(self.base_waypoints, wps_id, target_v)
                        wps_id += 1
                    else:
                        break
                self.acceleration_set = True
                rospy.loginfo("Set accleration trajectory success")
        else:
            # Get stop_id
            id_diff = stop_id - self.next_wp_index
            if id_diff >= 0 and id_diff < LOOKAHEAD_WPS and not self.deceleration_set:
                # Calculate distance till next red light
                total_dist = self.distance(self.base_waypoints, self.next_wp_index, stop_id)
                rospy.loginfo("Distance till red light: %f", total_dist)
                starting_v = self.get_waypoint_velocity(self.base_waypoints[self.next_wp_index])
                rospy.loginfo("Starting velocity: %f", starting_v)

                fn_s, fn_v, T = self.generate_fullstop_trajectory(starting_v, total_dist)

                # Set target velocity for each waypoint till stop line
                d = 0.
                for wps_id in range(self.next_wp_index+1, stop_id): 
                    d += self.distance(self.base_waypoints, wps_id-1, wps_id)
                    t = newton_solve(fn_s, fn_v, d, T)
                    # Make sure target velocity is between 0 and max_velocity
                    red_factor = ((stop_id*1.0-wps_id)/(stop_id-self.next_wp_index-1))**2.5
                    target_v = max(0., min(fn_v(t), self.max_velocity)) * red_factor
                    if target_v < 1.5:
                        target_v = 0
                    self.set_waypoint_velocity(self.base_waypoints, wps_id, target_v)

                # Set velocity at the stop line to 0.
                self.set_waypoint_velocity(self.base_waypoints, stop_id, 0.) 

                # Set acceleration trajectory after stop line
                avg_accel = 2.
                fn_s, fn_v, T = self.generate_acceleration_trajectory(0., self.max_velocity, avg_accel)
                wps_id = stop_id + 1
                d = 0.
                while wps_id <= self.base_waypoints[-1]:
                    d += self.distance(self.base_waypoints, wps_id-1, wps_id)
                    t = newton_solve(fn_s, fn_v, d, T)
                    target_v = max(0., min(fn_v(t), self.max_velocity))
                    if self.max_velocity - target_v > 0.1:
                        self.set_waypoint_velocity(self.base_waypoints, wps_id, target_v)
                        wps_id += 1
                    else:
                        break
                self.deceleration_set = True
                rospy.loginfo("Set full stop till red light trajectory success")
        # Publish final waypoints
        self.waypoints_to_pub = self.base_waypoints[self.next_wp_index:self.next_wp_index+LOOKAHEAD_WPS]
        msg = Lane()
        msg.waypoints = self.waypoints_to_pub
        self.final_waypoints_pub.publish(msg)

    def generate_acceleration_trajectory(self, start_v, end_v, avg_accel):
        dist_est = abs((end_v**2-start_v**2)/2/avg_accel)
        T_est = abs((start_v-end_v)/avg_accel)
        d = 1 * dist_est
        T = 1 * T_est
        start = [0., start_v, 0.]
        end = [d, end_v, 0.]
        alphas = JMT(start, end, T)
        fn_s = get_fn_s(alphas)
        fn_v = get_fn_v(alphas)
        return fn_s, fn_v, T

    def generate_fullstop_trajectory(self, start_v, d):
        decel_est = start_v**2 / (2 * d)
        rospy.loginfo("Estimated average deceleration: {}".format(decel_est))
        if (abs(start_v) < 0.00001):
            T_est = 2 * d / 0.00001
        else:
            T_est = 2 * d / start_v
        # Force the car to stop faster
        T = T_est * 1.
        decel_forced = start_v / T
        rospy.loginfo("Forced average deceleration: {}".format(decel_forced))
        start = [0., start_v, 0.]
        end = [d, 0., 0.]
        alphas = JMT(start, end, T)
        fn_s = get_fn_s(alphas)
        fn_v = get_fn_v(alphas)
        return fn_s, fn_v, T

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

def JMT(start, end, T):
    """
    Calculate jerk minimizing trajectory for start, end, and T
    Return coefficients
    """
    a0, a1, a2 = start[0], start[1], start[2]/2.0
    c0 = a0 + a1*T + a2*T**2
    c1 = a1 + 2*a2*T
    c2 = 2*a2

    A = np.array([
            [T**3,   T**4,    T**5],
            [3*T**2, 4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3],
        ])
    B = np.array([
            end[0] - c0,
            end[1] - c1,
            end[2] - c2
        ])

    a_3_4_5 = np.linalg.solve(A, B)
    alphas = np.concatenate([np.array([a0, a1, a2]), a_3_4_5])
    return alphas

def get_fn_s(alphas):
    """
    Return polynomials of distance travelled
    """
    def fn_s(t):
        #a0, a1, a2, a3, a4, a5 = alphas
        ts = np.array([1, t, t**2, t**3, t**4, t**5])
        return np.inner(alphas, ts)
    return fn_s

def get_fn_v(alphas):
    """
    Return polynomials of velocity
    """
    def fn_v(t):
        a0, a1, a2, a3, a4, a5 = alphas
        return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    return fn_v

def newton_solve(f, df, s, bound, tolerance=0.0001):
    """
    Using Newton method to solve x for f(x) = s
    """
    x0 = bound/2.0
    dx = x0
    while abs(dx) > tolerance:
        x1 = x0 - (f(x0)-s)/df(x0)
        dx = x0 - x1
        x0 = x1
    return x0

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
