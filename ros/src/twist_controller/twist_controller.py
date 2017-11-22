import math, time
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704

WEIGHT_PERSON = 75
MIN_SPEED = 1.0 * ONE_MPH


class Controller(object):
    def __init__(self, vehicle_mass,fuel_capacity,brake_deadband,accel_deadband,decel_limit,accel_limit,wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.accel_deadband = accel_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.last_time = None
        self.steer_lowpass = LowPassFilter(0.96, 1.) # from https://discussions.udacity.com/t/solved-compute-cte-without-car-position/364383/8
        self.throttle_lowpass = LowPassFilter(1., 3.) # acceleration should do something intelligent with max accle&dec mass etc
        self.brake_lowpass = LowPassFilter(1., 3.) # deccelaration
        self.yawController = YawController(wheel_base,steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)



        # TODO: how do I implement twiddle_
        kp = 1.5 # 1.5 1.0
        ki = 0.01 # 0.001
        kd = 0.01 # 0. 0.0.001
        self.velocity_pid = PID(kp, ki, kd,decel_limit,accel_limit)
        

    def control(self, proposed_velocity, proposed_angular,current_velocity,dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        msg = 'controller called with proposed_velocity={}, proposed_angular={}, current_velocity={}, dt={}'.format(proposed_velocity,proposed_angular,current_velocity,dt)        
        print(msg)
        # velocity
        velocity_change = self.velocity_pid.step(proposed_velocity-current_velocity,dt)
        
        throttle = 0
        brake = 0
        # acceleration
        if velocity_change > 0:
            throttle = self.throttle_lowpass.filt(velocity_change)
            throttle = throttle if throttle > self.accel_deadband else 0.0

        # brake
        else:
            brake = self.brake_lowpass.filt(-velocity_change)
            brake = brake * (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius
            brake = brake if brake > self.brake_deadband else 0.0
            

        # steering
        steer = self.yawController.get_steering(proposed_velocity, proposed_angular, current_velocity)
        steer = self.steer_lowpass.filt(steer)

        # Add case for stopping at red light
        if abs(proposed_velocity) < 0.001:
            throttle = 0.
            brake = 400.0 #.1
        #    steer = 0.

        self.last_time = time.time()

        msg = 'controller calclated throttle={}, brake={}, steer={}'.format(throttle,brake,steer)        
        print(msg)
        return throttle, brake, steer


    def reset(self):
        self.velocity_pid.reset()
