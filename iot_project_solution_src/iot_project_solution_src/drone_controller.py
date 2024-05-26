import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle

from geometry_msgs.msg import Point, Vector3, Twist
from nav_msgs.msg import Odometry
from iot_project_solution_interfaces.action import PatrollingAction

from iot_project_solution_src.math_utils import *

import numpy as np

# This variable is used for the drone to stay away from the ground
# Now that our movement also makes the drone fly up if necessary, the
# fly_to_altitude function should only be used to compensate if the drone is
# too close to the ground.
# This movement is necessary because rotors behave differently when close
# to the ground, so it is wise to always lift the drone up a little bit
# before doing any further movement.
DRONE_MIN_ALTITUDE_TO_PERFORM_MOVEMENT = 1


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")

        self.position = Point(x=0.0, y=0.0, z=0.0)
        self.yaw = 0

        self.cmd_vel_topic = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        self.odometry_topic = self.create_subscription(
            Odometry,
            'odometry',
            self.store_position_callback,
            10
        )

        self.patrol_action = ActionServer(
            self,
            PatrollingAction, 
            'patrol_targets',
            self.patrol_action_callback
        )

        self.wind = Vector3()

        self.wind_topic_sub = self.create_subscription(
            Vector3,
            '/set_wind',
            self.set_wind_callback,
            10
        )


    def store_position_callback(self, msg : Odometry):
        self.position = msg.pose.pose.position
        self.yaw = get_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )


    def patrol_action_callback(self, msg : ServerGoalHandle):

        command_goal : PatrollingAction.Goal = msg.request
        targets = command_goal.targets

        # move to altitude
        self.fly_to_altitude()
        
        count = 0
        for target in targets:

            count += 1

            # rotate to target
            self.rotate_to_target(target)
            # move to target
            self.move_to_target(target)
            # send feedback for the target reached
            self.report_target_reached(msg, count)


        msg.succeed()

        result = PatrollingAction.Result()
        result.success = "Patrolling completed!"

        #self.get_logger().info("Patrol task completed! Sending final result...")

        return result


    def fly_to_altitude(self, altitude = DRONE_MIN_ALTITUDE_TO_PERFORM_MOVEMENT):

        # Skip movement if desiderd altitude is already reached
        if (self.position.z >= altitude):
            return

        # Instantiate the move_up message
        move_up = Twist()
        move_up.linear = Vector3(x=0.0, y=0.0, z=1.0)  
        move_up.angular = Vector3(x=0.0, y=0.0, z=0.0)  

        self.cmd_vel_topic.publish(move_up)

        # Loop until for the drone reaches the desired altitude
        # Note that in order for the drone to be perfectly aligned with the
        # requested height (not required for the exercise), you should keep on
        # listening to the current position and reduce the linear speed when 
        # you get close to the desired altitude
        while(self.position.z < altitude):
            self.cmd_vel_topic.publish(move_up)
            time.sleep(0.1)

        # Stop movement after the altitue has been reached
        stop_mov = Twist()
        stop_mov.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_mov.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_mov)


    def rotate_to_target(self, target : Point, eps = 0.1):

        target = (target.x, target.y, target.z)

        # We compute the angle between the current target position and the target
        # position here

        start_position = (self.position.x, self.position.y)
        target_angle = angle_between_points(start_position, target)
        angle_to_rotate = target_angle - self.yaw

        # We verify the optimal direction of the rotation here
        rotation_dir = -1
        if angle_to_rotate < 0 or angle_to_rotate > math.pi:
            rotation_dir = 1
        
        # Prepare the cmd_vel message
        move_msg = Twist()
        move_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        move_msg.angular = Vector3(x=0.0, y=0.0, z = 0.5 * rotation_dir)


        # Publish the message until the correct rotation is reached (accounting for some eps error)
        # Note that here the eps also helps us stop the drone and not overshoot the target, as
        # the drone will keep moving for a while after it receives a stop message
        # Also note that rotating the drone too fast will make it loose altitude.
        # You can account for that by also giving some z linear speed to the rotation movement.
        while abs(angle_to_rotate) > eps:
            angle_to_rotate = target_angle - self.yaw

            #adjuste the angle to rotate if there is wind
            #move_msg.angular = self.adjust_for_wind(move_msg.angular)

            self.cmd_vel_topic.publish(move_msg)
            # No sleep here. We don't want to miss the angle by sleeping too much. Even 0.1 seconds
            # could make us miss the given epsilon interval

        # When done, send a stop message to be sure that the drone doesn't
        # overshoot its target
        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_msg)


    def move_to_target(self, target : Point, eps = 0.5, angle_eps = 0.05):

        current_position = (self.position.x, self.position.y, self.position.z)
        objective_point = (target.x, target.y, target.z)

        while point_distance(current_position, objective_point) > eps:

            current_position = (self.position.x, self.position.y, self.position.z)
            direction_vector = move_vector(current_position, objective_point)

                
            if (self.wind[0] != 0 or self.wind[1] != 0 or self.wind[2] != 0):

                drag_coefficient = 0.1  # Coefficiente di resistenza

           
                a_wind_x = (drag_coefficient * (self.wind[0]))
                a_wind_y = (drag_coefficient * (self.wind[1]))
                a_wind_z = (drag_coefficient * (self.wind[2]))

                
                
                new_position_x = current_position[0] + a_wind_x
                new_position_y = current_position[1] + a_wind_y
                new_position_z = current_position[2] + a_wind_z

                new_current_position = (new_position_x,new_position_y,new_position_z)

                new_direction_vector = move_vector(new_current_position, objective_point)

                mov = Twist()
                mov.linear = Vector3(x=new_direction_vector[0], y=0.0, z=new_direction_vector[1])
                mov.angular = Vector3(x=0.0, y=0.0, z=0.0)

            else:

                mov = Twist()
                mov.linear = Vector3(x=direction_vector[0], y=0.0, z=direction_vector[1])
                mov.angular = Vector3(x=0.0, y=0.0, z=0.0)

            angle = angle_between_points(current_position, objective_point)
            current_angle = self.yaw

            if not (angle-angle_eps < current_angle < angle+angle_eps):
                angle_diff = (current_angle-angle)
                mov.angular = Vector3(x=0.0, y=0.0, z=math.sin(angle_diff))

            

            self.cmd_vel_topic.publish(mov)

        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_msg)
   
    
    def set_wind_callback(self, msg):
        self.wind = (msg.x, msg.y, msg.z)



     
    def report_target_reached(self, goal_handle, target_count):

        feedback = PatrollingAction.Feedback()
        #self.get_logger().info("Target %d reached. Sending feedback." % target_count)
        feedback.progress = "Target %d reached" % target_count
        goal_handle.publish_feedback(feedback)


def main():
    rclpy.init()

    executor = MultiThreadedExecutor()
    drone_controller = DroneController()

    executor.add_node(drone_controller)
    executor.spin()

    executor.shutdown()
    drone_controller.destroy_node()

    rclpy.shutdown()
