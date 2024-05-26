import time
import random

from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from geometry_msgs.msg import Point, Vector3
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

from iot_project_interfaces.srv import TaskAssignment
from iot_project_solution_interfaces.action import PatrollingAction

import iot_project_solution_src.kmeans as kmeans

import sklearn
import sklearn.cluster as cluster

from rclpy.task import Future


class TaskAssigner(Node):

    def __init__(self):

        super().__init__('task_assigner')
            
        self.task = None
        self.no_drones = 0
        self.targets = []
        self.thresholds = []

        self.action_servers = []
        self.current_tasks =  []
        self.idle = []

        self.simulation_name = ""
        self.simulation_time = 0
        self.last_visits = []
        self.clock = 0
        self.score = []
        self.distance_matrix = []
        self.prova = None

        self.computing_score = []

        self.sim_time = 0

        self.msg = Vector3()

        self.drone_positions = []

        self.task_announcer = self.create_client(
            TaskAssignment,
            '/task_assigner/get_task'
        )

        self.sim_time_topic = self.create_subscription(
            Clock,
            '/world/iot_project_world/clock',
            self.store_sim_time_callback,
            10
        )

        self.wind_topic = self.create_publisher(
            Vector3,
            '/set_wind',
            10
        )

        self.create_timer(0.05, self.update_times)


    def update_times(self):
        future = self.task_announcer.call_async(TaskAssignment.Request())
        future.add_done_callback(self.update_times_callback)
    
    def update_times_callback(self, res : Future):
        
        result : TaskAssignment.Response = res.result()
        self.simulation_name = result.simulation_name
        self.simulation_time = result.simulation_time
        self.last_visits = result.last_visits
        self.expiration_times = result.target_thresholds

        self.aoi_weight = result.aoi_weight
        self.violation_weight = result.violation_weight
        self.fairness_weight = result.fairness_weight


    # Function used to wait for the current task. After receiving the task, it submits
    # to all the drone topics
    def get_task_and_subscribe_to_drones(self):

        self.get_logger().info("Task assigner has started. Waiting for task info")

        while not self.task_announcer.wait_for_service(timeout_sec = 1.0):
            time.sleep(0.5)

        self.get_logger().info("Task assigner is online. Requesting patrolling task")

        assignment_future = self.task_announcer.call_async(TaskAssignment.Request())
        assignment_future.add_done_callback(self.first_assignment_callback)


    # Callback used for when the patrolling task has been assigned for the first time.
    # It configures the task_assigner by saving some useful values from the response
    # (more are available for you to read and configure your algorithm, just check
    # the TaskAssignment.srv interface).
    # The full response is saved in self.task, so you can always use that to check
    # values you may have missed. Or just save them here by editing this function.
    # Once that is done, it creates a client for the action servers of all the drones
    def first_assignment_callback(self, assignment_future):

        # task : TaskAssignment.Response = assignment_future.result()

        # self.task = task
        # self.no_drones = task.no_drones
        # self.targets = task.target_positions
        # self.thresholds = task.target_thresholds

        # self.current_tasks = [None]*self.no_drones
        # self.idle = [True] * self.no_drones

        # self.wind_vector = task.wind_vector

        
        # self.msg.x = task.wind_vector.x
        # self.msg.y = task.wind_vector.y
        # self.msg.z = task.wind_vector.z

        # self.wind_topic_pub.publish(self.msg)
       
        # # self.wind_topic_pub.publish(msg)
        # #self.get_logger().info(msg)

        # # Now create a client for the action server of each drone
        # for d in range(self.no_drones):
        #     self.action_servers.append(
        #         ActionClient(
        #             self,
        #             PatrollingAction,
        #             'X3_%d/patrol_targets' % d,
        #         )
        #     )

        kmeans.first_assignment_callback(self, assignment_future)



    def register_drone_position(self, msg : Odometry, drone : str):
        self.drone_positions[drone] = msg.pose.pose.position


    # This method starts on a separate thread an ever-going patrolling task, it does that
    # by checking the idle state value of every drone and submitting a new goal as soon as
    # that value goes back to True
    def keep_patrolling(self):

        def keep_patrolling_inner():
            while True:
                for d in range(self.no_drones):
                    if self.idle[d]:

                        Thread(target=self.submit_task, args=(d,)).start()

                time.sleep(0.1)

        Thread(target=keep_patrolling_inner).start()

    
    # Submits a patrol task to a single drone. Basic implementation just takes the array
    # of targets and shuffles it. Is up to you to improve this part and come up with your own
    # algorithm.
    # 
    # TIP: It is highly suggested to start working on a better scheduling of the targets from here.
    #      some drones may want to inspect only a portion of the nodes, other maybe more.
    #
    #      You may also implement a reactive solution which checks for the target violation
    #      continuously and schedules precise tasks at each step. For that, you can call again
    #      the task_announcer service to get an updated view of the targets' state; the last
    #      visit of each target can be read from the array last_visits in the service message.
    #      The simulation time is already stored in self.sim_time for you to use in case
    #      Times are all in nanoseconds.
    def submit_task(self, drone_id, targets_to_patrol = None):

        # self.get_logger().info("Submitting task for drone X3_%s" % drone_id)
    
        # while not self.action_servers[drone_id].wait_for_server(timeout_sec = 1.0):
        #     return

        # self.idle[drone_id] = False

        # if not targets_to_patrol:
        #     targets_to_patrol = self.targets.copy()
        #     random.shuffle(targets_to_patrol)

        # patrol_task =  PatrollingAction.Goal()
        # patrol_task.targets = targets_to_patrol

        # patrol_future = self.action_servers[drone_id].send_goal_async(patrol_task)

        # This is a new construct for you. Basically, callbacks have no way of receiving arguments except
        # for the future itself. We circumvent such problem by creating an inline lambda functon which stores
        # the additional arguments ad-hoc and then calls the actual callback function
        # patrol_future.add_done_callback(lambda future, d = drone_id : self.patrol_submitted_callback(future, d))

        kmeans.submit_task(self, drone_id, targets_to_patrol)


    # Callback used to verify if the action has been accepted.
    # If it did, prepares a callback for when the action gets completed
    def patrol_submitted_callback(self, future, drone_id):

        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info("Task has been refused by the action server")
            return
        
        result_future = goal_handle.get_result_async()

        # Lambda function as a callback, check the function before if you don't know what you are looking at
        result_future.add_done_callback(lambda future, d = drone_id : self.patrol_completed_callback(future, d))


    # Callback used to update the idle state of the drone when the action ends
    def patrol_completed_callback(self, future, drone_id):
        self.get_logger().info("Patrolling action for drone X3_%s has been completed. Drone is going idle" % drone_id)
        self.idle[drone_id] = True

        
        if (kmeans.first_scenario == False and self.computing_score[drone_id] == False):
            self.computing_score[drone_id] = True
            kmeans.compute_score(self, drone_id)


    # Callback used to store simulation time
    def store_sim_time_callback(self, msg):
        self.clock = msg.clock.sec * 10**9 + msg.clock.nanosec


        
def main():

    time.sleep(3.0)
    
    rclpy.init()

    task_assigner = TaskAssigner()
    executor = MultiThreadedExecutor()
    executor.add_node(task_assigner)

    task_assigner.get_task_and_subscribe_to_drones()
    task_assigner.keep_patrolling()

    executor.spin()

    executor.shutdown()
    task_assigner.destroy_node()

    rclpy.shutdown()

