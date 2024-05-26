import sklearn.cluster as cluster
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

import time
import random

import threading
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from rosgraph_msgs.msg import Clock

from iot_project_interfaces.srv import TaskAssignment
from iot_project_solution_interfaces.action import PatrollingAction

import iot_project_solution_src.math_utils as math_utils
import numpy as np

from rclpy.task import Future


import copy

first_scenario = False

targets_in_cluster = []

wind_vector = [] 



# Callback used for when the patrolling task has been assigned for the first time.
# It configures the task_assigner by saving some useful values from the response
# (more are available for you to read and configure your algorithm, just check
# the TaskAssignment.srv interface).
# The full response is saved in self.task, so you can always use that to check
# values you may have missed. Or just save them here by editing this function.
# Once that is done, it creates a client for the action servers of all the drones


def first_assignment_callback(task_assigner, assignment_future):

    task : TaskAssignment.Response = assignment_future.result()
    
    global wind_vector
    wind_vector = task.wind_vector
    
    # clustering
        
    positions = [[point.x, point.y, point.z] for point in task.target_positions]
    clusterer = cluster.KMeans(n_clusters=task.no_drones) #Create a KMeans instance
    clusterer.fit(positions) #Compute K-means clustering for the given dataset "positions"
    
    setattr(task_assigner, 'assignments', None) #it creates the attribute assignments 
    task_assigner.assignments = [[] for _ in range(task.no_drones)]
    task_assigner.get_logger().info("Task assigner created the clusters: " + str(clusterer.labels_))

    #clusterer.labels_ is a list that contains the index of the cluster in which each target falls in 
    
    for idx, label in enumerate(clusterer.labels_):
        task_assigner.assignments[label].append(task.target_positions[idx])
        # task_assigner.get_logger().info(str(task.target_positions[idx]) + " assigned to drone " + str(label))
    task_assigner.get_logger().info("Task assigner created the assignments: " + str(task_assigner.assignments))

    task_assigner.task = task
    task_assigner.no_drones = task.no_drones
    task_assigner.targets = task.target_positions

    task_assigner.current_tasks = [None]*task_assigner.no_drones
    task_assigner.idle = [True] * task_assigner.no_drones

    task_assigner.thresholds = task.target_thresholds

    # drone_positions = task.drone_positions
    


    #if we are in scenario1 we will perfom the tsp algorithm
    lst = task_assigner.thresholds
    global first_scenario
    if (task_assigner.fairness_weight > 0.5):
        first_scenario = True

      

    if (first_scenario == True):
        for d in range(task_assigner.no_drones):
            new_assignment = []
            coordinates = []
            ass = []
            for i in task_assigner.assignments[d]:
                tmp = None
                tmp = (i.x, i.y, i.z)
                ass.append(tmp)

            lun = len(ass)


            fitness_coords = mlrose.TravellingSales(coords = ass)
            problem_fit = mlrose.TSPOpt(length = lun, fitness_fn = fitness_coords, maximize=False)
            best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)


            for el in best_state:
                x = []
                x = task_assigner.assignments[d][el]
                new_assignment.append(x)

            if (len(new_assignment) == 1):
                for _ in range(5000):
                    new_assignment.append(x)
            
            task_assigner.assignments[d] = list(new_assignment)
        


    
    else:
        task_assigner.score = [0 for _ in range(len(task_assigner.targets))]
        task_assigner.computing_score = [False for _ in range(task_assigner.no_drones)]
        
        global targets_in_cluster
        targets_in_cluster = task_assigner.assignments.copy() 

        for d in range(task_assigner.no_drones):
            tmp = []
            tmp.append(targets_in_cluster[d][0])
            task_assigner.assignments[d] = tmp.copy()

        # calculate the distance of each target from the other ones.
        task_assigner.distance_matrix = [[] for _ in range(len(task_assigner.targets))]
        for i in range(len(task_assigner.distance_matrix)):
            for j in range(len(task_assigner.distance_matrix)):
                # task_assigner.get_logger().info("Task assigner i-j: " + str(i) + " " + str(j))
                if i == j:
                    task_assigner.distance_matrix[i].append(0)
                else:
                    # distance_matrix[i].appent(1)
                    p1 = task_assigner.targets[i]
                    p2 = task_assigner.targets[j]
                    distance = math_utils.point_distance((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))
                    task_assigner.distance_matrix[i].append(distance)

    
    # Now create a client for the action server of each drone
    for d in range(task_assigner.no_drones):
        task_assigner.action_servers.append(
            ActionClient(
                task_assigner,
                PatrollingAction,
                'X3_%d/patrol_targets' % d,
            )
        )

    task_assigner.wind_topic.publish(wind_vector)


def compute_score(task_assigner, drone_id):
    
    ind = task_assigner.assignments[drone_id][0]
    current_target = task_assigner.targets.index(ind)

    new_assignment = []
    scores = [0 for _ in range(len(targets_in_cluster[drone_id]))]
    


    sum_of_thresholds = 0
    for t in targets_in_cluster[drone_id]:
        index = task_assigner.targets.index(t)
        sum_of_thresholds += task_assigner.thresholds[index]*10**9


    future = task_assigner.task_announcer.call_async(TaskAssignment.Request())
    future.add_done_callback(task_assigner.update_times_callback)
    

    clock = task_assigner.clock
    

    #Computing the sum of distances of the current target from all the other target in the cluster
    sum_of_distances = 0
    for target2 in targets_in_cluster[drone_id]:
        target2_index = task_assigner.targets.index(target2)
        sum_of_distances += task_assigner.distance_matrix[current_target][target2_index]

    for k, target in enumerate(targets_in_cluster[drone_id]):
        score = 0
        # compute score
        idx = task_assigner.targets.index(target)
        last_visit_normalized = task_assigner.last_visits[idx]/clock
       


        
        current_threshold = max(0, task_assigner.expiration_times[idx]*10**9 - (clock - task_assigner.last_visits[idx]))
        
        current_threshold_normalized = current_threshold / sum_of_thresholds

        target_index = task_assigner.targets.index(target)
        distance = 0
        if sum_of_distances != 0:
            distance = task_assigner.distance_matrix[current_target][target_index] / sum_of_distances

        
        score = 1-(task_assigner.aoi_weight*last_visit_normalized + distance / 30 + task_assigner.violation_weight * current_threshold_normalized)
        
        scores[k] = score
   
    
    print(scores)
    maximum_id = scores.index(max(scores))


    max_value = max(scores)
    max_value_number = scores.count(max_value)

    if max_value_number > 1:
        max_indexes = [index for index, value in enumerate(scores) if value == max_value]
        maximum_id = random.choice(max_indexes)


    task_assigner.prova = task_assigner.assignments[drone_id].copy()

    
    new_assignment.append(targets_in_cluster[drone_id][maximum_id])
    task_assigner.assignments[drone_id] = new_assignment.copy()


    task_assigner.computing_score[drone_id] = False



    


def submit_task(task_assigner, drone_id, targets_to_patrol = None):

    task_assigner.get_logger().info("Submitting task for drone X3_%s" % drone_id)

    # while len(task_assigner.action_servers) == 0:
    #     time.sleep(1)
    
    while not task_assigner.action_servers[drone_id].wait_for_server(timeout_sec = 1.0):
        return

    task_assigner.idle[drone_id] = False

    tmp = []

    if not targets_to_patrol:

        global first_scenario
        if(first_scenario == False):
            while(task_assigner.prova == task_assigner.assignments[drone_id]):
                compute_score(task_assigner, drone_id)

        targets_to_patrol = task_assigner.assignments[drone_id].copy()

    patrol_task =  PatrollingAction.Goal()
    patrol_task.targets = targets_to_patrol

    patrol_future = task_assigner.action_servers[drone_id].send_goal_async(patrol_task)

    # This is a new construct for you. Basically, callbacks have no way of receiving arguments except
    # for the future itself. We circumvent such problem by creating an inline lambda functon which stores
    # the additional arguments ad-hoc and then calls the actual callback function
    patrol_future.add_done_callback(lambda future, d = drone_id : task_assigner.patrol_submitted_callback(future, d))


 
    




