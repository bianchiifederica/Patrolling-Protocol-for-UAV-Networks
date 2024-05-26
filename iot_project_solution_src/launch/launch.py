from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def spawn_controllers(context, *args, **kwargs):

    nodes_to_spawn = []

    no_drones = LaunchConfiguration('no_drones').perform(context)
    no_drones = int(no_drones)

    for d in range(no_drones):
        nodes_to_spawn.append(
            Node(
                package='iot_project_solution_src',
                executable='drone_controller',
                namespace="X3_%d" % d,
                output='screen',
                emulate_tty=True,
                name='controller'
            )
        )
    return nodes_to_spawn

def generate_launch_description():

        # Spawn a task assigner first.
    # The task assigner should be able to communicate with
    # all the three drones and submit the correct goals to achieve
    task_assigner_node = Node(
        package='iot_project_solution_src',
        executable='task_assigner',
        output='screen',
        emulate_tty=True,
        name='task_assigner'
    )
    

    # Then spawn all the nodes. Each of them in his own namespace


    return LaunchDescription([task_assigner_node, OpaqueFunction(function=spawn_controllers)])
