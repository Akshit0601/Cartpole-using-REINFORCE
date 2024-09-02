#!/usr/bin/env python3

import warnings
from time import sleep
import rclpy.parameter
warnings.filterwarnings("ignore",category=DeprecationWarning)
import rclpy.node
import xacro
import rclpy
from gazebo_msgs.srv import SpawnEntity,DeleteEntity
from rclpy.executors import MultiThreadedExecutor
from gazebo_msgs.srv import SpawnEntity
from controller_manager_msgs.srv import LoadController, ConfigureController,  SwitchController
from std_srvs.srv import Empty

xacro_path = '/home/akshit/gazebo_ros2_control/src/gazebo_ros2_control_demos/urdf/test_pendulum_effort.xacro.urdf'
doc = xacro.parse(open(xacro_path))
xacro.process_doc(doc)



def reset_sim():
    global doc
    nh = rclpy.create_node("tester")

    unpause = nh.create_client(Empty,'/unpause_physics')
    future_pause = unpause.call_async(Empty.Request())
    rclpy.spin_until_future_complete(nh, future_pause)

    del_client = nh.create_client(DeleteEntity,"/delete_entity")
    while not del_client.wait_for_service(timeout_sec=1.0):
        print("deltion service not available")
    req_del = DeleteEntity.Request()
    req_del.name = 'cartpole'
    future = del_client.call_async(req_del)
    rclpy.spin_until_future_complete(nh,future=future)
    print(future.result())

    clien = nh.create_client(SpawnEntity,'/spawn_entity')
    while not clien.wait_for_service(timeout_sec=1):
        nh.get_logger().info("waiting for spawn service1")
    spawn_req = SpawnEntity.Request()
    spawn_req.xml = doc.toxml()
    spawn_req.initial_pose.position.x = 0.0
    spawn_req.initial_pose.position.y = 0.0
    spawn_req.initial_pose.orientation.z = 0.0
    spawn_req.name = 'cartpole'
    f = clien.call_async(spawn_req)
    rclpy.spin_until_future_complete(nh,future=f)
    print(f.result())

    client3 = nh.create_client(ConfigureController,'/controller_manager/configure_controller')
    while not client3.wait_for_service(timeout_sec=1):
        nh.get_logger().info("configuration server not available")

    client2 = nh.create_client(LoadController,'/controller_manager/load_controller')
    while not client2.wait_for_service(timeout_sec=1):
        nh.get_logger().info("loading server unavailable")

    client4 = nh.create_client(SwitchController,'/controller_manager/switch_controller')
    while not client4.wait_for_service(timeout_sec=1):
        nh.get_logger().info("activation server unavailable")

    print('loading')
    spawn_req2 = LoadController.Request()
    spawn_req2.name = 'effort_controller'
    f1 = client2.call_async(spawn_req2)
    rclpy.spin_until_future_complete(nh, future=f1)
    print(f1.result())



    print('configuring')
    req3 = ConfigureController.Request()
    req3.name = 'effort_controller'
    f2 = client3.call_async(req3)
    rclpy.spin_until_future_complete(nh,f2)
    print(f2.result())

    print('activating')
    req4 = SwitchController.Request()
    req4.activate_controllers = ['effort_controller']
    f3 = client4.call_async(req4)
    rclpy.spin_until_future_complete(nh,f3)
    print(f3.result())

    print('end')
    
    pause = nh.create_client(Empty,'/pause_physics')
    future_pause = pause.call_async(Empty.Request())
    rclpy.spin_until_future_complete(nh, future_pause)
    print(future_pause.result())

    nh.destroy_node()

