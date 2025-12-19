---
title: "Gazebo Simulation Environment"
sidebar_label: "Lesson 1: Gazebo Simulation Environment"
---

# Lesson 1: Gazebo Simulation Environment

## Introduction

Welcome to Module 2, where we explore simulation environments essential for developing and testing humanoid robots. In this first lesson, we'll focus on Gazebo, one of the most widely used robotics simulators in the ROS ecosystem. Gazebo provides realistic physics simulation, sensor models, and visualization capabilities that allow developers to test their humanoid robots in virtual environments before deploying them in the real world.

Simulation is a critical part of humanoid robotics development because these robots are expensive, complex, and potentially dangerous if not properly tested. Gazebo enables safe, cost-effective testing of locomotion algorithms, control strategies, and interaction scenarios without risk to equipment or people. It also allows for accelerated testing, where months of real-world experience can be simulated in hours.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Physics Simulation for Humanoid Robots

Gazebo uses advanced physics engines (like ODE, Bullet, and DART) to simulate the behavior of humanoid robots:

- **Rigid Body Dynamics**: Accurate simulation of robot body parts and their interactions
- **Contact Physics**: Modeling forces when the robot touches surfaces or objects
- **Friction and Inertia**: Realistic simulation of how robots move and balance
- **Actuator Modeling**: Simulation of servo motors, hydraulics, and other actuators

### Sensor Simulation

For humanoid robots to operate effectively in simulation:

- **Camera Simulation**: RGB, depth, and stereo cameras for vision processing
- **IMU Simulation**: Inertial measurement units for balance and orientation
- **Force/Torque Sensors**: Simulation of contact forces at joints and feet
- **LIDAR Simulation**: Range sensors for environment mapping and navigation

### Realistic Environment Modeling

Gazebo allows for detailed environment simulation:

- **Terrain Generation**: Modeling natural and artificial terrains
- **Object Physics**: Interactable objects with realistic properties
- **Lighting Conditions**: Day/night cycles and varying illumination
- **Weather Effects**: Though limited, some basic environmental effects

### Integration with ROS

Gazebo is tightly integrated with ROS:

- **Gazebo ROS Packages**: Bridge between Gazebo and ROS2 systems
- **Message Passing**: Direct communication between simulated and real robots
- **TF Frames**: Proper transformation of coordinate systems
- **URDF Support**: Direct loading of robot descriptions

## Detailed Technical Explanations

### Gazebo Architecture

Gazebo consists of several key components:

1. **Gazebo Server**: Core simulation engine that handles physics and sensor updates
2. **Gazebo Client**: Interface that provides visualization and user controls
3. **Plugins System**: Extensible framework for custom sensors and controllers
4. **Model Database**: Repository of robots, objects, and environments

### URDF and SDF Formats

- **URDF (Unified Robot Description Format)**: Used to define robot structure in ROS, can be imported into Gazebo
- **SDF (Simulation Description Format)**: Gazebo's native format that supports more simulation-specific features

### Physics Engine Integration

Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default engine, good general-purpose physics
- **Bullet**: More robust for complex contact scenarios
- **DART**: Advanced physics with better handling of articulated bodies

## Code Examples

### Basic Gazebo Launch File

```xml
<!-- launch/humanoid_gazebo.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_humanoid_description = FindPackageShare('humanoid_description')
    pkg_humanoid_gazebo = FindPackageShare('humanoid_gazebo')
    
    # World file
    world_file = PathJoinSubstitution([
        pkg_humanoid_gazebo,
        'worlds',
        'humanoid_world.world'
    ])

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            pkg_gazebo_ros,
            '/launch',
            '/gazebo.launch.py'
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true',
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                pkg_humanoid_description,
                'urdf',
                'humanoid.urdf'
            ])
        }]
    )

    # Spawn entity node to add robot to simulation
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0', 
            '-z', '1.0'  # Start slightly above ground
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

### URDF for a Simple Humanoid Robot

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="Blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="Red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="White">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="White"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.35"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="White"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="Blue"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0.0 -0.15 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="Blue"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0.0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="Red"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0.0 -0.15 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="Red"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.05 -0.1 -0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="Blue"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0.0 -0.2 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="200" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="Blue"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.008"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0.0 -0.2 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="White"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="-0.05 -0.1 -0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="Red"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0.0 -0.2 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="200" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="Red"/>
      <origin rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.008"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0.0 -0.2 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="White"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Gazebo Materials -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_lower_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_lower_arm">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="left_thigh">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_shin">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_foot">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="right_thigh">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_shin">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_foot">
    <material>Gazebo/White</material>
  </gazebo>
</robot>
```

### Gazebo Controller Configuration

```yaml
# config/humanoid_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    humanoid_joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    humanoid_position_controller:
      type: position_controllers/JointGroupPositionController

    humanoid_balance_controller:
      type: humanoid_controllers/BalanceController

humanoid_position_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint
      - left_shoulder_joint
      - left_elbow_joint
      - right_shoulder_joint
      - right_elbow_joint

humanoid_balance_controller:
  ros__parameters:
    com_height: 0.8  # Center of mass height
    control_frequency: 100.0
    balance_kp: 50.0
    balance_kd: 10.0
    max_tilt_angle: 0.2
```

## Diagrams

```
[Gazebo Simulation Architecture for Humanoid Robots]

[Real Robot] <===> [ROS Bridge] <===> [Gazebo Simulator]
    |                    |                   |
    |                    |                   |
    |--------------->[Controllers] <----------|
    |                    |
    |--------------->[Sensors]
    |                    |
    |--------------->[Physics Engine (ODE/Bullet/DART)]
                         |
    |--------------->[Environment Models]
    |                    |
    |--------------->[Sensor Simulations]
                         |
    |--------------->[Visualization (OGRE)]

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![Gazebo Architecture](/img/gazebo-architecture.png)`.

## Hands-on Exercises

1. **Exercise 1**: Install Gazebo and create a simple launch file that loads the URDF model shown in the code examples. Run the simulation and verify that the robot appears correctly in the Gazebo environment.

2. **Exercise 2**: Create a more detailed humanoid model with realistic dimensions based on human proportions. Add proper joint limits and mimic constraints to simulate human-like movement capabilities.

3. **Exercise 3**: Implement a controller plugin that applies torques to the joints to make the humanoid robot stand up from an initial position where it's lying down.

4. **Exercise 4**: Create a custom world file with obstacles and test the robot's ability to navigate around them. Add sensors like a camera or LIDAR to perceive the environment.

## Quiz

1. Which physics engines does Gazebo support?
   - a) ODE only
   - b) ODE, Bullet, and DART
   - c) Bullet and PhysX
   - d) Only custom engines

2. What does URDF stand for?
   - a) Universal Robot Definition Format
   - b) Unified Robot Description Format
   - c) Universal Robot Documentation Framework
   - d) Unified Robot Development Framework

3. What is the primary purpose of simulation in robotics development?
   - a) To replace real robots entirely
   - b) To test algorithms safely and cost-effectively before real-world deployment
   - c) To make robots look more appealing
   - d) To reduce computational requirements

4. True/False: Gazebo can simulate sensors like cameras and IMUs.
   - Answer: _____

5. Which ROS package provides the interface between Gazebo and ROS?
   - a) robot_state_publisher
   - b) gazebo_ros
   - c) tf2_ros
   - d) joint_state_publisher

## Summary

In this lesson, we explored Gazebo, the premier simulation environment for robotics development. We covered its key features, architecture, and how it enables safe and efficient development of humanoid robots. We looked at how to define robot models in URDF, launch simulations with appropriate controllers, and configure the physics and visualization parameters.

Gazebo is particularly valuable for humanoid robotics because it allows developers to test complex locomotion, balance, and manipulation algorithms without the risk and expense of physical testing. The ability to simulate realistic physics, sensors, and environments makes it an essential tool in the humanoid robot development pipeline.

In the next lesson, we'll explore Unity as another powerful simulation platform, particularly for more visually rich environments and human-robot interaction scenarios.

## Key Terms

- **Gazebo**: A physics-based robot simulation environment
- **URDF**: Unified Robot Description Format for defining robot models
- **SDF**: Simulation Description Format, Gazebo's native format
- **Physics Engine**: Software that simulates rigid body dynamics and collisions
- **Joint State Publisher**: ROS node that publishes joint position data
- **Robot Hardware Interface**: Interface between simulation and control algorithms
- **Model Database**: Repository of pre-built robot and environment models
- **ODE**: Open Dynamics Engine, a physics simulation library
- **Bullet**: Physics simulation library used by Gazebo
- **DART**: Dynamic Animation and Robotics Toolkit physics engine