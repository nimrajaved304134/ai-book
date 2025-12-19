---
title: "Introduction to ROS 2"
sidebar_label: "Lesson 1: Introduction to ROS 2"
---

# Lesson 1: Introduction to ROS 2

## Introduction

Welcome to the exciting world of robotics! In this first lesson, we'll lay the foundation for understanding Robot Operating System 2 (ROS 2), a flexible framework for writing robot software. While not a traditional operating system, ROS 2 provides the services you would expect from one, including hardware abstraction, low-level device control, implementation of commonly used functionality, message-passing between processes, and package management.

ROS 2 is the next generation of the popular ROS framework, designed specifically for production systems with improved security, real-time performance, and better multi-robot systems support. As you progress through this textbook, you'll see how ROS 2 serves as the backbone for developing sophisticated humanoid robots that can perceive, think, and act in complex environments.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Robot Operating System 2 (ROS 2) Overview

ROS 2 is a collection of software libraries and tools that help you build robot applications. It provides hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. ROS 2 was redesigned from the ground up to address the needs of production robot applications with:

- **Security**: Built-in security features for safe operation in production environments
- **Real-time Support**: Integration with real-time systems for time-critical applications
- **Multi-Robot Systems**: Native support for complex multi-robot scenarios
- **Middleware Flexibility**: Support for multiple DDS implementations

### Humanoid Robotics Context

In humanoid robotics, ROS 2 serves as the communication backbone between the various subsystems involved in human-like locomotion, manipulation, and interaction. This includes:

- Sensor fusion from cameras, IMUs, and joint encoders
- Control systems for joint actuators and balance maintenance
- AI perception pipelines that process vision and audio data
- Planning and navigation systems that enable movement through environments

## Detailed Technical Explanations

### Architecture Components

ROS 2 architecture is fundamentally based on a distributed network of nodes that communicate through topics, services, and actions:

1. **Nodes**: Processes that perform computation. In humanoid robots, nodes might control specific limbs, process sensor data, or execute high-level planning algorithms.

2. **Topics**: Named buses over which nodes exchange messages. Topics use a publish-subscribe communication pattern, ideal for continuous data streams like sensor readings.

3. **Services**: Request-response communication pattern where one node sends a request and another responds with data. Useful for single-shot operations like requesting robot calibration.

4. **Actions**: Goal-based communication pattern for long-running tasks that provide feedback. Actions are essential for humanoid robot behaviors that take time to complete, such as walking to a location.

### ROS 2 Middleware (DDS)

ROS 2 leverages Data Distribution Service (DDS) as its underlying communication middleware. DDS provides:

- **Quality of Service (QoS)**: Configurable settings for reliability, durability, and other communication characteristics
- **Discovery**: Automatic detection of nodes on the network
- **Publisher-Subscriber Pattern**: Decoupled communication between nodes

### Package Management

ROS 2 uses the `ament` build system and the `colcon` build tool for package management. Packages contain nodes, libraries, and other resources, organized in a standard structure that enables reuse and sharing across the robotics community.

## Code Examples

### Basic ROS 2 Publisher Node

Here's a simple publisher node that broadcasts robot status information:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        
        # Create publisher for robot status
        self.status_publisher = self.create_publisher(
            String, 
            'robot_status', 
            10
        )
        
        # Create publisher for joint states (for humanoid robot)
        self.joint_publisher = self.create_publisher(
            JointState, 
            'joint_states', 
            10
        )
        
        # Timer to publish data at regular intervals
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Counter for status messages
        self.status_counter = 0
        
        # Initialize joint state message
        self.joint_state = JointState()
        self.joint_state.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'right_shoulder_joint', 'right_elbow_joint'
        ]
        self.joint_state.position = [0.0] * len(self.joint_state.name)
        
    def timer_callback(self):
        # Publish robot status
        msg = String()
        msg.data = f'Robot status: Active - {self.status_counter} seconds'
        self.status_publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        
        # Publish joint states with simulated positions
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = [i * 0.1 + self.status_counter * 0.01 for i in range(len(self.joint_state.position))]
        self.joint_publisher.publish(self.joint_state)
        
        self.status_counter += 1

def main(args=None):
    rclpy.init(args=args)
    
    robot_status_publisher = RobotStatusPublisher()
    
    rclpy.spin(robot_status_publisher)
    
    # Destroy the node explicitly
    robot_status_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Basic ROS 2 Subscriber Node

Here's how to create a subscriber that receives and processes the published data:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class RobotStatusSubscriber(Node):
    def __init__(self):
        super().__init__('robot_status_subscriber')
        
        # Create subscription to robot status
        self.status_subscription = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10
        )
        self.status_subscription  # prevent unused variable warning
        
        # Create subscription to joint states
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10
        )
        self.joint_subscription  # prevent unused variable warning
        
    def status_callback(self, msg):
        self.get_logger().info(f'Received robot status: {msg.data}')
        
    def joint_callback(self, msg):
        # Process joint states for humanoid robot control
        self.get_logger().info(f'Received joint states for {len(msg.name)} joints')
        
        # Example: Print first 5 joint positions
        for i in range(min(5, len(msg.name))):
            self.get_logger().info(f'{msg.name[i]}: {msg.position[i]:.2f} rad')

def main(args=None):
    rclpy.init(args=args)
    
    robot_status_subscriber = RobotStatusSubscriber()
    
    rclpy.spin(robot_status_subscriber)
    
    # Destroy the node explicitly
    robot_status_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Diagrams

```
[Robot Operating System 2 Architecture Diagram]

ROS 2 Node 1 (Navigation)     ROS 2 Node 2 (Perception)     ROS 2 Node 3 (Control)
        |                              |                             |
        |----->[Topic: /cmd_vel]------->|                             |
        |                              |----->[Topic: /camera_data]-->|
        |                              |                             |
        |<-----[Topic: /odom]-----------|                             |
        |                              |                             |
        |<-----[Action: /move_to_goal]--|                             |
        |                              |                             |
        |                              |                             |
DDS Communication Layer (RMW) <----------------------------------------> Hardware Interface Layer
        |
        |----->[Service: /get_map]-----> Map Server
        |                              |
        |<-----[Service: /set_pose]----- AMCL Node

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![ROS 2 Architecture](/img/ros2-architecture.png)`.

## Hands-on Exercises

1. **Exercise 1**: Set up your ROS 2 development environment on your local machine. Follow the official installation guide for your operating system (Ubuntu, Windows, or macOS). Verify the installation by running `ros2 topic list` and `ros2 node list` in different terminals.

2. **Exercise 2**: Create a new ROS 2 package called `humanoid_tutorials` using `ros2 pkg create --build-type ament_python humanoid_tutorials`. Inside this package, create the publisher and subscriber nodes from the code examples above and test their communication.

3. **Exercise 3**: Create a custom message type for your humanoid robot that includes fields for joint positions, velocities, and effort for all degrees of freedom. Generate the message definition and use it in a publisher/subscriber pair.

4. **Exercise 4**: Use the `rqt_graph` tool to visualize the communication graph between your nodes. Experiment with launching multiple instances of your publisher and subscriber to observe how ROS 2 handles multiple nodes with the same topic subscriptions.

## Quiz

1. What is the primary difference between ROS 2 and its predecessor ROS 1?
   - a) ROS 2 is faster than ROS 1
   - b) ROS 2 uses DDS for communication while ROS 1 used a centralized master
   - c) ROS 2 is written in Python while ROS 1 was in C++
   - d) ROS 2 supports only Linux while ROS 1 supported multiple platforms

2. In ROS 2, what is the purpose of Quality of Service (QoS) profiles?
   - a) To limit the amount of data being transmitted
   - b) To configure communication characteristics like reliability and durability
   - c) To ensure nodes run faster
   - d) To reduce memory usage in ROS applications

3. Which communication pattern would be most appropriate for implementing a humanoid robot walking to a specific location?
   - a) Topic (publish/subscribe)
   - b) Service (request/response)
   - c) Action (goal-based with feedback)
   - d) Parameter server

4. True/False: ROS 2 packages must be written in C++ to be part of the ecosystem.
   - Answer: _____

5. What does DDS stand for in the context of ROS 2?
   - a) Direct Data Sharing
   - b) Distributed Data Service
   - c) Data Distribution Service
   - d) Decentralized Data System

## Summary

In this lesson, you've learned the fundamental concepts of ROS 2, the next-generation robot middleware framework. We covered its architecture, core components (nodes, topics, services, and actions), and how it serves as the communication backbone for humanoid robots. You've also seen practical code examples demonstrating publisher and subscriber nodes that could be used in humanoid robot applications.

ROS 2 provides the essential infrastructure for complex robot systems, enabling proper separation of concerns between different functionality modules while maintaining reliable communication. This foundation is critical for the advanced robotics applications we'll explore in subsequent lessons.

## Key Terms

- **DDS (Data Distribution Service)**: The middleware technology underlying ROS 2 for reliable, real-time data exchange
- **Node**: A process that performs computation in the ROS system
- **Topic**: A named bus over which nodes exchange messages using publish/subscribe communication
- **Service**: A request/reply communication pattern in ROS
- **Action**: A goal-based communication pattern with feedback and status updates
- **Package**: A modular unit containing nodes, libraries, and other resources in ROS
- **QoS (Quality of Service)**: Configurable settings that define how messages are handled by the middleware
- **RMW (ROS Middleware)**: The layer that abstracts the underlying DDS implementation
- **Humanoid Robot**: A robot with a body structure similar to that of a human
- **Joint State**: A message type containing information about robot joint positions, velocities, and efforts