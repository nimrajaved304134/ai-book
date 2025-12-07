---
sidebar_position: 1
---

# Lesson 1: Introduction to ROS 2

## Introduction
Welcome to the first lesson of Chapter 1 on ROS 2 (Robot Operating System 2). This lesson provides a foundational understanding of ROS 2, its architecture, and why it's essential for robotics development.

ROS 2 is a flexible framework for writing robot software that provides a collection of libraries and tools to help software developers create robot applications. It's an evolution from ROS 1, addressing many of the limitations and challenges of the original system, particularly in areas of security, real-time performance, and multiple robot systems.

## Concepts
ROS 2 is designed with several key concepts in mind:

1. **Middleware**: ROS 2 uses DDS (Data Distribution Service) as its underlying middleware, which enables distributed communication between different components of a robotic system.

2. **Nodes**: A node is a process that performs computation. In ROS 2, nodes are the fundamental building blocks of a robot application.

3. **Topics**: Topics are named buses over which nodes exchange messages. They enable asynchronous message passing between nodes.

4. **Services**: Services provide a synchronous request/reply communication pattern between nodes.

5. **Actions**: Actions are a more sophisticated form of communication that supports long-running requests with feedback and goal preemption.

6. **Packages**: Packages are the basic building blocks of ROS 2. They contain libraries, executables, and other resources needed for a specific functionality.

## Technical Deep Dive
The architecture of ROS 2 is based on the DDS standard, which provides a publish-subscribe pattern for communication. This allows for:

- **Decentralized Architecture**: Unlike ROS 1's master-slave model, ROS 2 uses a peer-to-peer discovery mechanism.
- **Quality of Service (QoS) Settings**: These allow fine-tuning of communication behavior based on the specific requirements of your system.
- **Security**: Native security features support authentication, access control, and encryption.

The DDS implementation provides several options:
- FastDDS (default)
- CycloneDDS
- RTI Connext DDS
- Eclipse iceoryx

## Diagrams
The ROS 2 architecture includes:
- DDS implementations acting as the middleware
- ROS 2 client libraries (rclcpp, rclpy)
- ROS 2 core tools (ros2 command line tools)


## Code Examples (Python/ROS 2)
Here's a simple example of a ROS 2 publisher node in Python:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises
1. Create a ROS 2 package named `my_robot_tutorial` with a publisher node that publishes your name and current timestamp at 2Hz frequency.

2. Modify the publisher to send a counter value instead of a hardcoded string. Create a subscriber node that listens to this counter and prints it to the console.

3. Design a simple ROS 2 service that takes two integers as input and returns their sum. Implement both the service server and client.

## Quiz
1. What does DDS stand for in the context of ROS 2?
   a) Data Distribution System
   b) Data Distribution Service
   c) Distributed Data System
   d) Digital Distribution Service

2. Which of the following is NOT a communication method in ROS 2?
   a) Topics
   b) Services
   c) Actions
   d) Protocols

3. True or False: ROS 2 uses a master-slave architecture like ROS 1.
   a) True
   b) False

## Summary
This lesson introduced the fundamental concepts of ROS 2, including its architecture based on DDS middleware, the key communication patterns (topics, services, actions), and how they enable distributed robotics applications. ROS 2 addresses limitations of ROS 1 with improved security, real-time performance, and multi-robot system support.

## Key Terms
- **DDS (Data Distribution Service)**: Middleware that implements the publish-subscribe pattern in ROS 2
- **Node**: A process that performs computation in ROS 2
- **Topic**: Named bus over which nodes exchange messages
- **Service**: Synchronous request/reply communication pattern
- **Action**: Advanced communication pattern with feedback and goal preemption
- **Package**: Basic building block containing libraries, executables, and resources
- **QoS (Quality of Service)**: Settings for fine-tuning communication behavior