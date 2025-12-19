---
title: "ROS 2 Nodes, Topics, and Services"
sidebar_label: "Lesson 2: ROS 2 Nodes, Topics, and Services"
---

# Lesson 2: ROS 2 Nodes, Topics, and Services

## Introduction

In this lesson, we'll dive deeper into the fundamental building blocks of ROS 2: nodes, topics, and services. These components form the core of ROS 2's distributed architecture and are essential for developing humanoid robots that can coordinate multiple sensors, actuators, and high-level decision-making systems. Understanding how these elements work is crucial for creating robust and maintainable robotic applications.

For humanoid robots, this architecture is particularly important because these systems must simultaneously process sensory data, control dozens of joints for balance and movement, respond to environmental stimuli, and execute complex behaviors. The decoupled nature of nodes enables different teams to develop specialized capabilities that can be integrated seamlessly.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Nodes in ROS 2

A node is the fundamental unit of computation in ROS 2. Each node performs a specific function and communicates with other nodes using topics, services, or actions. In humanoid robotics, nodes might include:

- **Sensor Processing Nodes**: Processing data from cameras, LIDAR, IMUs, and force/torque sensors
- **Control Nodes**: Managing low-level actuator control for joints and motors
- **Planning Nodes**: Computing movement trajectories and gait patterns for walking
- **AI Perception Nodes**: Running computer vision and audio processing algorithms
- **Behavior Nodes**: Coordinating high-level robot behaviors and responses

### Topics and Message Passing

Topics enable asynchronous communication between nodes using a publish-subscribe pattern. This is ideal for continuous data streams like:

- Camera image feeds for vision processing
- Joint position and velocity information
- IMU data for balance control
- Robot odometry and localization data

### Services and Request-Response Patterns

Services provide synchronous request-response communication. This pattern is suitable for:

- Calibration procedures
- Service requests that return a single result
- Configuration changes
- Diagnostic queries

### Humanoid Robotics Integration

In humanoid robots, proper use of nodes, topics, and services enables modular development where:

- Different teams can work on separate components without conflicts
- Individual components can be tested independently
- The system remains robust when components fail or are updated
- Different algorithms can be swapped in and out for comparison

## Detailed Technical Explanations

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes states like unconfigured, inactive, active, and finalized. Understanding these states is important for humanoid robots that need to manage power consumption and ensure safe operation:

- **Unconfigured**: Node is loaded but not configured
- **Inactive**: Node is configured but not yet activated
- **Active**: Node is running and processing callbacks
- **Finalized**: Node is shut down

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning communication characteristics:

- **Reliability**: Whether messages are guaranteed to be delivered (Reliable vs Best Effort)
- **Durability**: Whether messages are kept for late-joining subscribers (Transient Local vs Volatile)
- **History**: How many messages are kept for late-joining subscribers
- **Deadline**: Maximum time between messages
- **Liveliness**: How to detect if a publisher is alive

### Topic Types and Custom Messages

ROS 2 supports various standard message types while allowing custom message definitions:

- **Common Types**: `std_msgs`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`
- **Humanoid-Specific Types**: `sensor_msgs/JointState`, `control_msgs/JointTrajectory`, `humanoid_msgs/*`

## Code Examples

### Advanced Node with Parameters

Here's an example of a humanoid robot control node with configurable parameters:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('enable_balance_control', True)
        self.declare_parameter('max_velocity', 1.0)
        
        self.control_frequency = self.get_parameter('control_frequency').value
        self.enable_balance_control = self.get_parameter('enable_balance_control').value
        self.max_velocity = self.get_parameter('max_velocity').value
        
        # Create QoS profile for joint state data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers and subscribers
        self.joint_command_pub = self.create_publisher(
            JointState, 
            'joint_commands', 
            qos_profile
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)
        
        # Internal state
        self.current_joint_state = None
        self.target_velocity = Twist()
        
        self.get_logger().info(f'Humanoid controller initialized with {self.control_frequency}Hz control rate')
    
    def joint_state_callback(self, msg):
        self.current_joint_state = msg
        
    def cmd_vel_callback(self, msg):
        self.target_velocity = msg
        self.get_logger().info(f'Received velocity command: linear={msg.linear.x}, angular={msg.angular.z}')
    
    def control_loop(self):
        if self.current_joint_state is None:
            return
            
        # Create a command message based on current state and target
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'
        
        # For demonstration, copy current joint names and create target positions
        cmd_msg.name = self.current_joint_state.name.copy()
        
        # Calculate target positions based on velocity command (simplified)
        current_positions = np.array(self.current_joint_state.position)
        target_positions = current_positions.copy()
        
        # Apply simple control logic based on velocity command
        if self.target_velocity.linear.x > 0.1:  # Forward command
            # Increment hip joints to simulate step forward
            for i, name in enumerate(cmd_msg.name):
                if 'hip' in name:
                    target_positions[i] += 0.01  # Small step
                elif 'ankle' in name:
                    target_positions[i] -= 0.005  # Adjust ankle for balance
        elif self.target_velocity.linear.x < -0.1:  # Backward command
            for i, name in enumerate(cmd_msg.name):
                if 'hip' in name:
                    target_positions[i] -= 0.01
                elif 'ankle' in name:
                    target_positions[i] += 0.005
        
        cmd_msg.position = target_positions.tolist()
        cmd_msg.velocity = [0.0] * len(cmd_msg.position)  # Zero velocity for simplicity
        cmd_msg.effort = [0.0] * len(cmd_msg.position)  # Zero effort for simplicity
        
        # Publish command
        self.joint_command_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    
    controller = HumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down due to user interrupt')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Server and Client Example

Here's an example of a service for calibrating the humanoid robot's sensors:

```python
# calibration_service.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger
from sensor_msgs.msg import JointState

class CalibrationService(Node):
    def __init__(self):
        super().__init__('calibration_service')
        
        # Create service
        self.srv = self.create_service(
            Trigger, 
            'calibrate_robot', 
            self.calibrate_robot_callback
        )
        
        # Publisher for zeroing joint positions
        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', qos_profile)
        
        self.get_logger().info('Calibration service ready')
    
    def calibrate_robot_callback(self, request, response):
        self.get_logger().info('Calibration service called')
        
        # Perform calibration (in a real robot this would involve complex procedures)
        # For this example, we'll just zero out all joint positions
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'right_shoulder_joint', 'right_elbow_joint'
        ]
        joint_msg.position = [0.0] * len(joint_msg.name)
        joint_msg.velocity = [0.0] * len(joint_msg.name)
        joint_msg.effort = [0.0] * len(joint_msg.name)
        
        # Publish zero positions
        self.joint_pub.publish(joint_msg)
        
        # Simulate calibration time
        # In a real implementation, this might involve waiting for confirmation from encoders
        import time
        time.sleep(1.0)  # Simulate time for calibration process
        
        response.success = True
        response.message = 'Robot calibration completed successfully'
        
        self.get_logger().info(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    
    calibration_service = CalibrationService()
    
    try:
        rclpy.spin(calibration_service)
    except KeyboardInterrupt:
        calibration_service.get_logger().info('Shutting down calibration service')
    finally:
        calibration_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

```python
# calibration_client.py
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

class CalibrationClient(Node):
    def __init__(self):
        super().__init__('calibration_client')
        
        # Create client
        self.cli = self.create_client(Trigger, 'calibrate_robot')
        
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Calibration service not available, waiting again...')
        
        self.req = Trigger.Request()
    
    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    
    calibration_client = CalibrationClient()
    
    print('Sending calibration request...')
    response = calibration_client.send_request()
    
    if response is not None:
        print(f'Result of calibration: {response.success}, message: {response.message}')
    else:
        print('Calibration service call failed')
    
    calibration_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Diagrams

```
[Node Communication Patterns in ROS 2]

[Humanoid Robot Control Node]
         |
         | (Topic: /joint_states) - Publishes current joint positions
         | (Topic: /odom) - Publishes robot odometry
         |
         v
[Perception Node] 
         |
         | (Topic: /camera_data) - Publishes camera images
         | (Topic: /imu_data) - Publishes inertial measurements
         |
         v
[Planning Node]
         |
         | (Service: /get_plan) - Requests path planning
         | (Action: /move_base) - Requests navigation action
         |
         v
[Behavior Node]
         |
         | (Topic: /cmd_vel) - Publishes velocity commands
         | (Service: /calibrate_robot) - Provides calibration service
         |
         v
DDS Communication Layer (RMW)

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![ROS 2 Communication](/img/ros2-communication.png)`.

## Hands-on Exercises

1. **Exercise 1**: Create a parameter server for your humanoid robot that allows changing control parameters at runtime. Use the `declare_parameter()` method to define parameters for joint limits, control gains, and safety thresholds. Test changing parameters using the `ros2 param` command-line tool.

2. **Exercise 2**: Implement a service that allows remote configuration of your robot's walking gait parameters (step height, step length, walking speed). Create both a service server that handles configuration requests and a client that can send these requests.

3. **Exercise 3**: Create a custom message type for humanoid robot balance information that includes center of mass position, support polygon vertices, and stability metrics. Use this message to create a balance monitoring node that subscribes to joint states and publishes balance status.

4. **Exercise 4**: Use ROS 2's introspection tools to analyze the communication patterns of your nodes. Use `ros2 topic list`, `ros2 node list`, `ros2 node info <node_name>`, and `rqt_graph` to understand the data flow in your system.

## Quiz

1. What is the main difference between ROS 2 topics and services?
   - a) Topics are faster than services
   - b) Topics use publish/subscribe pattern, services use request/response pattern
   - c) Topics are for sensors, services are for actuators
   - d) There is no difference between topics and services

2. Which QoS policy specifies whether messages are guaranteed to be delivered?
   - a) History
   - b) Deadline
   - c) Reliability
   - d) Liveliness

3. What are the four main states in the ROS 2 node lifecycle?
   - a) Init, Start, Run, Stop
   - b) Unconfigured, Inactive, Active, Finalized
   - c) Boot, Ready, Working, Shutdown
   - d) Loading, Configured, Running, Terminated

4. True/False: In ROS 2, nodes can only communicate using topics and services.
   - Answer: _____

5. What is the purpose of the `declare_parameter()` method in ROS 2?
   - a) To create a new topic
   - b) To define configurable values that can be set at runtime
   - c) To register a new node
   - d) To publish messages to a topic

## Summary

In this lesson, we explored the core communication patterns in ROS 2: nodes, topics, and services. We learned how these components enable modular development and robust communication in complex systems like humanoid robots. We also examined how to properly configure these components using QoS settings and parameters to ensure reliable operation.

With this foundation, you're now ready to understand more advanced patterns like actions, which we'll cover in the next lesson. The modular architecture of ROS 2 enables the development of sophisticated humanoid robots where different capabilities can be developed, tested, and maintained independently.

## Key Terms

- **Node**: A process that performs computation in the ROS system
- **Topic**: Named buses over which nodes exchange messages using publish-subscribe
- **Service**: Request-response communication pattern in ROS
- **QoS (Quality of Service)**: Configurable settings that define how messages are handled
- **Reliability Policy**: QoS setting specifying whether messages must be delivered
- **Durability Policy**: QoS setting specifying message persistence for late-joining subscribers
- **Parameter**: Configurable values that can be set at runtime in ROS 2 nodes
- **Callback**: Function executed when a ROS 2 message is received
- **Publisher**: Component that sends messages on a topic
- **Subscriber**: Component that receives messages on a topic