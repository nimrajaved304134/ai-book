---
title: "ROS 2 Actions and Advanced Communication"
sidebar_label: "Lesson 3: ROS 2 Actions and Advanced Communication"
---

# Lesson 3: ROS 2 Actions and Advanced Communication

## Introduction

In this lesson, we'll explore ROS 2 actions, which are one of the most important communication patterns for humanoid robotics. Actions enable the implementation of long-running tasks with feedback, essential for humanoid behaviors like walking, manipulation, and navigation. We'll also cover advanced communication techniques that are particularly relevant for complex robotic systems.

Actions are particularly important in humanoid robotics because many fundamental robot behaviors take time to complete and require continuous feedback. Unlike services, which are synchronous and single-request/single-response, actions are asynchronous and can provide ongoing feedback about the progress of long-running tasks. This is crucial for humanoid robots that need to maintain balance and awareness while executing complex movements.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Actions in Humanoid Robotics

Actions are ideal for humanoid robot tasks that:

- Take a significant amount of time to complete
- Need to report progress during execution
- Might be preempted or canceled
- Require goal monitoring and feedback

Examples include:
- Walking to a specific location
- Manipulating an object with the robot's hands
- Executing a complex dance routine or choreographed movement
- Navigating through a crowded environment

### Action Structure

An action definition consists of three parts:
1. **Goal**: What the action should accomplish
2. **Result**: What happened when the goal was achieved
3. **Feedback**: Progress updates during goal execution

### Advanced Communication Considerations

For humanoid robots, communication patterns must account for:
- **Robustness**: Handling failures gracefully without falling
- **Timing**: Meeting real-time constraints for balance control
- **Coordination**: Multiple subsystems working together
- **Safety**: Graceful degradation when components fail

## Detailed Technical Explanations

### Action Message Types

Every action definition automatically creates three message types:
- `{ActionName}Goal`: Specifies the goal to achieve
- `{ActionName}Result`: Contains the result of achieving the goal
- `{ActionName}Feedback`: Provides progress updates during execution

### Action State Machine

Actions follow a state machine pattern:
- **PENDING**: Goal accepted but not yet started
- **ACTIVE**: Goal is currently being processed
- **RECALLING**: Goal being recalled before execution
- **REJECTED**: Goal rejected by the server
- **PREEMPTING**: Goal being replaced by a higher priority goal
- **PREEMPTED**: Goal was interrupted by a higher priority goal
- **SUCCEEDED**: Goal was achieved successfully
- **ABORTED**: Goal execution failed
- **RECALLING**: Goal being recalled after execution started
- **RECALLED**: Goal successfully recalled after execution started

### Advanced Topics: Events and Lifecycles

For humanoid robots requiring high reliability:
- **Node lifecycles**: Managing when components are ready
- **Event handling**: Reacting to special conditions during operation
- **Service composition**: Combining multiple services for complex tasks

## Code Examples

### Action Server for Humanoid Walking

Here's an implementation of a walking action server for a humanoid robot:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from builtin_interfaces.msg import Duration as DurationMsg
from humanoid_msgs.action import WalkToGoal  # Custom action message
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import time
import math


class HumanoidWalkActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_walk_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            WalkToGoal,
            'walk_to_goal',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Publishers for controlling the robot
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Subscribers for feedback
        self.odom_sub = self.create_subscription(
            Float64MultiArray,  # Simplified for example
            'robot_position', 
            self.odom_callback, 
            10
        )
        
        # Internal state
        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.current_goal = None
        self._goal_handle = None
        
        self.get_logger().info('Humanoid walk action server initialized')

    def odom_callback(self, msg):
        # Update current position from odometry
        if len(msg.data) >= 3:
            self.current_position = [msg.data[0], msg.data[1], msg.data[2]]

    def goal_callback(self, goal_request):
        # Accept all goals for now
        self.get_logger().info(f'Received goal request: ({goal_request.target_position.x}, {goal_request.target_position.y})')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept all cancel requests
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the walking action"""
        self.get_logger().info('Executing walk to goal...')
        self._goal_handle = goal_handle
        
        # Get goal position
        target_x = goal_handle.request.target_position.x
        target_y = goal_handle.request.target_position.y
        target_tolerance = goal_handle.request.tolerance
        
        # Calculate initial distance
        start_x, start_y = self.current_position[0], self.current_position[1]
        remaining_distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        # Initialize feedback
        feedback_msg = WalkToGoal.Feedback()
        result = WalkToGoal.Result()
        
        # Control loop for walking
        while remaining_distance > target_tolerance:
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal was canceled'
                self.get_logger().info('Goal was canceled')
                return result
            
            # Calculate distance to goal
            current_x, current_y = self.current_position[0], self.current_position[1]
            remaining_distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # Calculate direction to goal
            direction_x = target_x - current_x
            direction_y = target_y - current_y
            distance_to_goal = math.sqrt(direction_x**2 + direction_y**2)
            
            # Normalize direction vector
            if distance_to_goal > 0:
                direction_x /= distance_to_goal
                direction_y /= distance_to_goal
            else:
                # Already at goal
                break
            
            # Create velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = min(0.2, distance_to_goal) * direction_x  # Move toward target
            cmd_vel.linear.y = min(0.2, distance_to_goal) * direction_y  # Side movement if needed
            cmd_vel.angular.z = 0.0  # Simplified for this example
            
            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)
            
            # Update feedback
            feedback_msg.distance_to_goal = remaining_distance
            feedback_msg.remaining_time = DurationMsg(sec=int(remaining_distance / 0.1))  # Rough time estimate
            goal_handle.publish_feedback(feedback_msg)
            
            # Log progress
            self.get_logger().info(f'Distance to goal: {remaining_distance:.2f}m')
            
            # Sleep briefly to allow other processes
            time.sleep(0.1)
            
            # Check for goal cancellation during execution
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal was canceled during execution'
                self.get_logger().info('Goal was canceled during execution')
                return result
        
        # Goal reached
        goal_handle.succeed()
        result.success = True
        result.message = f'Reached goal position ({target_x}, {target_y})'
        
        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        self.get_logger().info(f'Successfully reached goal: {result.message}')
        return result


def main(args=None):
    rclpy.init(args=args)
    
    walk_action_server = HumanoidWalkActionServer()
    
    try:
        rclpy.spin(walk_action_server)
    except KeyboardInterrupt:
        walk_action_server.get_logger().info('Shutting down walk action server')
    finally:
        walk_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Action Client for Humanoid Walking

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from humanoid_msgs.action import WalkToGoal
from geometry_msgs.msg import Point

import time


class HumanoidWalkActionClient(Node):
    def __init__(self):
        super().__init__('humanoid_walk_action_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            WalkToGoal,
            'walk_to_goal'
        )

    def send_goal(self, x, y, tolerance=0.1):
        # Wait for action server to be available
        self._action_client.wait_for_server()
        
        # Create goal message
        goal_msg = WalkToGoal.Goal()
        goal_msg.target_position = Point(x=x, y=y, z=0.0)
        goal_msg.tolerance = tolerance
        
        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')
        
        # Shutdown after receiving result
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback received: {feedback.distance_to_goal:.2f}m to goal, '
            f'estimated time remaining: {feedback.remaining_time.sec}s'
        )


def main(args=None):
    rclpy.init(args=args)
    
    action_client = HumanoidWalkActionClient()
    
    # Send a goal to walk to position (2.0, 1.0)
    action_client.send_goal(2.0, 1.0)
    
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Shutting down action client')


if __name__ == '__main__':
    main()
```

### Advanced Communication Pattern: Publisher with QoS Configuration

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import numpy as np

class AdvancedPublisher(Node):
    def __init__(self):
        super().__init__('advanced_publisher')
        
        # Create different QoS profiles for different types of data
        # For safety-critical joint data (e.g., position and effort)
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # For non-critical status updates
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers with appropriate QoS
        self.joint_pub = self.create_publisher(JointState, 'joint_states', safety_qos)
        self.status_pub = self.create_publisher(JointState, 'robot_status', status_qos)
        
        # Timer for publishing
        self.timer = self.create_timer(0.01, self.publish_data)  # 100 Hz for joints
        
        # Initialize joint data
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 
            'right_shoulder_joint', 'right_elbow_joint',
            'head_pan_joint', 'head_tilt_joint'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        self.publish_counter = 0
        
    def publish_data(self):
        # Update joint positions with a simple oscillating pattern
        t = self.publish_counter * 0.01  # Time in seconds
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = 0.5 * math.sin(t * 2 * math.pi * 0.5 + i * 0.1)
            self.joint_velocities[i] = 0.5 * 2 * math.pi * 0.5 * math.cos(t * 2 * math.pi * 0.5 + i * 0.1)
        
        # Publish joint states with high reliability
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = self.joint_names
        joint_msg.position = self.joint_positions
        joint_msg.velocity = self.joint_velocities
        joint_msg.effort = self.joint_efforts  # Simplified: setting to zero
        
        self.joint_pub.publish(joint_msg)
        
        # Publish status with lower reliability requirements
        status_msg = JointState()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.header.frame_id = 'base_link'
        status_msg.name = ['status_flag']
        status_msg.position = [float(self.publish_counter)]
        status_msg.velocity = [1.0]  # Publishing rate indicator
        status_msg.effort = [0.0]
        
        self.status_pub.publish(status_msg)
        
        self.publish_counter += 1


def main(args=None):
    rclpy.init(args=args)
    
    advanced_publisher = AdvancedPublisher()
    
    try:
        rclpy.spin(advanced_publisher)
    except KeyboardInterrupt:
        advanced_publisher.get_logger().info('Shutting down advanced publisher')
    finally:
        advanced_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Diagrams

```
[ROS 2 Action Communication Pattern]

[Action Client Node]                    [Action Server Node]
        |                                       |
        |-------- Goal Request ----------------->|
        |                                       |
        |<------- Goal Acknowledged -------------|
        |                                       |
        |<------- Feedback Messages --------------| (multiple during execution)
        |                                       |
        |<------- Result Response ---------------|
        |                                       |
        |                                  [Robot Hardware/Controller]
        |                                       |
        |<------- Joint States, Odometry -------| (for feedback)

Action State Transitions:
PENDING -> ACTIVE -> SUCCEEDED/ABORTED/CANCELED
                |
                |-> PREEMPTING -> PREEMPTED

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![ROS 2 Actions](/img/ros2-actions.png)`.

## Hands-on Exercises

1. **Exercise 1**: Create a custom action message for humanoid robot manipulation that includes goal parameters for grasping position, orientation, and force limits. Implement both an action server and client for this manipulation action.

2. **Exercise 2**: Implement a priority-based action system where higher-priority goals can preempt lower-priority ones. Use the cancel_callback to implement this functionality for your humanoid robot's walking action.

3. **Exercise 3**: Create a multi-stage human movement where different actions are chained together. For example, implement a "go to location and pick up object" behavior that first uses the walking action followed by a manipulation action.

4. **Exercise 4**: Experiment with different QoS policies for various types of robot data. Test the impact of reliability vs. best-effort and different durability settings on communication timing and reliability.

## Quiz

1. What are the three parts of an action definition in ROS 2?
   - a) Request, Response, Status
   - b) Goal, Result, Feedback
   - c) Start, Continue, End
   - d) Command, Data, Control

2. Which state is NOT part of the ROS 2 action state machine?
   - a) PENDING
   - b) ACTIVE
   - c) COMPLETED
   - d) ABORTED

3. Actions are most appropriate for:
   - a) Single quick calculations
   - b) Long-running tasks with feedback
   - c) System configuration
   - d) Sensor data streaming

4. True/False: Action feedback is sent continuously during goal execution.
   - Answer: _____

5. In which situation would you prefer using an action over a service?
   - a) Requesting robot's current battery level
   - b) Requesting a robot to navigate to a specific location
   - c) Setting robot's LED color
   - d) All of the above

## Summary

In this lesson, we explored ROS 2 actions, which are essential for implementing long-running, complex behaviors in humanoid robots. Actions provide a way to handle tasks that take time to execute while providing ongoing feedback and allowing for preemption when necessary.

We implemented a walking action server and client that demonstrates how humanoid robots can receive navigation goals and report progress during execution. We also examined advanced communication patterns using Quality of Service settings to ensure critical data receives appropriate treatment.

Actions, along with nodes, topics, and services, complete the core communication patterns in ROS 2 that are essential for developing sophisticated humanoid robotics applications.

## Key Terms

- **Action**: Goal-oriented communication pattern with feedback and result
- **Goal**: The objective to be achieved by an action server
- **Feedback**: Progress updates sent during action execution
- **Result**: Final outcome of an action execution
- **Action Server**: Node that accepts and executes action goals
- **Action Client**: Node that sends action goals and receives feedback/results
- **QoS (Quality of Service)**: Settings for configuring message handling characteristics
- **Reliability Policy**: QoS setting for message delivery guarantees
- **Durability Policy**: QoS setting for message persistence
- **Preemption**: Replacing a running goal with a higher priority one