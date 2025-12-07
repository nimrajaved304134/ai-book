---
sidebar_position: 2
---

# Lesson 2: ROS 2 Architecture and Concepts

## Introduction
This lesson delves deeper into the architecture of ROS 2, exploring its core concepts and how different components work together to create a distributed robotics framework. Understanding the architecture is crucial for developing efficient and robust robotic applications.

## Concepts
The architecture of ROS 2 is fundamentally different from ROS 1, primarily due to its use of DDS (Data Distribution Service) as the middleware. This enables several key architectural concepts:

1. **Distributed Discovery**: Nodes can discover each other without requiring a central master, enabling more robust and flexible systems.

2. **Client Libraries**: ROS 2 provides multiple client libraries (rclcpp for C++, rclpy for Python) that wrap DDS implementations.

3. **Lifecycle Nodes**: Specialized nodes that provide a state machine for managing the initialization, activation, and deactivation of functionality.

4. **Composition**: The ability to combine multiple nodes into a single process to optimize performance and resource usage.

5. **Parameters**: A unified system for configuring nodes at runtime with type safety and introspection.

## Technical Deep Dive
The ROS 2 architecture is built on several layers:

- **Application Layer**: User code that creates nodes, publishers, subscribers, services, etc.
- **Client Library Layer**: rclcpp, rclpy, and other language-specific libraries
- **Middleware Layer**: DDS implementations (FastDDS, CycloneDDS, etc.)
- **Transport Layer**: Network protocols and interfaces

The architecture also includes:
- CLI tools (ros2 command set) for introspection and management
- Launch system for starting multiple nodes with configuration
- Logging system with hierarchical levels
- Testing frameworks for validation

## Diagrams
```
[Application Code]
       |
[Client Libraries - rclcpp/rclpy]
       |
[ROS Middleware - DDS]
       |
[Transport Layer - TCP/UDP]
```

ROS 2 Nodes Communication:
```
Node A(Publisher) ---- Topic ----> Node B(Subscriber)
      |                              |
Service Client <---> Service <---> Service Server
      |                              |
Action Client  <---> Action  <---> Action Server
```

## Code Examples (Python/ROS 2)
Creating a Lifecycle Node in Python:

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.executors import SingleThreadedExecutor

class LifecycleNodeExample(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_node_example')
        self.get_logger().info('Lifecycle node created, current state: unconfigured')
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('on_configure is called')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('on_activate is called')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('on_deactivate is called')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('on_cleanup is called')
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleNodeExample()
    
    # Execute transitions
    node.trigger_configure()
    node.trigger_activate()
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.trigger_deactivate()
        node.trigger_cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises
1. Create a lifecycle node that publishes a counter value, but only starts publishing after activation and stops when deactivated.

2. Implement a parameter server node that accepts parameters for minimum and maximum values, and use these parameters to constrain a published value.

3. Design a system with two nodes that can be composed into a single process, showing how composition can improve performance.

## Quiz
1. What is the primary middleware used in ROS 2?
   a) TCP/IP
   b) DDS
   c) HTTP
   d) MQTT

2. Which of the following is NOT a state in the ROS 2 lifecycle?
   a) Unconfigured
   b) Inactive
   c) Active
   d) Paused

3. True or False: ROS 2 nodes require a central master to discover each other.
   a) True
   b) False

## Summary
This lesson covered the architecture of ROS 2, highlighting how it differs from ROS 1 with its distributed discovery system and DDS middleware. We explored key architectural concepts like lifecycle nodes, composition, and parameters that enable more robust and maintainable robotics applications.

## Key Terms
- **DDS (Data Distribution Service)**: Middleware providing publish-subscribe communication
- **Client Libraries**: Language-specific APIs (rclcpp, rclpy) for ROS 2
- **Lifecycle Nodes**: Nodes with managed state transitions
- **Composition**: Running multiple nodes in a single process
- **Parameters**: Runtime configuration system for nodes
- **Distributed Discovery**: Peer-to-peer node discovery mechanism