---
title: "ROS 2 Packages, Launch Files, and Testing"
sidebar_label: "Lesson 4: ROS 2 Packages, Launch Files, and Testing"
---

# Lesson 4: ROS 2 Packages, Launch Files, and Testing

## Introduction

In this final lesson of Module 1, we'll cover essential topics for developing professional-level ROS 2 applications: packages for organizing code, launch files for managing complex systems, and testing strategies for ensuring reliable robot behavior. For humanoid robotics applications, these tools are critical for managing the complexity of systems that involve dozens of nodes working in concert.

A humanoid robot system typically consists of multiple packages handling different aspects of robot functionality: perception, planning, control, and hardware interfaces. Launch files help orchestrate these packages at startup, ensuring nodes start in the correct order with appropriate parameters. Finally, comprehensive testing ensures that the robot behaves reliably in various scenarios, which is especially important for safety when dealing with physical robots.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Package Organization in Humanoid Robotics

In humanoid robotics, a well-organized package structure typically includes:

- **Hardware Interface Packages**: `humanoid_hardware_interface`, `sensor_drivers`
- **Control Packages**: `humanoid_controller`, `walking_controller`, `balance_controller`
- **Perception Packages**: `vision_perception`, `audio_processing`, `object_detection`
- **Navigation Packages**: `path_planning`, `humanoid_navigation`, `obstacle_avoidance`
- **Behavior Packages**: `motion_primitives`, `task_planning`, `human_robot_interaction`

### Launch File Benefits for Complex Systems

Humanoid robots require coordination of many nodes simultaneously. Launch files enable:

- **Startup Management**: Ensuring nodes start in the correct order
- **Parameter Configuration**: Setting system-wide parameters efficiently
- **Conditional Launching**: Starting different nodes based on robot configuration
- **Process Monitoring**: Automatic restart of failed nodes

### Testing Strategies for Physical Systems

Testing humanoid robots requires special considerations:

- **Unit Testing**: For individual algorithms and functions
- **Integration Testing**: For interaction between system components
- **Simulation Testing**: Using physics engines to test robot behavior without risk
- **Safety Testing**: Verification that robot won't fall or harm humans
- **Performance Testing**: Ensuring real-time constraints are met

## Detailed Technical Explanations

### Package Structure and Conventions

A ROS 2 package typically follows this structure:

```
package_name/
├── CMakeLists.txt or pyproject.toml
├── package.xml
├── README.md
├── LICENSE
├── src/ (for C++) or package_name/ (for Python)
├── launch/
├── config/
├── test/
├── include/ (for C++)
├── scripts/
└── docs/
```

### Launch File Components

Launch files can include:

- **Nodes**: ROS nodes to run
- **Parameters**: System-wide configuration values
- **Remappings**: Changing topic/service names
- **Conditions**: Conditional launching of components
- **Actions**: Complex launch sequences

### Testing Frameworks

ROS 2 includes several testing frameworks:

- **gtest**: For C++ unit tests
- **pytest**: For Python unit tests
- **rostest**: For integration tests
- **gmock**: For C++ mocking

## Code Examples

### Package Structure Example

Here's an example of a well-structured ROS 2 package for humanoid robot walking control:

```
humanoid_walking_controller/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── walking_params.yaml
├── launch/
│   └── walking_controller.launch.py
├── src/
│   ├── walking_controller.cpp
│   └── gait_generator.cpp
├── include/
│   └── humanoid_walking_controller/
│       ├── walking_controller.hpp
│       └── gait_generator.hpp
├── test/
│   ├── test_walking_controller.cpp
│   └── test_gait_generator.cpp
└── scripts/
    └── visualize_gait.py
```

### Package.xml for a Humanoid Controller

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_walking_controller</name>
  <version>1.0.0</version>
  <description>Controller for humanoid robot walking</description>
  <maintainer email="robotics@example.com">Robotics Team</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>control_msgs</depend>
  <depend>realtime_tools</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Launch File Example for Humanoid Robot System

```python
# launch/humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_namespace = LaunchConfiguration('robot_namespace', default='humanoid')
    
    # Get the launch directory
    pkg_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')
    
    declare_robot_namespace = DeclareLaunchArgument(
        'robot_namespace',
        default_value='humanoid',
        description='Robot namespace for topics and TF')

    # Static transform broadcaster for robot base
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'humanoid_base'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Walking controller node
    walking_controller = Node(
        package='humanoid_walking_controller',
        executable='walking_controller_node',
        name='walking_controller',
        namespace=robot_namespace,
        parameters=[
            os.path.join(pkg_path, 'config', 'walking_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Balance controller node
    balance_controller = Node(
        package='humanoid_balance_controller',
        executable='balance_controller_node',
        name='balance_controller',
        namespace=robot_namespace,
        parameters=[
            os.path.join(pkg_path, 'config', 'balance_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Perception node for obstacle detection
    perception_node = Node(
        package='humanoid_perception',
        executable='obstacle_detector_node',
        name='obstacle_detector',
        namespace=robot_namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_robot_namespace)

    # Add nodes
    ld.add_action(static_transform_publisher)
    ld.add_action(walking_controller)
    ld.add_action(balance_controller)
    ld.add_action(perception_node)

    return ld
```

### YAML Configuration File

```yaml
# config/walking_params.yaml
humanoid:
  walking_controller:
    # Walking gait parameters
    step_height: 0.05  # meters
    step_length: 0.3   # meters
    step_duration: 1.0 # seconds
    max_forward_speed: 0.5  # m/s
    max_turn_speed: 0.5     # rad/s
    
    # Balance parameters
    balance_kp: 50.0
    balance_kd: 10.0
    com_height: 0.8  # Center of mass height in meters
    
    # Joint control parameters
    position_gain: 10.0
    velocity_gain: 1.0
    effort_limit: 100.0  # N*m
    
    # Safety limits
    max_tilt_angle: 0.2  # radians
    min_contact_force: 10.0  # N
```

### Unit Test Example

```python
# test/test_walking_controller.cpp
#include <gtest/gtest.h>
#include <memory>
#include "humanoid_walking_controller/walking_controller.hpp"

class WalkingControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        controller = std::make_shared<WalkingController>();
    }

    void TearDown() override {
        controller.reset();
    }

    std::shared_ptr<WalkingController> controller;
};

TEST_F(WalkingControllerTest, InitializeController) {
    EXPECT_NE(controller, nullptr);
    EXPECT_TRUE(controller->is_initialized());
}

TEST_F(WalkingControllerTest, CalculateStep) {
    geometry_msgs::msg::Point start_position = {0.0, 0.0, 0.0};
    geometry_msgs::msg::Point target_position = {1.0, 0.0, 0.0};
    
    auto step = controller->calculate_step(start_position, target_position);
    
    // Check that the step moves in the right direction
    EXPECT_GT(step.x, 0.0);
    EXPECT_NEAR(step.y, 0.0, 0.01);
}

TEST_F(WalkingControllerTest, ValidateGaitParameters) {
    auto params = controller->get_gait_parameters();
    
    // Check that parameters are within safe ranges
    EXPECT_GT(params.step_height, 0.0);
    EXPECT_LT(params.step_height, 0.3);  // Less than 30cm
    EXPECT_GT(params.step_duration, 0.1); // More than 0.1s for safety
}
```

### Python Test Example

```python
# test/test_gait_generator.py
import unittest
import numpy as np
from humanoid_walking_controller.gait_generator import GaitGenerator


class TestGaitGenerator(unittest.TestCase):
    def setUp(self):
        self.gait_gen = GaitGenerator(step_height=0.05, step_duration=1.0)
    
    def test_init(self):
        self.assertEqual(self.gait_gen.step_height, 0.05)
        self.assertEqual(self.gait_gen.step_duration, 1.0)
    
    def test_generate_foot_trajectory(self):
        # Test that foot trajectory starts and ends at ground level
        times = np.linspace(0, 1.0, 100)
        positions = self.gait_gen.generate_foot_trajectory(times)
        
        # Check that first and last position are at ground level (z=0)
        self.assertAlmostEqual(positions[0, 2], 0.0, places=2)
        self.assertAlmostEqual(positions[-1, 2], 0.0, places=2)
        
        # Check that highest point is approximately at step_height
        max_height = np.max(positions[:, 2])
        self.assertGreater(max_height, 0.04)  # Should be close to 0.05m
        self.assertLess(max_height, 0.06)    # Should not exceed 0.06m
    
    def test_stability_check(self):
        # Test that gait parameters result in stable walking
        is_stable = self.gait_gen.check_stability()
        self.assertTrue(is_stable)


if __name__ == '__main__':
    unittest.main()
```

## Diagrams

```
[ROS 2 Package and Launch Organization]

[Humanoid Robot System Architecture]

┌─────────────────────────────────────────────────────────────────┐
│                    Main Launch File                              │
│  (humanoid_robot.launch.py)                                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Walking         │  │ Balance         │  │ Perception      │  │
│  │ Controller      │  │ Controller      │  │ Node            │  │
│  │ (C++)           │  │ (C++)           │  │ (Python)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Gait Generator  │  │ ZMP Stabilizer  │  │ Obstacle        │  │
│  │ (C++)           │  │ (C++)           │  │ Detector        │  │
│  └─────────────────┘  └─────────────────┘  │ (Python)        │  │
│                                            └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                Configuration Files                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Walking         │  │ Balance         │  │ General         │  │
│  │ Params          │  │ Params          │  │ Params          │  │
│  │ (YAML)          │  │ (YAML)          │  │ (YAML)          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![ROS 2 Package Structure](/img/ros2-package-structure.png)`.

## Hands-on Exercises

1. **Exercise 1**: Create a new ROS 2 package for a humanoid robot's balance controller. Include the proper directory structure, package.xml, and CMakeLists.txt. Implement a basic balance controller node that subscribes to IMU data and joint states, and publishes corrective joint commands.

2. **Exercise 2**: Write a launch file that starts your humanoid robot's perception, planning, and control nodes with appropriate parameters. Add conditional logic to use different configurations based on whether you're running in simulation or on the real robot.

3. **Exercise 3**: Create unit tests for your humanoid robot's gait generation algorithm. Test edge cases like turning in place, walking backwards, and handling obstacles in the robot's path.

4. **Exercise 4**: Use the `ros2doctor` tool to analyze your running system. Verify that all nodes are communicating properly and that parameters are set correctly. Fix any issues discovered.

## Quiz

1. What is the purpose of a launch file in ROS 2?
   - a) To compile source code
   - b) To start multiple nodes with configuration
   - c) To debug nodes
   - d) To visualize robot models

2. Which command is used to build a ROS 2 workspace?
   - a) `ros2 build`
   - b) `catkin_make`
   - c) `colcon build`
   - d) `make build`

3. In which file would you typically define system-wide parameters for a humanoid robot?
   - a) package.xml
   - b) CMakeLists.txt
   - c) launch file
   - d) YAML configuration file

4. True/False: You can only run one node per package in ROS 2.
   - Answer: _____

5. Which testing framework would you use for C++ unit tests in ROS 2?
   - a) pytest
   - b) unittest
   - c) gtest
   - d) rostest

## Summary

In this final lesson of Module 1, we've covered essential tools for building professional ROS 2 applications: packages for organizing code, launch files for managing complex systems, and testing strategies for ensuring reliability. These tools are critical for humanoid robotics, where systems must manage the complexity of dozens of nodes working together to control a physical robot safely.

Proper package organization makes it easier to maintain and extend humanoid robot systems. Launch files ensure that all required nodes start with correct parameters and dependencies. Comprehensive testing, including unit, integration, and system tests, ensures that the robot behaves reliably in various scenarios.

With these foundational ROS 2 concepts mastered, you're now ready to explore how these systems integrate with simulation environments, which we'll cover in Module 2.

## Key Terms

- **Package**: Modular unit containing nodes, libraries, and resources in ROS
- **Launch File**: Configuration file to start multiple nodes with parameters
- **CMakeLists.txt**: Build configuration file for C++ packages
- **package.xml**: Metadata and dependency configuration for packages
- **YAML**: Configuration file format used in ROS 2
- **Unit Test**: Test of individual functions or classes
- **Integration Test**: Test of multiple components working together
- **Parameter Server**: System for managing configuration values in ROS
- **colcon**: Build system for ROS 2 workspaces
- **ament**: ROS 2 build system and testing framework