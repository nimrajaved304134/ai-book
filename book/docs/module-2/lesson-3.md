---
title: "Simulation Integration and Testing"
sidebar_label: "Lesson 3: Simulation Integration and Testing"
---

# Lesson 3: Simulation Integration and Testing

## Introduction

In this final lesson of Module 2, we'll explore how to effectively integrate Gazebo and Unity simulation environments with real robotic systems, and how to use these simulations for comprehensive testing of humanoid robots. Simulation-to-reality transfer is a critical capability that allows developers to validate algorithms in a safe, cost-effective environment before deploying them on expensive physical robots.

The integration of simulation with real systems is particularly important in humanoid robotics, where the cost of failure can be high due to expensive hardware and potential safety risks. Effective simulation integration allows for extensive testing of complex behaviors like walking, manipulation, and human-robot interaction before risking the physical system. We'll also cover how to structure simulation environments to maximize their effectiveness for different types of testing.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Simulation-to-Reality Transfer

Key challenges in transferring from simulation to reality include:

- **Reality Gap**: Differences between simulated and real sensor data
- **Model Fidelity**: How accurately simulation represents real physics
- **Control Robustness**: Controllers that work in simulation but fail in reality
- **Environmental Differences**: Simulated environments rarely capture all real-world complexities

### Hardware-in-the-Loop Testing

This approach integrates real hardware components with simulation:

- **Real Sensors**: Using actual sensors in simulated environments
- **Real Actuators**: Testing control algorithms on real motors
- **Mixed Reality**: Combining virtual objects with real environments
- **Safety First**: Protecting hardware while testing aggressive behaviors

### Simulation Testing Methodologies

For humanoid robots, simulations enable testing that would be dangerous or impossible in the real world:

- **Stress Testing**: Pushing robots to their physical limits safely
- **Edge Case Testing**: Unusual scenarios that are hard to recreate physically
- **Long-term Tests**: Extended operation that would be costly with real robots
- **Multi-Robot Scenarios**: Coordinating multiple robots without hardware costs

## Detailed Technical Explanations

### Gazebo-ROS Integration Patterns

1. **Direct Integration**: Running real controllers on simulated robots
2. **Sensor Fusion**: Combining simulated and real sensors
3. **Hardware Simulation**: Simulating missing hardware components
4. **Network Integration**: Running simulation and control on different machines

### Unity-ROS Integration Architecture

1. **Message Bridge**: Converting Unity data to ROS message formats
2. **Coordinate Systems**: Managing different coordinate systems between engines
3. **Timing Synchronization**: Aligning simulation time with real-time systems
4. **Performance Optimization**: Ensuring real-time performance during simulation

### Testing Strategies

- **Unit Testing**: Component-specific testing in isolation
- **Integration Testing**: Testing interaction between components
- **System Testing**: Validating complete robot behavior
- **Acceptance Testing**: Verifying the system meets requirements

## Code Examples

### Gazebo-ROS Integration Test Script

```python
#!/usr/bin/env python3
# test_simulation_integration.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from humanoid_msgs.action import WalkToGoal
from std_msgs.msg import Float64MultiArray

import time
import math


class SimulationIntegrationTest(Node):
    def __init__(self):
        super().__init__('simulation_integration_test')
        
        # Create QoS profile
        qos_profile = QoSProfile(depth=10)
        
        # Publishers for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', qos_profile)
        
        # Subscribers for robot feedback
        self.joint_state_sub = self.create_subscription(
            JointState, 
            'joint_states', 
            self.joint_state_callback, 
            qos_profile
        )
        self.imu_sub = self.create_subscription(
            Imu, 
            'imu', 
            self.imu_callback, 
            qos_profile
        )
        
        # Action client for walking
        self.walk_action_client = ActionClient(
            self,
            WalkToGoal,
            'walk_to_goal'
        )
        
        # Internal state
        self.current_joint_state = None
        self.current_imu = None
        self.test_results = {"joint_control": False, "balance": False, "walking": False}
        
        # Timer to run tests
        self.timer = self.create_timer(0.1, self.run_tests)
        self.test_counter = 0
        
        self.get_logger().info('Simulation integration test node initialized')
    
    def joint_state_callback(self, msg):
        self.current_joint_state = msg
    
    def imu_callback(self, msg):
        self.current_imu = msg
    
    def run_tests(self):
        self.test_counter += 1
        
        # Test 1: Joint control (every 10 iterations)
        if self.test_counter % 10 == 0 and self.current_joint_state:
            self.test_joint_control()
        
        # Test 2: Balance control (every 20 iterations)
        if self.test_counter % 20 == 0 and self.current_imu:
            self.test_balance_control()
        
        # Test 3: Walking (every 50 iterations)
        if self.test_counter % 50 == 0:
            self.test_walking_action()
        
        # Log test status periodically
        if self.test_counter % 100 == 0:
            self.log_test_results()
    
    def test_joint_control(self):
        """Test sending joint position commands"""
        if not self.current_joint_state:
            return
            
        # Create a small adjustment to current positions
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.current_joint_state.name
        cmd_msg.position = [
            pos + 0.1 * math.sin(self.test_counter * 0.1) 
            for pos in self.current_joint_state.position
        ]
        cmd_msg.velocity = [0.0] * len(cmd_msg.position)
        cmd_msg.effort = [0.0] * len(cmd_msg.position)
        
        self.joint_cmd_pub.publish(cmd_msg)
        self.test_results["joint_control"] = True
        self.get_logger().info('Joint control test completed')
    
    def test_balance_control(self):
        """Test balance maintenance based on IMU data"""
        if not self.current_imu:
            return
            
        # Read IMU data to assess balance
        roll = math.atan2(
            2.0 * (self.current_imu.orientation.w * self.current_imu.orientation.x + 
                  self.current_imu.orientation.y * self.current_imu.orientation.z),
            1.0 - 2.0 * (self.current_imu.orientation.x**2 + self.current_imu.orientation.y**2)
        )
        
        pitch = math.asin(
            2.0 * (self.current_imu.orientation.w * self.current_imu.orientation.y - 
                  self.current_imu.orientation.z * self.current_imu.orientation.x)
        )
        
        # Check if robot is balanced (within tolerances)
        max_angle = 0.1  # 0.1 radians (~5.7 degrees)
        if abs(roll) < max_angle and abs(pitch) < max_angle:
            self.test_results["balance"] = True
            self.get_logger().info('Balance test PASSED')
        else:
            self.test_results["balance"] = False
            self.get_logger().info(f'Balance test FAILED - Roll: {roll:.3f}, Pitch: {pitch:.3f}')
    
    def test_walking_action(self):
        """Test walking action with real controller"""
        if not self.walk_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Walk action server not available')
            return
        
        # Create a goal to walk forward
        goal_msg = WalkToGoal.Goal()
        goal_msg.target_position.x = 1.0  # 1 meter forward
        goal_msg.target_position.y = 0.0
        goal_msg.target_position.z = 0.0
        goal_msg.tolerance = 0.1  # 10 cm tolerance
        
        # Send the goal
        self.walk_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.walking_feedback_callback
        )
        
        self.get_logger().info('Walking test goal sent')
    
    def walking_feedback_callback(self, feedback_msg):
        """Handle feedback from walking action"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Walking progress: {feedback.distance_to_goal:.2f}m to goal'
        )
    
    def log_test_results(self):
        """Log current test results"""
        self.get_logger().info('--- Simulation Integration Test Results ---')
        for test, result in self.test_results.items():
            status = 'PASS' if result else 'FAIL'
            self.get_logger().info(f'{test}: {status}')
        self.get_logger().info('---------------------------------------')


def main(args=None):
    rclpy.init(args=args)
    
    test_node = SimulationIntegrationTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info('Shutting down simulation integration test')
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Unity-ROS Bridge with Test Integration

```csharp
// UnityROSTestBridge.cs - Enhanced ROS bridge for integration testing
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Humanoid_msgs;

public class UnityROSTestBridge : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Robot References")]
    public HumanoidRobotController robotController;
    public Transform robotRoot;
    public List<Transform> jointTransforms;
    public List<string> jointNames;
    
    [Header("Test Parameters")]
    public bool enableAutomaticTesting = true;
    public float testInterval = 5.0f;
    
    private RosConnection m_Ros;
    private float testTimer = 0f;
    private Dictionary<string, bool> testResults = new Dictionary<string, bool>();
    private int testCounter = 0;

    void Start()
    {
        // Initialize ROS connection
        m_Ros = RosConnection.GetOrCreateInstance();
        m_Ros.Connect(rosIP, rosPort);
        
        // Subscribe to ROS topics
        m_Ros.Subscribe<geometry_msgs.TwistMsg>("cmd_vel", CmdVelCallback);
        m_Ros.Subscribe<actionlib_msgs.GoalIDMsg>("/walk_to_goal/_action/cancel", WalkCancelCallback);
        
        // Initialize test results
        InitializeTestResults();
        
        Debug.Log("Unity-ROS Test Bridge initialized");
    }

    void Update()
    {
        if (enableAutomaticTesting)
        {
            HandleAutomaticTesting();
        }
        
        // Publish robot state at regular intervals
        PublishRobotState();
    }

    private void InitializeTestResults()
    {
        testResults["joint_control"] = false;
        testResults["balance_control"] = false;
        testResults["walking"] = false;
        testResults["sensor_simulation"] = false;
    }

    private void HandleAutomaticTesting()
    {
        testTimer += Time.deltaTime;
        
        if (testTimer >= testInterval)
        {
            ExecuteNextTest();
            testTimer = 0f;
        }
    }

    private void ExecuteNextTest()
    {
        switch (testCounter % 4)  // Cycle through 4 different tests
        {
            case 0:
                TestJointControl();
                break;
            case 1:
                TestBalanceControl();
                break;
            case 2:
                TestWalking();
                break;
            case 3:
                TestSensorSimulation();
                break;
        }
        
        testCounter++;
    }

    private void TestJointControl()
    {
        // Send random joint commands to test control
        var jointStateMsg = new sensor_msgs_JointStateMsg();
        jointStateMsg.name = jointNames.ToArray();
        jointStateMsg.position = new double[jointNames.Count];
        jointStateMsg.velocity = new double[jointNames.Count];
        jointStateMsg.effort = new double[jointNames.Count];
        jointStateMsg.header = new std_msgs.HeaderMsg();
        jointStateMsg.header.frame_id = "base_link";

        // Set random positions for each joint
        for (int i = 0; i < jointNames.Count; i++)
        {
            jointStateMsg.position[i] = Random.Range(-1.0, 1.0); // Random positions between -1 and 1 rad
            jointStateMsg.velocity[i] = 0.0;
            jointStateMsg.effort[i] = 0.0;
        }

        jointStateMsg.header.stamp = new builtin_interfaces.TimeMsg();
        m_Ros.Publish("joint_commands", jointStateMsg);
        
        testResults["joint_control"] = true;
        Debug.Log($"Joint control test {testCounter / 4 + 1} executed");
    }

    private void TestBalanceControl()
    {
        // Verify the robot is maintaining balance
        Vector3 robotPosition = robotRoot.position;
        
        // Check if the robot is upright (not fallen over)
        bool isUpright = Mathf.Abs(robotPosition.y - 0.8f) < 0.5f; // Assume normal COM height is 0.8m
        
        testResults["balance_control"] = isUpright;
        
        if (isUpright)
        {
            Debug.Log("Balance test PASSED");
        }
        else
        {
            Debug.Log($"Balance test FAILED - Robot position: {robotPosition}");
        }
    }

    private void TestWalking()
    {
        // Send a walking goal through ROS
        // In a real implementation, this would use an action client
        // For this simulation, we'll directly call the robot controller
        
        Vector3 destination = robotRoot.position + new Vector3(
            Random.Range(-2.0f, 2.0f),  // Random X displacement
            0,                          // No Y displacement
            Random.Range(-2.0f, 2.0f)   // Random Z displacement
        );
        
        robotController.StartWalking(destination);
        
        testResults["walking"] = true;
        Debug.Log($"Walking test to {destination} initiated");
    }

    private void TestSensorSimulation()
    {
        // In a real implementation, this would verify sensor data
        // For this example, we'll just simulate creating some sensor data
        
        // Create a simulated IMU message
        var imuMsg = new sensor_msgs.ImuMsg();
        imuMsg.header.frame_id = "imu_link";
        imuMsg.header.stamp = new builtin_interfaces.TimeMsg();
        
        // Set orientation (simplified)
        imuMsg.orientation.x = robotRoot.rotation.x;
        imuMsg.orientation.y = robotRoot.rotation.y;
        imuMsg.orientation.z = robotRoot.rotation.z;
        imuMsg.orientation.w = robotRoot.rotation.w;
        
        // Set angular velocity (simplified)
        imuMsg.angular_velocity.x = Random.Range(-0.1f, 0.1f);
        imuMsg.angular_velocity.y = Random.Range(-0.1f, 0.1f);
        imuMsg.angular_velocity.z = Random.Range(-0.1f, 0.1f);
        
        // Set linear acceleration (simplified)
        imuMsg.linear_acceleration.x = Random.Range(-1.0f, 1.0f);
        imuMsg.linear_acceleration.y = Physics.gravity.y + Random.Range(-0.5f, 0.5f);
        imuMsg.linear_acceleration.z = Random.Range(-1.0f, 1.0f);
        
        m_Ros.Publish("imu", imuMsg);
        
        testResults["sensor_simulation"] = true;
        Debug.Log("Sensor simulation test executed");
    }

    private void PublishRobotState()
    {
        // Publish joint states
        var jointStateMsg = new sensor_msgs.JointStateMsg();
        jointStateMsg.name = jointNames.ToArray();
        jointStateMsg.position = new double[jointNames.Count];
        jointStateMsg.velocity = new double[jointNames.Count];
        jointStateMsg.effort = new double[jointNames.Count];
        jointStateMsg.header = new std_msgs.HeaderMsg();
        jointStateMsg.header.frame_id = "base_link";

        for (int i = 0; i < jointTransforms.Count && i < jointNames.Count; i++)
        {
            // Convert Unity rotation to radians (simplified)
            jointStateMsg.position[i] = jointTransforms[i].localEulerAngles.y * Mathf.Deg2Rad;
            jointStateMsg.velocity[i] = 0.0;  // Simplified velocity
            jointStateMsg.effort[i] = 0.0;    // Simplified effort
        }

        jointStateMsg.header.stamp = new builtin_interfaces.TimeMsg();
        m_Ros.Publish("joint_states", jointStateMsg);
        
        // Publish robot transform
        var tfMsg = new geometry_msgs.TransformStampedMsg();
        tfMsg.header.frame_id = "world";
        tfMsg.child_frame_id = "humanoid_base";
        tfMsg.header.stamp = new builtin_interfaces.TimeMsg();
        
        tfMsg.transform.translation.x = robotRoot.position.x;
        tfMsg.transform.translation.y = robotRoot.position.y;
        tfMsg.transform.translation.z = robotRoot.position.z;
        
        tfMsg.transform.rotation.x = robotRoot.rotation.x;
        tfMsg.transform.rotation.y = robotRoot.rotation.y;
        tfMsg.transform.rotation.z = robotRoot.rotation.z;
        tfMsg.transform.rotation.w = robotRoot.rotation.w;
        
        var tfArray = new geometry_msgs.TransformStampedMsg[] { tfMsg };
        var tfMsgPack = new tf2_msgs.TFMessageMsg(tfArray);
        m_Ros.Publish("/tf", tfMsgPack);
    }

    // ROS Message Callbacks
    private void CmdVelCallback(geometry_msgs.TwistMsg msg)
    {
        // Apply velocity command to robot
        robotController.SetTargetVelocity((float)msg.linear.x, (float)msg.angular.z);
    }

    private void WalkCancelCallback(actionlib_msgs.GoalIDMsg msg)
    {
        // Cancel any walking action
        robotController.StopWalking();
    }

    void OnApplicationQuit()
    {
        // Log test results on quit
        Debug.Log("=== Final Test Results ===");
        foreach (var result in testResults)
        {
            Debug.Log($"{result.Key}: {(result.Value ? "PASS" : "FAIL")}");
        }
    }
}
```

### Simulation Test Suite Configuration

```yaml
# config/simulation_test_suite.yaml
simulation_test_suite:
  # Test suite parameters
  enable_automatic_tests: true
  test_interval: 5.0  # seconds between tests
  test_scenarios:
    - name: "basic_functionality"
      description: "Test basic robot functions in simulation"
      tests:
        - joint_control_test
        - sensor_simulation_test
        - communication_test
    - name: "balance_and_locomotion"
      description: "Test robot balance and walking capabilities"
      tests:
        - balance_stability_test
        - walking_trajectory_test
        - obstacle_avoidance_test
    - name: "human_robot_interaction"
      description: "Test interaction with humans in simulation"
      tests:
        - gesture_recognition_test
        - voice_command_test
        - safe_interaction_test

  # Simulation parameters
  simulation_fidelity:
    physics_accuracy: 0.001  # seconds
    rendering_quality: high
    sensor_noise:
      camera: 0.01  # 1% noise
      imu: 0.005    # 0.5% noise
      joint_encoders: 0.001  # 0.1% noise

  # Test validation parameters
  validation_criteria:
    # Balance validation
    max_tilt_angle: 0.1  # radians
    min_contact_force: 5.0  # newtons
    
    # Walking validation
    max_step_error: 0.05  # meters
    min_step_success_rate: 0.95  # 95% success rate
    
    # Safety validation
    min_distance_to_human: 0.5  # meters
    max_robot_velocity: 1.0  # m/s

  # Performance metrics
  performance_thresholds:
    sim_real_time_factor: 1.0  # Should run at real-time speed
    cpu_usage_limit: 80.0  # percent
    memory_limit: 4096  # MB
```

## Diagrams

```
[Simulation Integration and Testing Architecture]

[Real Robot] <===> [Controller Hardware] <===> [Simulation Environment]
    |                    |                          |
    |                    |                          |
    |--------->[Safety]---|--------->[Validation]----|
    |                    |                          |
    |--------->[Logging]--|--------->[Testing]-------|
    |                    |                          |
    |--------->[Debug]----|--------->[Metrics]-------|
    |                                                 |
    |----------------------------------------->[Analysis]

Testing Workflow:
[Test Scenario] -> [Simulation Run] -> [Data Collection] -> [Validation] -> [Results]

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![Simulation Integration](/img/simulation-integration.png)`.

## Hands-on Exercises

1. **Exercise 1**: Set up a simulation-to-reality transfer pipeline where you train a humanoid robot controller in simulation and then test it on a real robot (or a more accurate simulation). Document the differences in performance and identify the reality gap.

2. **Exercise 2**: Implement a hardware-in-the-loop test where real sensors are used in a simulated environment. For example, use a real camera in a Unity environment and test how well perception algorithms work with mixed real/simulated data.

3. **Exercise 3**: Create a comprehensive test suite that runs automatically in simulation and validates different aspects of humanoid robot behavior: walking, balance, manipulation, and interaction.

4. **Exercise 4**: Design and execute a "stress test" in simulation that pushes the humanoid robot to its physical limits to identify failure modes and improve the control algorithms before real-world testing.

## Quiz

1. What is the main purpose of hardware-in-the-loop testing?
   - a) To replace simulation entirely
   - b) To integrate real hardware components with simulation for safer testing
   - c) To make simulation run faster
   - d) To reduce the cost of simulation

2. What does the term "reality gap" refer to in robotics?
   - a) The gap between the robot and objects
   - b) Differences between simulated and real-world performance
   - c) The time delay in robot responses
   - d) Physical gaps in robot structure

3. Which approach is best for testing robot behaviors that could be dangerous in the real world?
   - a) Only test with real robots
   - b) Only test in simulation
   - c) Use simulation to identify safe parameters, then test on real robots
   - d) Skip testing entirely

4. True/False: Simulation testing can completely replace real-world testing for humanoid robots.
   - Answer: _____

5. What is the purpose of domain randomization in simulation?
   - a) To make simulation run faster
   - b) To reduce computational requirements
   - c) To improve the transfer of learned behaviors from simulation to reality
   - d) To make graphics look better

## Summary

In this final lesson of Module 2, we've covered the critical topic of integrating simulation environments with real robotic systems and using these tools for comprehensive testing of humanoid robots. We explored both Gazebo and Unity as simulation platforms, their unique advantages, and how they can be used together to validate robot behaviors safely and efficiently.

We examined the challenges of simulation-to-reality transfer, including the reality gap and model fidelity issues. We also looked at practical implementations of simulation integration, including ROS bridges and test frameworks that can validate robot performance across multiple domains.

Simulation testing is an indispensable tool in humanoid robotics development, allowing for extensive validation of complex behaviors before risking expensive hardware or human safety. The combination of Gazebo's physics accuracy and Unity's visual fidelity provides a comprehensive simulation toolkit for modern humanoid robot development.

## Key Terms

- **Simulation-to-Reality Transfer**: Process of applying behaviors learned in simulation to real robots
- **Reality Gap**: Differences between simulated and real-world robot performance
- **Hardware-in-the-Loop**: Testing approach integrating real hardware with simulation
- **Domain Randomization**: Technique to vary simulation parameters to improve transfer learning
- **Stress Testing**: Testing to push systems to their operational limits
- **Validation Criteria**: Standards for determining if a test result is acceptable
- **Performance Metrics**: Quantitative measures of system performance
- **Test Suite**: Collection of tests for validating system functionality
- **Safety First**: Design principle prioritizing safety in testing
- **Edge Case**: Unusual scenario that might not be encountered in normal testing