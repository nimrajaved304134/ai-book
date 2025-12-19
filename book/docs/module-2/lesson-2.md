---
title: "Unity Simulation Environment"
sidebar_label: "Lesson 2: Unity Simulation Environment"
---

# Lesson 2: Unity Simulation Environment

## Introduction

In this lesson, we'll explore Unity as a simulation environment for humanoid robotics. While Gazebo is the traditional choice for ROS-based robotics simulation, Unity has emerged as a powerful alternative, especially for applications requiring high-quality graphics, human-robot interaction studies, and immersive virtual environments. Unity's real-time rendering capabilities, extensive asset library, and robust physics engine make it ideal for simulating complex humanoid scenarios and testing perception algorithms.

Unity's strength lies in its ability to create photorealistic environments that can help bridge the reality gap often encountered when deploying perception-based robots. For humanoid robots, this is particularly important when developing computer vision systems, as the high-fidelity rendering can produce sensor data that closely resembles that from real-world cameras. Additionally, Unity's intuitive interface enables researchers and developers to quickly create complex scenarios for testing robot behaviors.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Unity Robotics Framework

Unity provides several tools specifically for robotics simulation:

- **Unity Robotics Hub**: Centralized access to robotics tools and resources
- **Unity ML-Agents**: For training robotic agents using reinforcement learning
- **Unity Perception Package**: For generating synthetic training data for computer vision
- **ROS#**: A bridge between Unity and ROS/ROS2 systems

### High-Fidelity Simulation

Unity excels at creating realistic environments:
- **Physically-Based Rendering (PBR)**: Materials that accurately simulate real-world surfaces
- **Dynamic Lighting**: Global illumination and real-time lighting effects
- **Environmental Effects**: Weather, day/night cycles, and particle systems
- **High-Quality Physics**: Accurate collision detection and rigid body dynamics

### Human-Robot Interaction

Unity is particularly well-suited for HRI research:
- **3D Character Animation**: Detailed humanoid robot models with realistic movement
- **VR/AR Integration**: Testing robots in virtual and augmented reality environments
- **Multi-Modal Interfaces**: Combined visual, audio, and haptic feedback
- **Behavior Modeling**: Complex AI-driven character behaviors

### Synthetic Data Generation

Unity can generate training data for AI models:
- **Variety of Environments**: Diverse scenarios for robust AI training
- **Labeling Tools**: Automatic generation of semantic segmentation and bounding boxes
- **Sensor Simulation**: Accurate simulation of cameras, LIDAR, and other sensors
- **Domain Randomization**: Randomizing visual properties to improve real-world performance

## Detailed Technical Explanations

### Unity-Ros Bridge (ROS#)

The Unity-Ros bridge enables communication between Unity and ROS systems:
- **Message Passing**: Direct integration of ROS messages within Unity scripts
- **TF Transformations**: Proper handling of coordinate transformations
- **Service Calls**: Ability to call ROS services from Unity
- **Action Management**: Handling ROS actions within the Unity engine

### Unity Perception Package

This package enables generation of synthetic training data:
- **Camera Capture**: Multiple virtual cameras with different properties
- **Annotation Tools**: Automatic generation of labels for training data
- **Domain Randomization**: Randomization of textures, lighting, and environmental properties
- **Sensor Simulation**: Accurate simulation of depth sensors, IMUs, and other devices

### Physics Simulation in Unity

Unity's physics engine includes:
- **NVIDIA PhysX**: Industrial-grade physics simulation
- **Collision Detection**: Accurate detection of complex shapes
- **Joint Systems**: Constraints and articulation bodies for robot joint simulation
- **Cloth Simulation**: For simulating flexible materials

## Code Examples

### Unity ROS Bridge Setup

```csharp
// RosConnection.cs - Setup Unity to communicate with ROS
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class RosConnection : MonoBehaviour
{
    RosConnection m_Ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    public string topicName = "unity_data";
    public string jointStateTopic = "joint_states";
    public string cmdVelTopic = "cmd_vel";

    // Robot joint names and references
    public List<string> jointNames = new List<string>();
    public List<Transform> jointTransforms = new List<Transform>();

    // Robot movement parameters
    public float linearVelocity = 0f;
    public float angularVelocity = 0f;

    void Start()
    {
        // Get the ROS connection object
        m_Ros = RosConnection.GetOrCreateInstance();
        m_Ros.Connect(rosIP, rosPort);

        // Start a coroutine to continuously publish messages
        StartCoroutine(PublishJointStates());
    }

    IEnumerator PublishJointStates()
    {
        // Create a joint state message
        var jointStateMsg = new sensor_msgs_JointStateMsg();
        jointStateMsg.name = jointNames.ToArray();
        jointStateMsg.position = new double[jointNames.Count];
        jointStateMsg.velocity = new double[jointNames.Count];
        jointStateMsg.effort = new double[jointNames.Count];
        jointStateMsg.header = new std_msgs.HeaderMsg();
        
        while (true)
        {
            // Update joint positions from transforms
            for (int i = 0; i < jointTransforms.Count; i++)
            {
                if (i < jointNames.Count)
                {
                    // Convert Unity rotation (in degrees) to radians
                    jointStateMsg.position[i] = jointTransforms[i].localEulerAngles.y * Mathf.Deg2Rad;
                    jointStateMsg.velocity[i] = 0.0; // Simplified
                    jointStateMsg.effort[i] = 0.0;   // Simplified
                }
            }
            
            // Update header timestamp
            jointStateMsg.header.stamp = new builtin_interfaces.TimeMsg();
            jointStateMsg.header.frame_id = "base_link";

            // Publish the joint state message
            m_Ros.Publish(jointStateTopic, jointStateMsg);

            yield return new WaitForSeconds(0.1f); // Publish at 10Hz
        }
    }

    public void SendVelocityCommand(float linear, float angular)
    {
        var cmdVelMsg = new geometry_msgs.TwistMsg();
        cmdVelMsg.linear = new geometry_msgs.Vector3Msg(linear, 0, 0);
        cmdVelMsg.angular = new geometry_msgs.Vector3Msg(0, 0, angular);

        m_Ros.Publish(cmdVelTopic, cmdVelMsg);
    }
}
```

### Humanoid Robot Controller in Unity

```csharp
// HumanoidRobotController.cs - Script to control humanoid robot in Unity
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HumanoidRobotController : MonoBehaviour
{
    [Header("Joint References")]
    public Transform leftHip;
    public Transform leftKnee;
    public Transform leftAnkle;
    public Transform rightHip;
    public Transform rightKnee;
    public Transform rightAnkle;
    public Transform leftShoulder;
    public Transform leftElbow;
    public Transform rightShoulder;
    public Transform rightElbow;

    [Header("Walking Parameters")]
    public float stepHeight = 0.1f;
    public float stepLength = 0.3f;
    public float stepDuration = 1.0f;
    public float walkSpeed = 0.5f;
    
    [Header("Balance Parameters")]
    public Transform centerOfMass;
    public float balanceKp = 50.0f;
    public float balanceKd = 10.0f;
    public Transform targetComHeight;

    // Walking state
    private bool isWalking = false;
    private float walkTimer = 0f;
    private Vector3 targetPosition = Vector3.zero;
    private Vector3 initialPosition = Vector3.zero;
    
    // Balance control
    private Vector3 previousCoMVelocity = Vector3.zero;
    private Vector3 targetCoMPosition = Vector3.zero;

    void Start()
    {
        // Set initial target COM position
        targetCoMPosition = centerOfMass.position;
    }

    void Update()
    {
        if (isWalking)
        {
            WalkCycle();
        }
        
        MaintainBalance();
    }

    public void StartWalking(Vector3 destination)
    {
        isWalking = true;
        targetPosition = destination;
        initialPosition = transform.position;
        walkTimer = 0f;
    }

    private void WalkCycle()
    {
        walkTimer += Time.deltaTime;
        float progress = walkTimer / stepDuration;
        
        if (progress >= 1.0f)
        {
            // Reached destination, stop walking
            isWalking = false;
            return;
        }

        // Calculate step trajectory
        Vector3 direction = (targetPosition - initialPosition).normalized;
        float distance = Vector3.Distance(targetPosition, initialPosition);
        Vector3 newPosition = initialPosition + direction * distance * progress;
        
        // Add vertical oscillation for walking motion
        float verticalOffset = Mathf.Sin(progress * Mathf.PI) * stepHeight;
        newPosition.y += verticalOffset;
        
        // Apply new position
        transform.position = newPosition;

        // Animate legs for walking
        AnimateLegs(progress);
    }

    private void AnimateLegs(float progress)
    {
        // Simple walking animation
        float legPhase = progress * 2 * Mathf.PI;
        
        // Left leg moves during first half of step, right leg during second half
        float leftLegAngle = 0, rightLegAngle = 0;
        
        if (progress < 0.5f)
        {
            // Left leg forward swing
            leftLegAngle = Mathf.Sin(legPhase * 2) * 30f;  // Swing forward
            rightLegAngle = -Mathf.Sin(legPhase * 2) * 15f; // Swing back
        }
        else
        {
            // Right leg forward swing
            leftLegAngle = -Mathf.Sin(legPhase * 2) * 15f; // Swing back
            rightLegAngle = Mathf.Sin(legPhase * 2) * 30f;  // Swing forward
        }

        // Apply leg movements
        leftHip.localEulerAngles = new Vector3(0, leftLegAngle, 0);
        rightHip.localEulerAngles = new Vector3(0, -rightLegAngle, 0);
        
        // Knee movements coordinated with hip
        leftKnee.localEulerAngles = new Vector3(0, -leftLegAngle * 0.5f, 0);
        rightKnee.localEulerAngles = new Vector3(0, rightLegAngle * 0.5f, 0);
    }

    private void MaintainBalance()
    {
        // Calculate current center of mass
        Vector3 currentCoM = centerOfMass.position;
        
        // Calculate velocity for PD controller
        Vector3 currentCoMVelocity = (currentCoM - previousCoMVelocity) / Time.deltaTime;
        previousCoMVelocity = currentCoM;
        
        // Calculate error from target position
        Vector3 error = targetCoMPosition - currentCoM;
        
        // Apply PD control to maintain balance
        Vector3 correctiveForce = (error * balanceKp) + (currentCoMVelocity * balanceKd);
        
        // Apply corrective forces to joints (simplified implementation)
        // In a real implementation, this would involve inverse kinematics
        ApplyBalanceCorrection(correctiveForce);
    }

    private void ApplyBalanceCorrection(Vector3 correctiveForce)
    {
        // Simplified balance correction - in reality this would use inverse kinematics
        // to calculate appropriate joint angles
        
        // For now, just adjust ankle angles based on lateral forces
        if (Mathf.Abs(correctiveForce.x) > 0.1f)
        {
            float ankleAngle = correctiveForce.x * 5f; // Arbitrary scaling
            leftAnkle.localEulerAngles = new Vector3(0, ankleAngle, 0);
            rightAnkle.localEulerAngles = new Vector3(0, -ankleAngle, 0);
        }
    }

    // Public method to receive commands from ROS
    public void SetTargetVelocity(float linear, float angular)
    {
        if (linear != 0 || angular != 0)
        {
            // Calculate target position based on current position and velocity
            Vector3 forward = transform.forward * linear * Time.deltaTime;
            transform.Translate(forward, Space.World);
            
            // Apply angular rotation
            transform.Rotate(Vector3.up, angular * Time.deltaTime * Mathf.Rad2Deg, Space.World);
            
            isWalking = true;
        }
        else
        {
            isWalking = false;
        }
    }
}
```

### Perception Data Generation Script

```csharp
// PerceptionDataGenerator.cs - Script to generate synthetic training data
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;

public class PerceptionDataGenerator : MonoBehaviour
{
    public Camera mainCamera;
    public List<Renderer> sceneObjects;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public string outputDirectory = "PerceptionData";

    [Header("Domain Randomization")]
    public bool enableDomainRandomization = true;
    public float minLightIntensity = 0.5f;
    public float maxLightIntensity = 2.0f;
    public Color[] randomColors = { Color.red, Color.blue, Color.green, Color.yellow, Color.cyan, Color.magenta };

    private RenderTexture renderTexture;
    private int frameCount = 0;

    void Start()
    {
        // Create render texture for camera capture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        mainCamera.targetTexture = renderTexture;

        // Start data generation coroutine
        StartCoroutine(GeneratePerceptionData());
    }

    IEnumerator GeneratePerceptionData()
    {
        while (true)
        {
            // Apply domain randomization if enabled
            if (enableDomainRandomization)
            {
                ApplyDomainRandomization();
            }

            // Wait for the next frame to ensure rendering is complete
            yield return new WaitForEndOfFrame();

            // Capture RGB image
            Texture2D rgbImage = CaptureImage();

            // Generate synthetic annotations
            Texture2D segmentationImage = GenerateSegmentationMap();
            
            // Save images
            SaveImage(rgbImage, $"rgb_frame_{frameCount:0000}.png");
            SaveImage(segmentationImage, $"seg_frame_{frameCount:0000}.png");

            // Publish sensor data to ROS if connected
            PublishSensorData(rgbImage);

            frameCount++;

            // Wait before next capture (adjust for desired frame rate)
            yield return new WaitForSeconds(1.0f / 5.0f); // 5 FPS
        }
    }

    private Texture2D CaptureImage()
    {
        // Create a temporary RenderTexture to read from
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Create texture to hold the image
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        image.Apply();

        // Restore active RenderTexture
        RenderTexture.active = currentRT;
        
        return image;
    }

    private Texture2D GenerateSegmentationMap()
    {
        // Create segmentation map (simplified version)
        Texture2D segmentation = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        // For now, just generate a solid color - in practice, this would be a detailed semantic segmentation
        // where each pixel represents the class of object at that location
        for (int y = 0; y < imageHeight; y++)
        {
            for (int x = 0; x < imageWidth; x++)
            {
                // Simple algorithm to assign colors based on object IDs
                // This is highly simplified - real implementation would use object rendering IDs
                segmentation.SetPixel(x, y, GetSegmentationColorForPixel(x, y));
            }
        }
        
        segmentation.Apply();
        return segmentation;
    }

    private Color GetSegmentationColorForPixel(int x, int y)
    {
        // Simplified segmentation mapping - in reality, this would use
        // a more sophisticated approach to identify which object is at each pixel
        
        // For demonstration, we'll assign colors based on screen coordinates
        // and some random elements to simulate objects in the scene
        float rand = Mathf.PerlinNoise(x * 0.01f, y * 0.01f);
        if (rand > 0.7f)
            return Color.red;   // Class 1
        else if (rand > 0.4f)
            return Color.blue;  // Class 2
        else
            return Color.black; // Background
    }

    private void SaveImage(Texture2D image, string filename)
    {
        string fullPath = Path.Combine(outputDirectory, filename);
        
        // Ensure directory exists
        Directory.CreateDirectory(outputDirectory);
        
        byte[] bytes = image.EncodeToPNG();
        File.WriteAllBytes(fullPath, bytes);
        
        Debug.Log($"Saved image: {fullPath}");
    }

    private void PublishSensorData(Texture2D image)
    {
        // Check if ROS connection exists
        var ros = RosConnection.GetOrCreateInstance();
        if (ros == null)
            return;

        // Create and populate sensor message (simplified)
        // In a real implementation, you'd convert the Texture2D to the appropriate ROS message format
        var imageMsg = new sensor_msgs.ImageMsg();
        imageMsg.height = (uint)image.height;
        imageMsg.width = (uint)image.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(image.width * 3); // 3 bytes per pixel for RGB

        // Convert texture data to byte array (simplified)
        // Note: In practice, you'd need to properly encode the image data
        // according to the sensor_msgs/Image specification
        byte[] imageData = image.EncodeToPNG();

        ros.Publish("perception_data", imageMsg);
    }

    private void ApplyDomainRandomization()
    {
        // Randomize lighting
        foreach (Light light in FindObjectsOfType<Light>())
        {
            light.intensity = Random.Range(minLightIntensity, maxLightIntensity);
        }

        // Randomize colors of objects
        foreach (Renderer renderer in sceneObjects)
        {
            Material mat = renderer.material;
            int colorIdx = Random.Range(0, randomColors.Length);
            mat.color = randomColors[colorIdx];
        }
    }
}
```

## Diagrams

```
[Unity Simulation Architecture for Humanoid Robots]

[Unity Scene] <===> [ROS Bridge (ROS#)] <===> [ROS System]
    |                    |                      |
    |                    |                      |
    |--> [Robot Model]   |--> [Message]         |--> [Controllers]
    |    [Animation]     |    [Transforms]      |    [Sensors]
    |    [Physics]       |    [Services]        |    [Navigation]
    |                    |                      |
    |--> [Environment]   |--> [Perception]      |--> [AI/ML Training]
         [Lighting]          [Synthetic Data]        [Simulation]
         [Materials]         [Domain Rand]           [Testing]

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![Unity Architecture](/img/unity-architecture.png)`.

## Hands-on Exercises

1. **Exercise 1**: Install Unity and the Unity Robotics packages. Set up a simple scene with a humanoid model and connect it to a ROS network using the ROS# bridge.

2. **Exercise 2**: Create a more sophisticated humanoid model in Unity with realistic joint constraints and implement a basic walking animation using inverse kinematics.

3. **Exercise 3**: Use the Unity Perception package to generate synthetic training data for a computer vision task. Create a dataset with domain randomization to train a model that can work in real-world scenarios.

4. **Exercise 4**: Design a human-robot interaction scenario in Unity where the humanoid robot responds to human gestures or speech commands simulated in the environment.

## Quiz

1. What is the primary advantage of using Unity over Gazebo for robotics simulation?
   - a) Better physics simulation
   - b) Higher quality graphics and visualization
   - c) Lower computational requirements
   - d) Easier to install

2. What is domain randomization used for in Unity robotics?
   - a) Randomizing network connections
   - b) Generating diverse training data to improve real-world performance
   - c) Randomizing robot control algorithms
   - d) Randomizing physics parameters

3. Which Unity package is specifically designed for generating synthetic training data?
   - a) ML-Agents
   - b) Unity Robotics Hub
   - c) Unity Perception
   - d) ROS#

4. True/False: Unity can generate realistic sensor data for training computer vision algorithms.
   - Answer: _____

5. What does ROS# enable in Unity?
   - a) Running Unity on robots
   - b) Communication between Unity and ROS systems
   - c) Creating robots in Unity
   - d) Generating code for robots

## Summary

In this lesson, we explored Unity as a powerful simulation environment for humanoid robotics. Unity offers high-fidelity graphics, robust physics simulation, and tools for generating synthetic training data that can bridge the reality gap in robot perception systems.

We looked at the Unity Robotics framework, including the ROS# bridge that enables communication between Unity and ROS systems. We also examined the Unity Perception package and its role in generating synthetic data for AI training, which is particularly valuable for humanoid robots that rely heavily on computer vision.

Unity's strength lies in creating visually rich environments for testing human-robot interaction scenarios and generating diverse training datasets. While Gazebo excels at physics accuracy, Unity excels at visual fidelity and user experience, making it an excellent complement to traditional robotics simulators.

## Key Terms

- **Unity Robotics Hub**: Centralized access to robotics tools for Unity
- **ROS#**: Bridge connecting Unity and ROS/ROS2 systems
- **Unity Perception Package**: Tools for generating synthetic training data
- **Domain Randomization**: Technique for varying visual properties to improve model generalization
- **ML-Agents**: Unity package for training AI using reinforcement learning
- **Physically-Based Rendering (PBR)**: Materials that accurately simulate real-world appearance
- **Perception Data Generation**: Creating synthetic sensor data for AI training
- **Inverse Kinematics (IK)**: Calculating joint angles to achieve desired end effector positions
- **Synthetic Training Data**: Artificially generated data for training AI models
- **Reality Gap**: Difference between simulation and real-world performance