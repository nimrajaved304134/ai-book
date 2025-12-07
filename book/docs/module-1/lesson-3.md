---
sidebar_position: 3
---

# Lesson 3: ROS 2 Packages and Build System

## Introduction
This lesson covers the structure and creation of ROS 2 packages, the fundamental building blocks of ROS 2 applications. We'll explore the package structure, build system (colcon), and best practices for organizing your ROS 2 code.

## Concepts
ROS 2 packages are the basic units of code organization and distribution. They contain:

1. **Source Code**: C++ or Python code implementing nodes, libraries, etc.
2. **Package Manifest**: package.xml file defining metadata and dependencies
3. **CMakeLists.txt**: Build configuration for C++ packages
4. **setup.py**: Build configuration for Python packages
5. **Launch Files**: XML or Python files to start multiple nodes
6. **Configuration Files**: YAML files for parameters
7. **Resource Files**: URDF, meshes, textures, etc.

The build system in ROS 2 is based on colcon, which is a command-line build tool that supports various build systems (CMake, ament_cmake, etc.).

## Technical Deep Dive
The package structure in ROS 2 follows a standard layout:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for CMake packages
├── package.xml             # Package manifest
├── src/                    # C++ source files
├── include/my_robot_package/ # C++ header files
├── scripts/                # Standalone scripts (Python, shell, etc.)
├── launch/                 # Launch files
├── config/                 # Configuration files
├── test/                   # Unit and integration tests
├── msg/                    # Custom message definitions
├── srv/                    # Custom service definitions
├── action/                 # Custom action definitions
└── README.md               # Package documentation
```

The colcon build system:
- colcon build: Compiles all packages in the workspace
- colcon test: Runs tests for packages
- colcon list: Lists packages in the workspace
- colcon graph: Shows dependencies between packages

## Diagrams
```
Workspace (my_robot_ws/)
├── src/
    ├── package_1/
    ├── package_2/
    └── package_3/
```

Build Process:
```
Source Code + Dependencies --> colcon build --> Compiled Artifacts
```

## Code Examples (Python/ROS 2)
Example package.xml file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_tutorial</name>
  <version>0.0.0</version>
  <description>Tutorials for my robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Example CMakeLists.txt for a C++ package:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Create executable
add_executable(my_robot_node src/my_robot_node.cpp)
ament_target_dependencies(my_robot_node rclcpp std_msgs)

# Install executables
install(TARGETS
  my_robot_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Exercises
1. Create a new ROS 2 package named "my_robot_description" with proper structure including launch files, URDF, and configuration files.

2. Implement a launch file that starts three different nodes with different parameters, showing how launch files can manage complex systems.

3. Create a custom message type for robot sensor readings and implement publisher and subscriber nodes that use this message type.

## Quiz
1. What is the build system used in ROS 2?
   a) catkin
   b) colcon
   c) cmake
   d) make

2. Which file defines the metadata and dependencies of a ROS 2 package?
   a) CMakeLists.txt
   b) setup.py
   c) package.xml
   d) manifest.json

3. True or False: All ROS 2 packages must contain both C++ and Python code.
   a) True
   b) False

## Summary
This lesson covered the structure of ROS 2 packages, the build system (colcon), and the standard directory layout. Understanding packages is essential for organizing and distributing ROS 2 code effectively. We explored both Python and C++ package structures and their respective build configurations.

## Key Terms
- **Packages**: Fundamental building blocks of ROS 2 applications
- **colcon**: Command-line build tool for ROS 2
- **package.xml**: Manifest file defining package metadata and dependencies
- **CMakeLists.txt**: Build configuration for C++ packages
- **setup.py**: Build configuration for Python packages
- **Launch Files**: Files to start multiple nodes at once
- **Workspace**: Directory containing multiple related packages
- **ament**: ROS 2 build system and package format