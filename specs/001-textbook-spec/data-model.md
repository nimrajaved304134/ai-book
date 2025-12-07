# Data Model for Physical AI & Humanoid Robotics Textbook

## Chapter Entity
- **name**: String - The name of the chapter (e.g., "Introduction to Physical AI")
- **moduleNumber**: Integer - The module number (1-4)
- **description**: String - A brief description of the chapter
- **lessons**: Array of Lesson entities - The lessons that belong to this chapter
- **learningObjectives**: Array of String - The learning objectives for the chapter

## Lesson Entity
- **name**: String - The name of the lesson (e.g., "Understanding Robot Kinematics")
- **lessonNumber**: Integer - The lesson number within the chapter (1-13)
- **chapter**: Chapter entity - The chapter this lesson belongs to
- **introduction**: String - Introduction section of the lesson
- **concepts**: String - Concepts covered in the lesson
- **technicalDeepDive**: String - Technical details about the concepts
- **diagrams**: Array of String - File paths to diagrams used in the lesson
- **codeExamples**: Array of CodeExample entities - Code examples in the lesson
- **exercises**: Array of Exercise entities - Exercises in the lesson
- **quiz**: Array of QuizQuestion entities - Quiz questions for the lesson
- **summary**: String - Summary of the lesson
- **keyTerms**: Array of String - Key terms defined in the lesson
- **learningObjectives**: Array of String - Learning objectives specific to the lesson

## CodeExample Entity
- **language**: String - The programming language (e.g., "Python", "ROS 2")
- **code**: String - The actual code example
- **description**: String - Description of what the code example demonstrates
- **fileName**: String - Name of the file if stored separately

## Exercise Entity
- **question**: String - The exercise question
- **type**: String - Type of exercise (e.g., "multiple-choice", "short-answer", "programming")
- **difficulty**: String - Difficulty level (e.g., "beginner", "intermediate", "advanced")
- **solution**: String - The solution to the exercise
- **hints**: Array of String - Optional hints for the exercise

## QuizQuestion Entity
- **question**: String - The quiz question
- **options**: Array of String - Multiple choice options (if applicable)
- **correctAnswer**: String - The correct answer
- **explanation**: String - Explanation of the correct answer
- **difficulty**: String - Difficulty level (e.g., "beginner", "intermediate", "advanced")

## DocusaurusComponent Entity
- **name**: String - Name of the component (e.g., "Header", "Sidebar", "ChapterCard", "LessonLayout")
- **description**: String - Description of what the component does
- **props**: Array of Prop entities - Properties the component accepts
- **dependencies**: Array of String - Other components or libraries this component depends on

## Prop Entity
- **name**: String - Name of the property
- **type**: String - Type of the property (e.g., "string", "number", "boolean", "object")
- **isRequired**: Boolean - Whether the property is required
- **defaultValue**: String - Default value if not required
- **description**: String - Description of what the property does