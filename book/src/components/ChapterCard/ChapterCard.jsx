import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './ChapterCard.module.css';

function ChapterCard({ title, description, lessons, moduleNumber, color = 'blue' }) {
  const colorClasses = {
    blue: styles.blue,
    green: styles.green,
    purple: styles.purple,
    orange: styles.orange
  };

  const selectedColor = colorClasses[color] || colorClasses.blue;

  return (
    <div className={clsx(styles.chapterCard, selectedColor)}>
      <div className={styles.cardHeader}>
        <h2 className={styles.cardTitle}>
          <span className={styles.moduleNumber}>Module {moduleNumber}: </span>
          {title}
        </h2>
      </div>
      
      <div className={styles.cardBody}>
        <p className={styles.cardDescription}>{description}</p>
        
        <div className={styles.lessonsContainer}>
          <h3 className={styles.lessonsTitle}>Lessons:</h3>
          <ul className={styles.lessonsList}>
            {lessons.map((lesson, index) => (
              <li key={index} className={styles.lessonItem}>
                <Link to={lesson.path} className={styles.lessonLink}>
                  {lesson.number}. {lesson.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>
      
      <div className={styles.cardFooter}>
        <Link to={lessons[0]?.path} className={styles.startButton}>
          Start Learning
        </Link>
      </div>
    </div>
  );
}

export default function ChapterCards() {
  const chapters = [
    {
      title: "ROS 2 Fundamentals",
      description: "Learn the foundational concepts of Robot Operating System 2, the framework for developing robotic applications.",
      moduleNumber: 1,
      color: 'blue',
      lessons: [
        { number: 1, title: "Introduction to ROS 2", path: "/docs/module-1/lesson-1" },
        { number: 2, title: "Nodes, Topics, and Services", path: "/docs/module-1/lesson-2" },
        { number: 3, title: "Actions and Advanced Communication", path: "/docs/module-1/lesson-3" },
        { number: 4, title: "Packages, Launch Files, and Testing", path: "/docs/module-1/lesson-4" }
      ]
    },
    {
      title: "Simulation with Gazebo & Unity",
      description: "Explore simulation environments essential for developing and testing humanoid robots safely and efficiently.",
      moduleNumber: 2,
      color: 'green',
      lessons: [
        { number: 1, title: "Gazebo Simulation Environment", path: "/docs/module-2/lesson-1" },
        { number: 2, title: "Unity Simulation Environment", path: "/docs/module-2/lesson-2" },
        { number: 3, title: "Simulation Integration and Testing", path: "/docs/module-2/lesson-3" }
      ]
    },
    {
      title: "NVIDIA Isaac Platform",
      description: "Discover the comprehensive solution for developing, simulating, and deploying AI-powered robots.",
      moduleNumber: 3,
      color: 'purple',
      lessons: [
        { number: 1, title: "Introduction to NVIDIA Isaac Platform", path: "/docs/module-3/lesson-1" },
        { number: 2, title: "Isaac Sim for Humanoid Robotics", path: "/docs/module-3/lesson-2" },
        { number: 3, title: "Isaac AI Models and Deployment", path: "/docs/module-3/lesson-3" }
      ]
    },
    {
      title: "Vision-Language-Action Models",
      description: "Learn about cutting-edge AI that integrates visual perception, natural language, and action generation.",
      moduleNumber: 4,
      color: 'orange',
      lessons: [
        { number: 1, title: "Introduction to VLA Models", path: "/docs/module-4/lesson-1" },
        { number: 2, title: "Advanced VLA Architectures", path: "/docs/module-4/lesson-2" },
        { number: 3, title: "VLA Applications in Humanoid Robotics", path: "/docs/module-4/lesson-3" }
      ]
    }
  ];

  return (
    <div className={styles.chapterCardsContainer}>
      {chapters.map((chapter, index) => (
        <ChapterCard
          key={index}
          title={chapter.title}
          description={chapter.description}
          lessons={chapter.lessons}
          moduleNumber={chapter.moduleNumber}
          color={chapter.color}
        />
      ))}
    </div>
  );
}