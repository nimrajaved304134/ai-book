import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './LessonLayout.module.css';

function LessonSidebar({ currentModule, currentLesson, lessons }) {
  return (
    <div className={styles.lessonSidebar}>
      <h3 className={styles.sidebarTitle}>Module {currentModule}</h3>
      
      <ul className={styles.lessonList}>
        {lessons.map((lesson, index) => (
          <li key={index} className={styles.lessonItem}>
            <a 
              href={lesson.path} 
              className={clsx(styles.lessonLink, {
                [styles.currentLesson]: lesson.path === currentLesson
              })}
            >
              {lesson.number}. {lesson.title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function LessonNavigation({ currentLessonIndex, lessons }) {
  const prevLesson = currentLessonIndex > 0 ? lessons[currentLessonIndex - 1] : null;
  const nextLesson = currentLessonIndex < lessons.length - 1 ? lessons[currentLessonIndex + 1] : null;
  
  return (
    <div className={styles.lessonNavigation}>
      <div className={styles.navButton}>
        {prevLesson ? (
          <a href={prevLesson.path} className={styles.prevButton}>
            ← Previous: {prevLesson.title}
          </a>
        ) : (
          <span className={styles.disabledButton}>← Previous</span>
        )}
      </div>
      
      <div className={styles.navButton}>
        {nextLesson ? (
          <a href={nextLesson.path} className={styles.nextButton}>
            Next: {nextLesson.title} →
          </a>
        ) : (
          <span className={styles.disabledButton}>Next →</span>
        )}
      </div>
    </div>
  );
}

export default function LessonLayout({ children, moduleNumber, lessonNumber, lessonTitle, currentPath }) {
  const { siteConfig } = useDocusaurusContext();
  
  // Define lessons structure based on module
  const getLessonsForModule = (module) => {
    switch(module) {
      case 1:
        return [
          { number: 1, title: "Introduction to ROS 2", path: "/docs/module-1/lesson-1" },
          { number: 2, title: "Nodes, Topics, and Services", path: "/docs/module-1/lesson-2" },
          { number: 3, title: "Actions and Advanced Communication", path: "/docs/module-1/lesson-3" },
          { number: 4, title: "Packages, Launch Files, and Testing", path: "/docs/module-1/lesson-4" }
        ];
      case 2:
        return [
          { number: 1, title: "Gazebo Simulation Environment", path: "/docs/module-2/lesson-1" },
          { number: 2, title: "Unity Simulation Environment", path: "/docs/module-2/lesson-2" },
          { number: 3, title: "Simulation Integration and Testing", path: "/docs/module-2/lesson-3" }
        ];
      case 3:
        return [
          { number: 1, title: "Introduction to NVIDIA Isaac Platform", path: "/docs/module-3/lesson-1" },
          { number: 2, title: "Isaac Sim for Humanoid Robotics", path: "/docs/module-3/lesson-2" },
          { number: 3, title: "Isaac AI Models and Deployment", path: "/docs/module-3/lesson-3" }
        ];
      case 4:
        return [
          { number: 1, title: "Introduction to VLA Models", path: "/docs/module-4/lesson-1" },
          { number: 2, title: "Advanced VLA Architectures", path: "/docs/module-4/lesson-2" },
          { number: 3, title: "VLA Applications in Humanoid Robotics", path: "/docs/module-4/lesson-3" }
        ];
      default:
        return [];
    }
  };
  
  const lessons = getLessonsForModule(moduleNumber);
  const currentLessonIndex = lessons.findIndex(lesson => lesson.path === currentPath);
  
  return (
    <Layout
      title={`${lessonTitle} - ${siteConfig.title}`}
      description={lessonTitle}>
      <div className={styles.lessonPage}>
        <LessonSidebar 
          currentModule={moduleNumber} 
          currentLesson={currentPath} 
          lessons={lessons} 
        />
        
        <main className={styles.lessonContent}>
          <header className={styles.lessonHeader}>
            <h1 className={styles.lessonTitle}>
              Module {moduleNumber}, Lesson {lessonNumber}: {lessonTitle}
            </h1>
            <div className={styles.lessonMeta}>
              <span className={styles.lessonBadge}>Module {moduleNumber}</span>
              <span className={styles.lessonBadge}>Lesson {lessonNumber}</span>
            </div>
          </header>
          
          <div className={styles.lessonBody}>
            {children}
          </div>
          
          <LessonNavigation 
            currentLessonIndex={currentLessonIndex} 
            lessons={lessons} 
          />
        </main>
      </div>
    </Layout>
  );
}