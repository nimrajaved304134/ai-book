import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './Sidebar.module.css';

function SidebarItem({ to, label, isActive }) {
  return (
    <li className={clsx(styles.sidebarItem, { [styles.active]: isActive })}>
      <Link to={to} className={styles.sidebarLink}>
        {label}
      </Link>
    </li>
  );
}

function SidebarSection({ title, items, isActive }) {
  return (
    <div className={clsx(styles.sidebarSection, { [styles.active]: isActive })}>
      <h3 className={styles.sectionTitle}>{title}</h3>
      <ul className={styles.sidebarList}>
        {items.map((item, index) => (
          <SidebarItem 
            key={index} 
            to={item.to} 
            label={item.label} 
            isActive={item.active} 
          />
        ))}
      </ul>
    </div>
  );
}

export default function Sidebar() {
  const sidebarData = [
    {
      title: "Module 1: ROS 2 Fundamentals",
      items: [
        { to: "/docs/module-1/lesson-1", label: "Introduction to ROS 2", active: false },
        { to: "/docs/module-1/lesson-2", label: "Nodes, Topics, and Services", active: false },
        { to: "/docs/module-1/lesson-3", label: "Actions and Advanced Communication", active: false },
        { to: "/docs/module-1/lesson-4", label: "Packages, Launch Files, and Testing", active: false }
      ]
    },
    {
      title: "Module 2: Simulation with Gazebo & Unity",
      items: [
        { to: "/docs/module-2/lesson-1", label: "Gazebo Simulation Environment", active: false },
        { to: "/docs/module-2/lesson-2", label: "Unity Simulation Environment", active: false },
        { to: "/docs/module-2/lesson-3", label: "Simulation Integration and Testing", active: false }
      ]
    },
    {
      title: "Module 3: NVIDIA Isaac Platform",
      items: [
        { to: "/docs/module-3/lesson-1", label: "Introduction to NVIDIA Isaac Platform", active: false },
        { to: "/docs/module-3/lesson-2", label: "Isaac Sim for Humanoid Robotics", active: false },
        { to: "/docs/module-3/lesson-3", label: "Isaac AI Models and Deployment", active: false }
      ]
    },
    {
      title: "Module 4: Vision-Language-Action Models",
      items: [
        { to: "/docs/module-4/lesson-1", label: "Introduction to VLA Models", active: false },
        { to: "/docs/module-4/lesson-2", label: "Advanced VLA Architectures", active: false },
        { to: "/docs/module-4/lesson-3", label: "VLA Applications in Humanoid Robotics", active: false }
      ]
    }
  ];

  return (
    <div className={styles.sidebar}>
      <nav className={styles.nav}>
        {sidebarData.map((section, index) => (
          <SidebarSection 
            key={index} 
            title={section.title} 
            items={section.items} 
            isActive={section.items.some(item => item.active)} 
          />
        ))}
      </nav>
    </div>
  );
}