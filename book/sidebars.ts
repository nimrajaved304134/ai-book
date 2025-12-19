import path from 'path';

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      items: [
        'module-1/lesson-1',
        'module-1/lesson-2',
        'module-1/lesson-3',
        'module-1/lesson-4'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Simulation with Gazebo & Unity',
      items: [
        'module-2/lesson-1',
        'module-2/lesson-2',
        'module-2/lesson-3'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac Platform',
      items: [
        'module-3/lesson-1',
        'module-3/lesson-2',
        'module-3/lesson-3'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Models',
      items: [
        'module-4/lesson-1',
        'module-4/lesson-2',
        'module-4/lesson-3'
      ],
    },
  ],
};

export default sidebars;