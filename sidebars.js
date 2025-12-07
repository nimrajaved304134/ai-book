// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Chapter 1: ROS 2',
      items: [
        {
          type: 'category',
          label: 'Lessons',
          items: [
            'module-1/lesson-1',
            'module-1/lesson-2',
            'module-1/lesson-3',
          ],
          link: {
            type: 'generated-index',
            title: 'Chapter 1 Lessons',
            description: 'Learn about ROS 2 fundamentals',
            slug: '/module-1',
          },
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 2: Gazebo & Unity Simulation',
      items: [
        {
          type: 'category',
          label: 'Lessons',
          items: [
            'module-2/lesson-1',
            'module-2/lesson-2',
          ],
          link: {
            type: 'generated-index',
            title: 'Chapter 2 Lessons',
            description: 'Learn about simulation environments',
            slug: '/module-2',
          },
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 3: NVIDIA Isaac',
      items: [
        {
          type: 'category',
          label: 'Lessons',
          items: [
            'module-3/lesson-1',
          ],
          link: {
            type: 'generated-index',
            title: 'Chapter 3 Lessons',
            description: 'Learn about NVIDIA Isaac robotics platform',
            slug: '/module-3',
          },
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 4: Vision-Language-Action',
      items: [
        {
          type: 'category',
          label: 'Lessons',
          items: [
            'module-4/lesson-1',
          ],
          link: {
            type: 'generated-index',
            title: 'Chapter 4 Lessons',
            description: 'Learn about vision-language-action models in robotics',
            slug: '/module-4',
          },
        },
      ],
    },
  ],
};

module.exports = sidebars;