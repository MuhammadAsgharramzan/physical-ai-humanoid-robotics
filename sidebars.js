// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: Introduction to Physical AI and Humanoid Robotics',
      items: [
        'module-1/lesson-1-fundamentals-of-physical-ai',
        'module-1/lesson-2-digital-intelligence-vs-embodied-agents',
        'module-1/lesson-3-current-state-of-humanoid-robotics',
        'module-1/lesson-4-key-challenges-and-opportunities'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Foundations of Embodied Intelligence',
      items: [
        'module-2/lesson-1-sensorimotor-integration',
        'module-2/lesson-2-perception-action-loops',
        'module-2/lesson-3-examples-of-humanoid-robots',
        'module-2/lesson-4-role-of-embodiment-in-learning'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: AI Techniques for Robotics',
      items: [
        'module-3/lesson-1-computer-vision-for-robotics',
        'module-3/lesson-2-machine-learning-for-robot-control',
        'module-3/lesson-3-path-planning-and-navigation',
        'module-3/lesson-4-integration-of-ai-techniques'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Human-Robot Interaction',
      items: [
        'module-4/lesson-1-intuitive-interfaces',
        'module-4/lesson-2-natural-language-processing',
        'module-4/lesson-3-social-robotics-principles',
        'module-4/lesson-4-ethical-considerations'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 5: Advanced Control Systems',
      items: [
        'module-5/lesson-1-advanced-control-systems',
        'module-5/lesson-2-dynamic-interaction-patterns',
        'module-5/lesson-3-integration-of-advanced-concepts'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 6: Real-World Deployment and Validation',
      items: [
        'module-6/lesson-1-real-world-deployment-considerations',
        'module-6/lesson-2-validation-and-testing-methods',
        'module-6/module-6-lesson-3'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/glossary'
      ],
      collapsed: true,
    }
  ],
};

module.exports = sidebars;