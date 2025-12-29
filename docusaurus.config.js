// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.vsLight;
const darkCodeTheme = require('prism-react-renderer').themes.vsDark;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Digital Intelligence to Embodied Agents',
  url: 'https://your-domain.vercel.app',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // GitHub pages deployment config.
  organizationName: 'MuhammadAsgharramzan', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/robot-icon.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            type: 'dropdown',
            label: 'Account',
            position: 'right',
            items: [
              {
                label: 'Login',
                to: '/login',
              },
              {
                label: 'Sign Up',
                to: '/signup',
              },
            ],
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Chapters',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
              {
                label: 'Module 1: Physical AI Fundamentals',
                to: '/docs/module-1/lesson-1-fundamentals-of-physical-ai',
              },
              {
                label: 'Module 2: Embodied Intelligence',
                to: '/docs/module-2/lesson-1-sensorimotor-integration',
              },
              {
                label: 'Module 3: AI Techniques for Robotics',
                to: '/docs/module-3/lesson-1-computer-vision-for-robotics',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'ROS2 Documentation',
                href: 'https://docs.ros.org/en/humble/',
              },
              {
                label: 'Gazebo Simulator',
                href: 'https://gazebosim.org/',
              },
              {
                label: 'Isaac Sim',
                href: 'https://developer.nvidia.com/isaac-sim',
              },
              {
                label: 'Research Papers',
                href: '#',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Discussions',
                href: 'https://github.com/MuhammadAsgharramzan/physical-ai-humanoid-robotics/discussions',
              },
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
            ],
          },
          {
            title: 'GitHub',
            items: [
              {
                label: 'Repository',
                href: 'https://github.com/MuhammadAsgharramzan/physical-ai-humanoid-robotics',
              },
              {
                label: 'Issues',
                href: 'https://github.com/MuhammadAsgharramzan/physical-ai-humanoid-robotics/issues',
              },
              {
                label: 'Contributing',
                href: 'https://github.com/MuhammadAsgharramzan/physical-ai-humanoid-robotics/blob/main/CONTRIBUTING.md',
              },
            ],
          },
          {
            title: 'Legal',
            items: [
              {
                label: 'Privacy Policy',
                href: '/privacy',
              },
              {
                label: 'Terms of Service',
                href: '/tos',
              },
              {
                label: 'MIT License',
                href: 'https://github.com/MuhammadAsgharramzan/physical-ai-humanoid-robotics/blob/main/LICENSE',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Project. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'bash', 'json', 'yaml'],
      },
    }),
};

module.exports = config;