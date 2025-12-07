# Physical AI & Humanoid Robotics Textbook

This repository contains a comprehensive textbook on Physical AI and Humanoid Robotics, built with Docusaurus.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

The site is configured for deployment to GitHub Pages. To deploy:

1. Update the `organizationName` and `projectName` in `docusaurus.config.js`
2. Run the deploy command:
   ```bash
   GIT_USER=<Your GitHub username> \
   CURRENT_BRANCH=main \
   USE_SSH=true \
   npm run deploy
   ```

Alternatively, if you're using GitHub Actions, push your changes to the main branch and the site will be automatically deployed via the workflow defined in `.github/workflows/deploy.yml`.

## File Structure

The textbook content is organized in the `book/chapters/` directory:
- `module-1/` - ROS 2
- `module-2/` - Gazebo & Unity Simulation
- `module-3/` - NVIDIA Isaac
- `module-4/` - Vision-Language-Action

Each module contains multiple lessons in markdown format.

## Custom Components

The site includes custom Docusaurus components for an enhanced learning experience:
- Header
- Sidebar
- ChapterCard
- LessonLayout

## Technology Stack

- [Docusaurus](https://docusaurus.io/): Static site generator
- [React](https://reactjs.org/): UI library
- [Tailwind CSS](https://tailwindcss.com/): Styling framework
- [GitHub Pages](https://pages.github.com/): Hosting