# Quickstart Guide for Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js v18+ installed
- npm or yarn package manager
- Git for version control
- Text editor (VS Code recommended)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agenticai-book
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Initialize Docusaurus**
   ```bash
   npx create-docusaurus@latest website classic
   ```

4. **Install Tailwind CSS**
   ```bash
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

5. **Start the development server**
   ```bash
   npm start
   ```

## Customization

1. **Update configuration**
   - Edit `docusaurus.config.js` to customize site metadata, theme, and navigation

2. **Add textbook content**
   - Add chapters to the `/book/chapters/` directory
   - Each module gets its own subdirectory (module-1/, module-2/, etc.)
   - Add lessons as markdown files within each module directory

3. **Customize theme**
   - Components are in `/src/components/`
   - CSS is in `/src/css/`
   - Theme overrides are in `/src/theme/`

4. **Build for production**
   ```bash
   npm run build
   ```

## Deployment to GitHub Pages

1. **Set up GitHub Pages in repository settings**
   - Go to Settings > Pages
   - Select source as GitHub Actions

2. **Update docusaurus.config.js**
   - Set `organizationName`, `projectName`, and `deploymentBranch`

3. **Build and deploy**
   ```bash
   GIT_USER=<Your GitHub username> \
   CURRENT_BRANCH=main \
   USE_SSH=true \
   npm run deploy
   ```

## Content Creation Guidelines

1. **Lesson Structure**:
   Each lesson should follow this pedagogical structure:
   - Introduction
   - Concepts
   - Technical Deep Dive
   - Diagrams
   - Code Examples (Python/ROS 2)
   - Exercises
   - Quiz
   - Summary
   - Key Terms

2. **Adding Diagrams**:
   - Place images in `/static/images/`
   - Reference in markdown as `![Alt text](/img/image-name.png)`

3. **Code Examples**:
   - Use appropriate syntax highlighting
   - Example for Python: ```python
   - Example for ROS 2: ```ros