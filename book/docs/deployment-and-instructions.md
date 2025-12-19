# Running, Building, and Deploying the Physical AI & Humanoid Robotics Textbook

## Prerequisites

Before you begin, ensure you have the following installed:

- Node.js (version 18 or higher)
- npm or yarn package manager
- Git for version control

## Setting Up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/agenticai-book/agenticai-book.git
   cd agenticai-book
   ```

2. **Navigate to the book directory**
   ```bash
   cd book
   ```

3. **Install dependencies**
   ```bash
   npm install
   # or if using yarn
   yarn install
   ```

## Running the Development Server

To start a local development server with hot reloading:

```bash
npm start
# or if using yarn
yarn start
```

This will start the development server at `http://localhost:3000`. The site will automatically reload when you make changes to the content or components.

## Building the Static Site

To generate the static site for production:

```bash
npm run build
# or if using yarn
yarn run build
```

This will create a `build` directory with the complete static site that can be served by any static hosting service.

## Deployment Options

### GitHub Pages Deployment

1. **Configure your `docusaurus.config.js`**:
   Ensure the `organizationName`, `projectName`, and `baseUrl` are correctly set in your configuration.

2. **Deploy using the command**:
   ```bash
   GIT_USER=<Your GitHub username> \
   CURRENT_BRANCH=main \
   USE_SSH=true \
   npm run deploy
   ```

### Netlify Deployment

1. **Connect your GitHub repository to Netlify**:
   - Go to [Netlify](https://www.netlify.com/)
   - Click "New site from Git"
   - Select your GitHub repository
   - For build settings, use:
     - Build command: `cd book && npm run build`
     - Publish directory: `book/build`

### Vercel Deployment

1. **Deploy directly from your GitHub repository**:
   - Go to [Vercel](https://vercel.com/)
   - Import your Git repository
   - For deployment settings:
     - Build command: `cd book && npm run build`
     - Output directory: `book/build`

### Custom Static Hosting

The `build` directory contains a complete static site that can be hosted on any static file server:

1. **Build the site**: `npm run build`
2. **Upload the contents of the `build` directory** to your static hosting provider
3. **Configure your domain** to point to the hosting service

## Customization

### Adding New Content

To add new lessons or modify existing content:

1. Create/edit `.md` or `.mdx` files in the `book/docs` directory
2. Update the `book/sidebars.js` file to include new content in the navigation

### Custom Components

The custom Docusaurus components are located in `book/src/components/`. You can modify or extend these components as needed for your specific requirements.

### Styling

- Custom CSS can be added to `book/src/css/custom.css`
- Tailwind CSS classes can be used throughout the components if configured properly
- Component-specific styles should be added as CSS modules

## Troubleshooting

### Common Issues

1. **Dependency conflicts or errors**:
   - Clear the npm cache: `npm cache clean --force`
   - Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

2. **Port already in use**:
   - The development server uses port 3000 by default
   - To use a different port: `npm start -- --port 8080`

3. **Build fails due to memory issues**:
   - Increase Node.js memory limit: `export NODE_OPTIONS="--max_old_space_size=4096"`
   - Then run the build command again

4. **Images or assets not loading**:
   - Ensure assets are placed in the `book/static` directory
   - Reference them with absolute paths: `/img/filename.jpg`

## Performance Optimization

### For Better Build Performance:

1. **Use npm v7 or later** for faster dependency resolution
2. **Enable Docusaurus cache**:
   ```bash
   npm run build -- --cache
   ```
3. **Enable bundle analysis**:
   ```bash
   npm run build -- --bundle-analyzer
   ```

### For Better Runtime Performance:

1. **Enable Google Analytics** (if configured) to monitor performance metrics
2. **Optimize images** before adding to the site
3. **Minimize use of heavy client-side components** for faster loading

## Updating Content

### For Regular Updates:

1. Make changes to the documentation files in `book/docs/`
2. Test changes locally by running `npm start`
3. Commit and push changes to Git
4. The site will automatically update based on your deployment setup

### For Major Updates:

1. Consider versioning your documentation
2. Test the build process to ensure all links work correctly
3. Check for any broken links with: `npm run build && npm run serve`
4. Verify all interactive components function as expected

## Development Workflow

### For Content Writers:

1. Fork the repository
2. Create a new branch: `git checkout -b new-content`
3. Add or modify content in the `book/docs/` directory
4. Run `npm start` to preview changes
5. Create a pull request to the main repository

### For Developers:

1. Follow the same steps as content writers
2. Make code changes to React components in `book/src/components/`
3. Ensure compatibility with Docusaurus v3.x
4. Test both development and production builds
5. Follow React best practices for component development

## Additional Features

### Search

The documentation includes a search functionality powered by Algolia. To customize or disable search, modify the `themeConfig.algolia` section in `docusaurus.config.js`.

### Internationalization

To add multiple languages:

1. Add locales to the `i18n` section in `docusaurus.config.js`
2. Create translation files in the `i18n` directory
3. Use the `translate` API for text that needs translation

### Analytics

The site can be configured with analytics tools like Google Analytics. Add your tracking ID in the `docusaurus.config.js` file under the `gtag` section.

---

For more detailed documentation about Docusaurus features and customization options, visit the [official Docusaurus documentation](https://docusaurus.io/docs).

For questions about the Physical AI & Humanoid Robotics textbook content, please open an issue in the GitHub repository.