import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import { useLocation } from '@docusaurus/router';
import { usePluralForm } from '@docusaurus/theme-common';
import { useDocsSidebar } from '@docusaurus/theme-common/internal';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { ThemeClassNames } from '@docusaurus/theme-common';
import styles from './LessonLayout.module.css';

function LessonLayout({ children, title, description }) {
  return (
    <Layout title={title} description={description}>
      <div className="container margin-vert--lg">
        <div className="row">
          <main className="col col--9">
            <div className={styles.lessonContent}>
              {children}
            </div>
          </main>
          <aside className="col col--3">
            <div className={styles.lessonSidebar}>
              <h3>Lesson Navigation</h3>
              <ul className={styles.sidebarList}>
                <li><a href="#introduction" className={styles.sidebarLink}>Introduction</a></li>
                <li><a href="#concepts" className={styles.sidebarLink}>Concepts</a></li>
                <li><a href="#technical-deep-dive" className={styles.sidebarLink}>Technical Deep Dive</a></li>
                <li><a href="#diagrams" className={styles.sidebarLink}>Diagrams</a></li>
                <li><a href="#code-examples" className={styles.sidebarLink}>Code Examples</a></li>
                <li><a href="#exercises" className={styles.sidebarLink}>Exercises</a></li>
                <li><a href="#quiz" className={styles.sidebarLink}>Quiz</a></li>
                <li><a href="#summary" className={styles.sidebarLink}>Summary</a></li>
                <li><a href="#key-terms" className={styles.sidebarLink}>Key Terms</a></li>
              </ul>
            </div>
          </aside>
        </div>
      </div>
    </Layout>
  );
}

export default LessonLayout;