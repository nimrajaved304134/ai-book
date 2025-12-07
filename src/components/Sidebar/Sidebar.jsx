import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useLocation } from '@docusaurus/router';
import { usePluralForm } from '@docusaurus/theme-common';
import { useDocsSidebar } from '@docusaurus/theme-common/internal';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { ThemeClassNames } from '@docusaurus/theme-common';
import styles from './Sidebar.module.css';

function Sidebar() {
  const { pathname } = useLocation();
  const { sidebar } = useDocsSidebar();

  return (
    <aside className={clsx(ThemeClassNames.docs.docSidebarContainer, styles.sidebar)}>
      <nav className={ThemeClassNames.docs.docSidebarMenu}>
        <ul className={clsx(styles.menu, 'clean-list')}>
          {sidebar.map((item, index) => (
            <li key={index} className={styles.menuItem}>
              <Link 
                to={useBaseUrl(item.href)} 
                className={clsx(styles.menuLink, {
                  [styles.menuLinkActive]: pathname === item.href
                })}
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}

export default Sidebar;