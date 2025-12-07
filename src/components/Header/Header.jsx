import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useColorMode } from '@docusaurus/theme-common';
import { useBaseUrl } from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './Header.module.css';

function Header() {
  const { colorMode } = useColorMode();
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={clsx(
              'button button--secondary button--lg',
              styles.getStarted,
            )}
            to={useBaseUrl('docs/intro')}>
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

export default Header;