import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './ChapterCard.module.css';

function ChapterCard({ title, description, to, number }) {
  return (
    <div className={clsx(styles.card, 'padding--md')}>
      <div className={styles.cardBody}>
        <h3 className={styles.cardTitle}>
          <span className={styles.chapterNumber}>Chapter {number}: </span>
          {title}
        </h3>
        <p className={styles.cardDescription}>{description}</p>
        <Link 
          className={clsx('button button--secondary button--block', styles.cardButton)}
          to={to}
        >
          Start Learning
        </Link>
      </div>
    </div>
  );
}

export default ChapterCard;