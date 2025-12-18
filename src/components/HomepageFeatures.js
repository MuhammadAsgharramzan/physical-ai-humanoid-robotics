import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Learn Physical AI',
    Svg: require('../../static/img/ai-icon.svg').default,
    description: (
      <>
        Explore the fundamentals of Physical AI and how it differs from traditional digital intelligence.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: require('../../static/img/robot-icon.svg').default,
    description: (
      <>
        Understand the current state of humanoid robotics and the challenges involved in creating embodied agents.
      </>
    ),
  },
  {
    title: 'Practical Applications',
    Svg: require('../../static/img/application-icon.svg').default,
    description: (
      <>
        Discover real-world applications and deployment strategies for humanoid robots in various environments.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}