import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

type Track = {
  num: string;
  title: string;
  href: string;
  description: string;
  modules: string;
};

const tracks: Track[] = [
  {
    num: 'Track 1',
    title: 'Foundations',
    href: '/foundations',
    description:
      'Linear algebra, measure-theoretic probability, stochastic calculus, optimization, information theory, numerical methods.',
    modules: 'Modules 01–08',
  },
  {
    num: 'Track 2',
    title: 'Computation',
    href: '/computation',
    description:
      'Python for quants, C++ low-latency, Rust systems, data structures, FPGA hardware, tick-data stores, distributed systems.',
    modules: 'Modules 09–16',
  },
  {
    num: 'Track 3',
    title: 'Asset Pricing',
    href: '/asset-pricing',
    description:
      'Equilibrium pricing, derivatives & Greeks, stochastic volatility, fixed income, time series, Kalman filters, microstructure, portfolio risk.',
    modules: 'Modules 17–24',
  },
  {
    num: 'Track 4',
    title: 'Advanced Alpha',
    href: '/advanced-alpha',
    description:
      'Stat arb, ML & deep learning, RL execution, NLP/LLMs, HFT, TCA, backtesting, regime detection, alternative data.',
    modules: 'Modules 25–34',
  },
];

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={`hero-gradient ${styles.hero}`}>
      <div className="container">
        <div className={styles.eyebrow}>Quantitative Finance · Reference</div>
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.ctaRow}>
          <Link className="button button--primary button--lg" to="/foundations">
            Start with Foundations
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/advanced-alpha">
            Jump to Alpha
          </Link>
        </div>
        <div className={styles.stats}>
          <div>
            <strong>34</strong>
            <span>Modules</span>
          </div>
          <div>
            <strong>4</strong>
            <span>Tracks</span>
          </div>
          <div>
            <strong>∞</strong>
            <span>Rabbit holes</span>
          </div>
        </div>
      </div>
    </header>
  );
}

function Tracks() {
  return (
    <section className={styles.tracks}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Four tracks, one curriculum
        </Heading>
        <p className={styles.sectionLede}>
          Each module is self-contained with prerequisites, forward references, worked
          examples in Python and C++, and exercises. Read linearly, or jump in at any
          track.
        </p>
        <div className={styles.grid}>
          {tracks.map((t) => (
            <Link key={t.title} to={t.href} className="track-card">
              <div className="track-num">{t.num} · {t.modules}</div>
              <h3>{t.title}</h3>
              <p>{t.description}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function Pitch() {
  return (
    <section className={styles.pitch}>
      <div className="container">
        <div className={styles.pitchGrid}>
          <div>
            <Heading as="h3">Rigorous, not hand-wavy</Heading>
            <p>
              Theorems stated precisely, derivations shown, assumptions labelled. Where
              the math matters, the math is on the page.
            </p>
          </div>
          <div>
            <Heading as="h3">Built for practitioners</Heading>
            <p>
              Every foundational idea is tied to an implementation — Python, C++, or
              Rust — and to the market mechanism it actually governs.
            </p>
          </div>
          <div>
            <Heading as="h3">End-to-end</Heading>
            <p>
              From inner products to order books, from PCA to RL execution. One
              consistent notation, one cross-referenced graph of modules.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A rigorous, end-to-end reference for quantitative finance.">
      <Hero />
      <main>
        <Tracks />
        <Pitch />
      </main>
    </Layout>
  );
}
