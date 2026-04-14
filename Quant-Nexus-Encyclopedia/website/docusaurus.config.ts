import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Quant Nexus Encyclopedia',
  tagline: 'A rigorous, end-to-end reference for quantitative finance — from linear algebra to low-latency alpha.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://quant-nexus.example.com',
  baseUrl: '/',

  organizationName: 'quant-nexus',
  projectName: 'quant-nexus-encyclopedia',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
    format: 'detect',
  },

  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },
    navbar: {
      title: 'Quant Nexus',
      hideOnScroll: true,
      logo: {
        alt: 'Quant Nexus Encyclopedia',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Encyclopedia',
        },
        {
          type: 'search',
          position: 'right',
        },
        {
          href: 'https://github.com/',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Tracks',
          items: [
            {label: 'Foundations', to: '/foundations'},
            {label: 'Computation', to: '/computation'},
            {label: 'Asset Pricing', to: '/asset-pricing'},
            {label: 'Advanced Alpha', to: '/advanced-alpha'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'Docusaurus', href: 'https://docusaurus.io'},
            {label: 'KaTeX', href: 'https://katex.org'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Quant Nexus Encyclopedia. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['python', 'cpp', 'rust', 'bash', 'json', 'yaml', 'sql'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
