import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en" className="dark">
      <Head>
        <meta charSet="utf-8" />
        <meta name="description" content="NAT Workflow Builder - Visual agent workflow designer" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <body className="bg-canvas text-gray-100">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

