import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { WorkflowProvider } from '@/contexts/WorkflowContext';
import { RegistryProvider } from '@/contexts/RegistryContext';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <RegistryProvider>
      <WorkflowProvider>
        <Component {...pageProps} />
      </WorkflowProvider>
    </RegistryProvider>
  );
}
