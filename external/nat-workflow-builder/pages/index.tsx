import Head from 'next/head';
import dynamic from 'next/dynamic';
import { FlowSidebar } from '@/components/Sidebar/FlowSidebar';

// Dynamically import FlowCanvas to avoid SSR issues with React Flow
const FlowCanvas = dynamic(
  () => import('@/components/FlowCanvas/FlowCanvas').then((mod) => mod.FlowCanvas),
  { 
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center bg-canvas">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 text-sm">Loading workflow editor...</p>
        </div>
      </div>
    ),
  }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>NAT Workflow Builder</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="description" content="Visual workflow builder for NeMo Agent Toolkit" />
      </Head>

      <div className="flex h-screen w-screen overflow-hidden bg-canvas">
        {/* Sidebar with component palette */}
        <FlowSidebar />

        {/* Main canvas area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="h-14 bg-canvas-light border-b border-gray-700 flex items-center px-6 justify-between flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                <span className="text-black font-bold text-sm">N</span>
              </div>
              <h1 className="font-display text-lg font-semibold tracking-tight text-white">
                NAT Workflow Builder
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-400">
                Drag components • Double-click to configure • Connect outputs → inputs
              </span>
            </div>
          </header>

          {/* Canvas */}
          <div className="flex-1 overflow-hidden">
            <FlowCanvas />
          </div>
        </main>
      </div>
    </>
  );
}
