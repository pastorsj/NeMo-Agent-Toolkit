import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useState, useRef, useCallback } from 'react';
import { Upload, Download, X, CheckCircle, AlertCircle, Loader2, Play, Trash2, ExternalLink, Square, RefreshCw } from 'lucide-react';
import { FlowSidebar } from '@/components/Sidebar/FlowSidebar';
import { EnvVarPanel, EnvVarBadge } from '@/components/EnvVars/EnvVarPanel';
import { registryAPI, ImportedWorkflowState, ExportComponent, ExportConnection } from '@/lib/api';
import { useWorkflow } from '@/contexts/WorkflowContext';

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

interface ImportResult {
  success: boolean;
  error_message: string | null;
  error_details: string[] | null;
  workflow_state: ImportedWorkflowState | null;
}

interface ExportResult {
  success: boolean;
  error_message: string | null;
  warnings: string[];
  yaml_content: string | null;
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [showResultModal, setShowResultModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const { loadImportedState, clearWorkflow, components, connections, hasUnresolvedEnvVars, environmentVariables } = useWorkflow();

  // Environment variables panel state
  const [showEnvVarPanel, setShowEnvVarPanel] = useState(false);

  // Run workflow state
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runningWorkflow, setRunningWorkflow] = useState<{ url: string; processId: string } | null>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file extension
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith('.yaml') && !fileName.endsWith('.yml')) {
      setImportResult({
        success: false,
        error_message: 'Invalid file type',
        error_details: ['Please upload a YAML file (.yaml or .yml)'],
        workflow_state: null,
      });
      setShowResultModal(true);
      return;
    }

    setUploadedFileName(file.name);
    setIsImporting(true);

    try {
      // Read file content
      const yamlContent = await file.text();

      // Send to import API (validates and parses in one step)
      const workflowState = await registryAPI.importConfig(yamlContent);
      
      setImportResult({
        success: true,
        error_message: null,
        error_details: null,
        workflow_state: workflowState,
      });
      setShowResultModal(true);
    } catch (error) {
      console.error('Import error:', error);
      setImportResult({
        success: false,
        error_message: 'Failed to import configuration',
        error_details: [error instanceof Error ? error.message : 'Unknown error occurred'],
        workflow_state: null,
      });
      setShowResultModal(true);
    } finally {
      setIsImporting(false);
      // Reset file input so the same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const closeModal = () => {
    setShowResultModal(false);
    setImportResult(null);
  };

  const handleLoadWorkflow = () => {
    if (importResult?.workflow_state) {
      // Clear existing workflow and load the imported state
      clearWorkflow();
      loadImportedState(importResult.workflow_state);
      closeModal();
    }
  };

  // Export workflow to YAML
  const handleExportClick = async () => {
    if (components.length === 0) {
      setExportResult({
        success: false,
        error_message: 'No components to export',
        warnings: [],
        yaml_content: null,
      });
      setShowExportModal(true);
      return;
    }

    setIsExporting(true);

    try {
      // Convert components to export format
      const exportComponents: ExportComponent[] = components.map((comp) => ({
        id: comp.id,
        component_type: comp.type,
        full_type: comp.registeredType?.full_type || '',
        config: comp.config || {},
      }));

      // Convert connections to export format
      const exportConnections: ExportConnection[] = connections.map((conn) => ({
        source_id: conn.sourceId,
        target_id: conn.targetId,
        target_field: conn.targetField,
      }));

      const result = await registryAPI.exportConfig({
        components: exportComponents,
        connections: exportConnections,
        workflow_name: 'my_workflow',
      });

      setExportResult({
        success: result.success,
        error_message: result.error_message,
        warnings: result.warnings,
        yaml_content: result.yaml_content,
      });
      setShowExportModal(true);
    } catch (error) {
      console.error('Export error:', error);
      setExportResult({
        success: false,
        error_message: error instanceof Error ? error.message : 'Unknown error occurred',
        warnings: [],
        yaml_content: null,
      });
      setShowExportModal(true);
    } finally {
      setIsExporting(false);
    }
  };

  const closeExportModal = () => {
    setShowExportModal(false);
    setExportResult(null);
  };

  const handleDownloadYaml = () => {
    if (exportResult?.yaml_content) {
      const blob = new Blob([exportResult.yaml_content], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'workflow.yaml';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      closeExportModal();
    }
  };

  // Run workflow - export and start nat serve
  const handleRunClick = useCallback(async () => {
    if (components.length === 0) {
      setRunError('Add some components to run the workflow');
      return;
    }

    setIsStarting(true);
    setRunError(null);

    try {
      // Convert components to export format
      const exportComponents: ExportComponent[] = components.map((comp) => ({
        id: comp.id,
        component_type: comp.type,
        full_type: comp.registeredType?.full_type || '',
        config: comp.config || {},
      }));

      // Convert connections to export format
      const exportConnections: ExportConnection[] = connections.map((conn) => ({
        source_id: conn.sourceId,
        target_id: conn.targetId,
        target_field: conn.targetField,
      }));

      // Start the workflow using nat serve
      const result = await registryAPI.startWorkflow({
        components: exportComponents,
        connections: exportConnections,
        workflow_name: 'workflow_builder_runtime',
      });

      if (!result.success) {
        setRunError(result.error_message || 'Failed to start workflow');
        return;
      }

      // Store the running workflow info
      setRunningWorkflow({
        url: result.url,
        processId: result.process_id,
      });

      // Open the nat-ui in a new tab
      window.open(result.url, '_blank');
    } catch (error) {
      console.error('Run error:', error);
      setRunError(error instanceof Error ? error.message : 'Failed to start workflow');
    } finally {
      setIsStarting(false);
    }
  }, [components, connections]);

  // Stop running workflow
  const handleStopClick = useCallback(async () => {
    if (!runningWorkflow) return;

    setIsStopping(true);
    setRunError(null);

    try {
      await registryAPI.stopWorkflow(runningWorkflow.processId);
      setRunningWorkflow(null);
    } catch (error) {
      console.error('Stop error:', error);
      setRunError(error instanceof Error ? error.message : 'Failed to stop workflow');
    } finally {
      setIsStopping(false);
    }
  }, [runningWorkflow]);

  // Restart workflow with current config
  const handleRestartClick = useCallback(async () => {
    if (!runningWorkflow) return;

    // Stop the current workflow first
    setIsStopping(true);
    try {
      await registryAPI.stopWorkflow(runningWorkflow.processId);
      setRunningWorkflow(null);
    } catch (error) {
      console.error('Stop error during restart:', error);
    } finally {
      setIsStopping(false);
    }

    // Start a new workflow
    await handleRunClick();
  }, [runningWorkflow, handleRunClick]);

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
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400 hidden lg:block mr-2">
                Drag components • Double-click to configure • Connect outputs → inputs
              </span>

              {/* Hidden file input for import */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".yaml,.yml"
                onChange={handleFileChange}
                className="hidden"
              />

              {/* Import Button */}
              <button
                onClick={handleUploadClick}
                disabled={isImporting}
                className="p-2.5 bg-gray-700/50 hover:bg-gray-600 border border-gray-600 rounded-lg text-gray-300 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Import YAML config"
              >
                {isImporting ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Upload size={18} />
                )}
              </button>

              {/* Export Button */}
              <button
                onClick={handleExportClick}
                disabled={isExporting || components.length === 0}
                className="p-2.5 bg-gray-700/50 hover:bg-gray-600 border border-gray-600 rounded-lg text-gray-300 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={components.length === 0 ? 'Add components to export' : 'Export to YAML'}
              >
                {isExporting ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Download size={18} />
                )}
              </button>

              {/* Run Button - disabled when workflow is already running or has unresolved env vars */}
              <button
                onClick={handleRunClick}
                disabled={isStarting || components.length === 0 || runningWorkflow !== null || hasUnresolvedEnvVars}
                className="p-2.5 bg-accent hover:bg-accent/80 rounded-lg text-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                title={
                  hasUnresolvedEnvVars
                    ? 'Please provide values for all environment variables first'
                    : runningWorkflow
                    ? 'Workflow already running - use Restart or Stop'
                    : components.length === 0
                    ? 'Add components to run'
                    : 'Run workflow (opens nat-ui)'
                }
              >
                {isStarting ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    <span className="text-xs font-medium hidden sm:inline">Starting...</span>
                  </>
                ) : (
                  <Play size={18} fill="currentColor" />
                )}
              </button>

              {/* Running Workflow Controls */}
              {runningWorkflow && (
                <>
                  {/* Open in Browser */}
                  <button
                    onClick={() => window.open(runningWorkflow.url, '_blank')}
                    className="p-2.5 bg-green-600 hover:bg-green-500 rounded-lg text-white transition-colors"
                    title={`Open running workflow at ${runningWorkflow.url}`}
                  >
                    <ExternalLink size={18} />
                  </button>

                  {/* Restart with new config */}
                  <button
                    onClick={handleRestartClick}
                    disabled={isStarting || isStopping}
                    className="p-2.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Restart workflow with current config"
                  >
                    {isStarting ? (
                      <Loader2 size={18} className="animate-spin" />
                    ) : (
                      <RefreshCw size={18} />
                    )}
                  </button>

                  {/* Stop */}
                  <button
                    onClick={handleStopClick}
                    disabled={isStopping}
                    className="p-2.5 bg-red-600 hover:bg-red-500 rounded-lg text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Stop running workflow"
                  >
                    {isStopping ? (
                      <Loader2 size={18} className="animate-spin" />
                    ) : (
                      <Square size={18} fill="currentColor" />
                    )}
                  </button>
                </>
              )}

              {/* Environment Variables Badge */}
              {environmentVariables.length > 0 && (
                <EnvVarBadge onClick={() => setShowEnvVarPanel(!showEnvVarPanel)} />
              )}

              {/* Clear/Delete Button */}
              <button
                onClick={() => {
                  if (components.length === 0) return;
                  if (confirm('Clear all components and connections?')) {
                    clearWorkflow();
                  }
                }}
                disabled={components.length === 0}
                className="p-2.5 bg-gray-700/50 hover:bg-red-500/20 border border-gray-600 hover:border-red-500/50 rounded-lg text-gray-300 hover:text-red-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={components.length === 0 ? 'No components to clear' : 'Clear workflow'}
              >
                <Trash2 size={18} />
              </button>
            </div>
          </header>

          {/* Starting Workflow Banner */}
          {isStarting && (
            <div className="bg-accent/10 border-b border-accent/30 px-6 py-3 flex items-center gap-3">
              <Loader2 size={18} className="text-accent animate-spin flex-shrink-0" />
              <p className="text-sm text-accent flex-1">
                Starting workflow... This may take 1-2 minutes (backend initialization + UI compilation).
              </p>
            </div>
          )}

          {/* Running Workflow Banner */}
          {runningWorkflow && !isStarting && (
            <div className="bg-green-500/10 border-b border-green-500/30 px-6 py-3 flex items-center gap-3">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse flex-shrink-0" />
              <p className="text-sm text-green-400 flex-1">
                Workflow running at <a href={runningWorkflow.url} target="_blank" rel="noopener noreferrer" className="underline hover:text-green-300">{runningWorkflow.url}</a>
              </p>
              <span className="text-xs text-green-500/70">
                Use the buttons above to open, restart, or stop
              </span>
            </div>
          )}

          {/* Run Error Banner */}
          {runError && !isStarting && (
            <div className="bg-red-500/10 border-b border-red-500/30 px-6 py-3 flex items-center gap-3">
              <AlertCircle size={18} className="text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-400 flex-1 whitespace-pre-line">{runError}</p>
              <button
                onClick={() => setRunError(null)}
                className="text-red-400 hover:text-red-300 p-1"
              >
                <X size={16} />
              </button>
            </div>
          )}

          {/* Canvas + Environment Variables Panel */}
          <div className="flex-1 flex overflow-hidden">
            {/* Canvas */}
            <div className="flex-1 overflow-hidden">
              <FlowCanvas />
            </div>

            {/* Environment Variables Panel (right side) */}
            <EnvVarPanel 
              isOpen={showEnvVarPanel} 
              onClose={() => setShowEnvVarPanel(false)} 
            />
          </div>
        </main>
      </div>

      {/* Import Result Modal */}
      {showResultModal && importResult && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-canvas-light border border-gray-700 rounded-xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className={`px-6 py-4 border-b border-gray-700 flex items-center justify-between ${
              importResult.success ? 'bg-green-500/10' : 'bg-red-500/10'
            }`}>
              <div className="flex items-center gap-3">
                {importResult.success ? (
                  <CheckCircle size={24} className="text-green-500" />
                ) : (
                  <AlertCircle size={24} className="text-red-500" />
                )}
                <div>
                  <h2 className="font-display text-lg font-semibold text-white">
                    {importResult.success ? 'Configuration Parsed Successfully' : 'Import Failed'}
                  </h2>
                  <p className="text-sm text-gray-400">{uploadedFileName}</p>
                </div>
              </div>
              <button
                onClick={closeModal}
                className="p-2 hover:bg-gray-700/50 rounded-lg transition-colors"
              >
                <X size={20} className="text-gray-400" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6">
              {importResult.success && importResult.workflow_state ? (
                <div className="space-y-4">
                  <div className="text-center py-2">
                    <p className="text-green-400 text-lg font-medium mb-2">
                      ✓ Configuration parsed successfully!
                    </p>
                    <p className="text-gray-400 text-sm">
                      Found {importResult.workflow_state.components.length} components and{' '}
                      {importResult.workflow_state.connections.length} connections.
                    </p>
                  </div>
                  
                  {/* Component summary */}
                  {importResult.workflow_state.components.length > 0 && (
                    <div className="bg-gray-900 rounded-lg p-4">
                      <p className="text-sm text-gray-400 font-medium mb-2">Components to load:</p>
                      <div className="flex flex-wrap gap-2">
                        {importResult.workflow_state.components.map((comp) => (
                          <span
                            key={comp.id}
                            className="text-xs px-2 py-1 bg-gray-800 rounded text-gray-300 border border-gray-700"
                          >
                            {comp.name}
                            <span className="text-gray-500 ml-1">({comp.component_type})</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <p className="text-sm text-amber-400 bg-amber-500/10 px-3 py-2 rounded border border-amber-500/20">
                    ⚠️ Loading this workflow will replace any existing components on the canvas.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {importResult.error_message && (
                    <p className="text-red-400 font-medium">
                      {importResult.error_message}
                    </p>
                  )}
                  
                  {importResult.error_details && importResult.error_details.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-sm text-gray-400 font-medium">Error Details:</p>
                      <div className="bg-gray-900 rounded-lg p-4 space-y-2 max-h-60 overflow-y-auto">
                        {importResult.error_details.map((detail, index) => (
                          <div
                            key={index}
                            className="text-sm font-mono text-red-300 bg-red-500/10 px-3 py-2 rounded border border-red-500/20"
                          >
                            {detail}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 border-t border-gray-700 flex justify-end gap-3">
              <button
                onClick={closeModal}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              {importResult.success && importResult.workflow_state && (
                <button
                  onClick={handleLoadWorkflow}
                  className="px-4 py-2 bg-accent hover:bg-accent/90 rounded-lg text-black text-sm font-semibold transition-colors"
                >
                  Load Workflow
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Export Result Modal */}
      {showExportModal && exportResult && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-canvas-light border border-gray-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className={`px-6 py-4 border-b border-gray-700 flex items-center justify-between ${
              exportResult.success ? 'bg-green-500/10' : 'bg-red-500/10'
            }`}>
              <div className="flex items-center gap-3">
                {exportResult.success ? (
                  <CheckCircle size={24} className="text-green-500" />
                ) : (
                  <AlertCircle size={24} className="text-red-500" />
                )}
                <div>
                  <h2 className="font-display text-lg font-semibold text-white">
                    {exportResult.success ? 'Export Successful' : 'Export Failed'}
                  </h2>
                  <p className="text-sm text-gray-400">
                    {exportResult.success
                      ? 'Your workflow is ready to download'
                      : 'There was an issue exporting your workflow'}
                  </p>
                </div>
              </div>
              <button
                onClick={closeExportModal}
                className="p-2 hover:bg-gray-700/50 rounded-lg transition-colors"
              >
                <X size={20} className="text-gray-400" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6">
              {exportResult.success && exportResult.yaml_content ? (
                <div className="space-y-4">
                  {/* Warnings */}
                  {exportResult.warnings.length > 0 && (
                    <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                      <p className="text-sm text-amber-400 font-medium mb-2">Warnings:</p>
                      <ul className="list-disc list-inside text-sm text-amber-300 space-y-1">
                        {exportResult.warnings.map((warning, index) => (
                          <li key={index}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* YAML Preview */}
                  <div className="space-y-2">
                    <p className="text-sm text-gray-400 font-medium">Generated YAML:</p>
                    <pre className="bg-gray-900 rounded-lg p-4 text-sm font-mono text-gray-300 overflow-x-auto max-h-80 overflow-y-auto border border-gray-700">
                      {exportResult.yaml_content}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {exportResult.error_message && (
                    <p className="text-red-400 font-medium">
                      {exportResult.error_message}
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 border-t border-gray-700 flex justify-end gap-3">
              <button
                onClick={closeExportModal}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Close
              </button>
              {exportResult.success && exportResult.yaml_content && (
                <button
                  onClick={handleDownloadYaml}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-semibold transition-colors"
                >
                  <Download size={16} />
                  Download YAML
                </button>
              )}
            </div>
          </div>
        </div>
      )}

    </>
  );
}
