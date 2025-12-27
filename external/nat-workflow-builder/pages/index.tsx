import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useState, useRef, useCallback } from 'react';
import { Upload, Download, X, CheckCircle, AlertCircle, Loader2, Play } from 'lucide-react';
import { FlowSidebar } from '@/components/Sidebar/FlowSidebar';
import { ChatSlideOver } from '@/components/Chat';
import { registryAPI, ImportedWorkflowState, ExportComponent, ExportConnection, CreateSessionRequest, ValidateWorkflowRequest } from '@/lib/api';
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
  const [isValidating, setIsValidating] = useState(false);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [showResultModal, setShowResultModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const { loadImportedState, clearWorkflow, components, connections } = useWorkflow();

  // Chat panel state
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatSessionRequest, setChatSessionRequest] = useState<CreateSessionRequest | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

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

  // Run workflow - validate and open chat panel
  const handleRunClick = useCallback(async () => {
    if (components.length === 0) {
      setRunError('Add some components to run the workflow');
      return;
    }

    setIsValidating(true);
    setRunError(null);

    try {
      // Build the request
      const workflowComponents = components.map((comp) => ({
        id: comp.id,
        component_type: comp.type,
        full_type: comp.registeredType?.full_type || '',
        config: comp.config || {},
      }));

      const workflowConnections = connections.map((conn) => ({
        id: conn.id,
        source_id: conn.sourceId,
        target_id: conn.targetId,
        target_field: conn.targetField,
      }));

      // Validate the workflow first
      const validationRequest: ValidateWorkflowRequest = {
        components: workflowComponents,
        connections: workflowConnections,
      };

      const validationResult = await registryAPI.validateWorkflow(validationRequest);

      if (!validationResult.valid) {
        setRunError(validationResult.errors.join('\n'));
        return;
      }

      // Create session request
      const sessionRequest: CreateSessionRequest = {
        components: workflowComponents,
        connections: workflowConnections,
      };

      setChatSessionRequest(sessionRequest);
      setIsChatOpen(true);
    } catch (error) {
      console.error('Run error:', error);
      setRunError(error instanceof Error ? error.message : 'Failed to start workflow');
    } finally {
      setIsValidating(false);
    }
  }, [components, connections]);

  const handleChatClose = useCallback(() => {
    setIsChatOpen(false);
    setChatSessionRequest(null);
  }, []);

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
              <span className="text-sm text-gray-400 hidden lg:block">
                Drag components • Double-click to configure • Connect outputs → inputs
              </span>
              
              {/* Upload Config Button */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".yaml,.yml"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={handleUploadClick}
                disabled={isImporting}
                className="flex items-center gap-2 px-4 py-2 bg-accent/10 hover:bg-accent/20 border border-accent/30 rounded-lg text-accent text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isImporting ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Upload size={16} />
                )}
                <span>{isImporting ? 'Importing...' : 'Import Config'}</span>
              </button>

              {/* Export Config Button */}
              <button
                onClick={handleExportClick}
                disabled={isExporting || components.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-lg text-blue-400 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={components.length === 0 ? 'Add components to export' : 'Export workflow to YAML'}
              >
                {isExporting ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Download size={16} />
                )}
                <span>{isExporting ? 'Exporting...' : 'Export Config'}</span>
              </button>

              {/* Run Workflow Button */}
              <button
                onClick={handleRunClick}
                disabled={isValidating || components.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/90 rounded-lg text-black text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={components.length === 0 ? 'Add components to run' : 'Run your workflow'}
              >
                {isValidating ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Play size={16} fill="currentColor" />
                )}
                <span>{isValidating ? 'Validating...' : 'Run'}</span>
              </button>
            </div>
          </header>

          {/* Run Error Banner */}
          {runError && (
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

          {/* Canvas */}
          <div className="flex-1 overflow-hidden">
            <FlowCanvas />
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

      {/* Chat Slide-Over Panel */}
      <ChatSlideOver
        isOpen={isChatOpen}
        onClose={handleChatClose}
        sessionRequest={chatSessionRequest}
      />
    </>
  );
}
