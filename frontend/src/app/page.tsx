'use client';

import { useState, useEffect, useRef, ChangeEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import axios from 'axios';
import {
  MessageSquare,
  Search,
  Settings,
  Upload,
  Send,
  Trash2,
  RotateCcw,
  FileText,
  Lightbulb,
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  AlertCircle,
  Loader2
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface JobStatusResponse {
  job_id: number;
  uploaded_files: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  step: string;
  progress: number;
  error_message: string | null;
  uploaded_at: string;
  updated_at: string;
}

interface InspectedNode {
  chunk_id: string;
  filename: string;
  text: string;
  image_url?: string;
  table_url?: string;
  score?: number;
}

interface IngestionVaultProps {
  wipeTrigger: number;
}

function IngestionVault({ wipeTrigger }: IngestionVaultProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state whenever Wipe Knowledge Base action is triggered
  useEffect(() => {
    if (wipeTrigger > 0) {
      setSelectedFiles([]);
      setCurrentJobId(null);
      setJobStatus(null);
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [wipeTrigger]);

  // Poll Job Status API whenever an active jobId exists and job is not done
  useEffect(() => {
    if (!currentJobId) return;

    const pollInterval = setInterval(async () => {
      try {
        const res = await axios.get<JobStatusResponse>(
          `${API_BASE_URL}/job/status/job_id`,
          { params: { job_id: currentJobId } }
        );
        setJobStatus(res.data);

        // Stop polling when job finishes
        if (res.data.status === 'completed' || res.data.status === 'failed') {
          clearInterval(pollInterval);
          setIsUploading(false);
        }
      } catch (err) {
        console.error('Error fetching job status:', err);
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [currentJobId]);

  // Handle local file selection
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  // Submit files to backend ingestion API
  const handleBuildKnowledgeBase = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setJobStatus(null);

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append('files', file));

    try {
      const res = await axios.post(`${API_BASE_URL}/vault/ingest-mixed`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const jobId = res.data.job_id || 1;
      setCurrentJobId(jobId);
    } catch (err: any) {
      setIsUploading(false);
      setJobStatus({
        job_id: 0,
        uploaded_files: selectedFiles.map((f) => f.name).join(', '),
        status: 'failed',
        step: 'Ingestion request failed',
        progress: 0,
        error_message: err.message || 'Upload failed',
        uploaded_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }
  };

  return (
    <div className="bg-[#161B22] border border-slate-800 rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
        <Upload className="w-4 h-4 text-emerald-400" />
        <h2 className="text-sm font-semibold text-white">Ingestion Vault</h2>
      </div>

      <p className="text-xs text-slate-400">Upload files:</p>

      {/* File Dropzone / Picker */}
      <div
        onClick={() => fileInputRef.current?.click()}
        className="border border-dashed border-slate-700 hover:border-emerald-500/50 bg-slate-900/50 hover:bg-slate-900 rounded-lg p-4 text-center cursor-pointer transition-colors"
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".csv,.pdf,.doc,.docx,.txt"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="flex items-center justify-center gap-2 text-slate-400 hover:text-slate-200 text-xs">
          <Upload className="w-4 h-4" />
          <span>Choose CSV, PDF, or Doc files</span>
        </div>
      </div>

      {/* Selected File Names List */}
      {selectedFiles.length > 0 && (
        <div className="flex flex-wrap gap-2 my-1">
          {selectedFiles.map((f, i) => (
            <div
              key={i}
              className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 text-slate-200 text-xs px-2.5 py-1 rounded-md"
            >
              <FileText className="w-3.5 h-3.5 text-emerald-400" />
              <span className="truncate max-w-[200px]">{f.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* Build Knowledge Base Trigger */}
      <button
        onClick={handleBuildKnowledgeBase}
        disabled={selectedFiles.length === 0 || isUploading}
        className="w-full bg-emerald-950/60 hover:bg-emerald-900/60 border border-emerald-700/50 text-emerald-200 hover:text-white text-xs py-2 rounded-md transition flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isUploading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Processing Ingestion...
          </>
        ) : (
          '🚀 Build Knowledge Base Data'
        )}
      </button>

      {/* Real-time Job Status Indicator */}
      {jobStatus && (
        <div className="mt-2 space-y-2 border-t border-slate-800 pt-3 text-xs">
          {jobStatus.uploaded_files && (
            <div className="flex items-center gap-2 text-slate-300">
              <span className="text-slate-500">File:</span>
              <span className="font-mono bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-emerald-300">
                {jobStatus.uploaded_files}
              </span>
            </div>
          )}

          <div className="w-full bg-slate-900 border border-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                jobStatus.status === 'failed'
                  ? 'bg-red-500'
                  : jobStatus.status === 'completed'
                  ? 'bg-emerald-500'
                  : 'bg-emerald-400'
              }`}
              style={{ width: `${jobStatus.progress}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-slate-300">
              {jobStatus.status === 'completed' && (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              )}
              {jobStatus.status === 'failed' && (
                <AlertCircle className="w-3.5 h-3.5 text-red-400" />
              )}
              {jobStatus.status === 'processing' && (
                <Loader2 className="w-3.5 h-3.5 text-emerald-400 animate-spin" />
              )}
              <span
                className={
                  jobStatus.status === 'failed'
                    ? 'text-red-400'
                    : jobStatus.status === 'completed'
                    ? 'text-emerald-400'
                    : 'text-slate-300'
                }
              >
                {jobStatus.step}
              </span>
            </span>

            <span className="text-slate-400 font-mono">{jobStatus.progress}%</span>
          </div>

          {jobStatus.error_message && (
            <p className="text-red-400 text-xs bg-red-950/40 border border-red-800/40 p-2 rounded">
              {jobStatus.error_message}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  // Settings State
  const [maxChunkSize, setMaxChunkSize] = useState<number>(1000);
  const [chunkOverlap, setChunkOverlap] = useState<number>(200);
  const [retrievalK, setRetrievalK] = useState<number>(3);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [topK, setTopK] = useState<number>(40);

  // Chat & Inspector State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState<string>('');
  const [inspectedNodes, setInspectedNodes] = useState<InspectedNode[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Ingestion reset signal counter
  const [wipeTrigger, setWipeTrigger] = useState<number>(0);

  const handleSendMessage = async () => {
    if (!inputQuery.trim() || isLoading) return;

    const userMsg = inputQuery;
    setInputQuery('');
    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/vault/chat`, {
        messages: [{ role: 'user', content: userMsg }],
        retrieval_k: retrievalK,
        temperature: temperature,
        top_k: topK,
      });

      const { content, inspected_nodes } = response.data;
      setMessages((prev) => [...prev, { role: 'assistant', content }]);
      if (inspected_nodes) setInspectedNodes(inspected_nodes);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '❌ Connection error to agent service.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleWipeData = async () => {
    if (!confirm('Are you sure you want to wipe all knowledge base data?')) return;
    try {
      await axios.post(`${API_BASE_URL}/job/clear`);
      setInspectedNodes([]);
      setWipeTrigger((prev) => prev + 1); // Trigger full reset in IngestionVault
      alert('Knowledge base cleared successfully.');
    } catch {
      alert('Failed to clear knowledge base.');
    }
  };

  return (
    <div className="min-h-screen bg-[#0E1117] text-slate-200 p-6 flex flex-col gap-6">
      {/* Header */}
      <header className="flex items-center gap-2 border-b border-slate-800 pb-4">
        <Cpu className="w-6 h-6 text-emerald-400" />
        <h1 className="text-xl font-bold text-white tracking-wide">
          LangGraph Multimodal AI Agent Hub
        </h1>
      </header>

      {/* Main Grid: 12 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Section (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col gap-6">

          {/* Row 1: Side-by-Side Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Active Conversation Space */}
            <div className="bg-[#161B22] border border-slate-800 rounded-lg p-4 h-[420px] flex flex-col">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-800 mb-3">
                <MessageSquare className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-white">Active Conversation Space</h2>
              </div>
              <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {messages.length === 0 ? (
                  <p className="text-slate-500 text-xs text-center mt-32">
                    No active conversation. Send a message to start.
                  </p>
                ) : (
                  messages.map((m, i) => (
                    <div
                      key={i}
                      className={`p-2.5 rounded text-xs ${
                        m.role === 'user'
                          ? 'bg-emerald-950/40 border border-emerald-800/40 text-emerald-200 ml-4'
                          : 'bg-slate-900 border border-slate-800 text-slate-300 mr-4'
                      }`}
                    >
                      {m.role === 'user' ? (
                        <span className="whitespace-pre-wrap">{m.content}</span>
                      ) : (
                        <div className="prose prose-invert max-w-none text-xs text-slate-300 
                            [&_table]:w-full [&_table]:border-collapse [&_table]:my-2 
                            [&_th]:border [&_th]:border-slate-700 [&_th]:p-1.5 [&_th]:bg-slate-800 [&_th]:text-left
                            [&_td]:border [&_td]:border-slate-800 [&_td]:p-1.5 
                            [&_p]:my-1 [&_ul]:list-disc [&_ul]:ml-4 [&_ol]:list-decimal [&_ol]:ml-4">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {m.content}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  ))
                )}
                {isLoading && (
                  <p className="text-xs text-emerald-400 animate-pulse">Agent is typing...</p>
                )}
              </div>
            </div>

            {/* Context & Asset Inspector */}
            <div className="bg-[#161B22] border border-slate-800 rounded-lg p-4 h-[420px] flex flex-col">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-800 mb-3">
                <Search className="w-4 h-4 text-cyan-400" />
                <h2 className="text-sm font-semibold text-white">Context & Asset Inspector</h2>
              </div>
              <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {inspectedNodes.length === 0 ? (
                  <div className="bg-[#1C2128] border border-cyan-900/40 rounded p-3 text-cyan-300 text-xs flex items-start gap-2">
                    <Lightbulb className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <span>
                      Any extracted tables, schemas, or diagram layouts selected by the agent tools
                      will stack right here.
                    </span>
                  </div>
                ) : (
                  inspectedNodes.map((node, i) => (
                    <div
                      key={i}
                      className="bg-[#0D1117] border border-slate-800 rounded p-2.5 text-xs"
                    >
                      <p className="text-cyan-400 font-semibold mb-1">{node.filename}</p>
                      <p className="text-slate-300 text-[11px] font-mono">{node.text}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Ask to Agent */}
          <div className="bg-[#161B22] border border-slate-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <MessageSquare className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-white">Ask to Agent</h2>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Message your local knowledge agent..."
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                className="flex-1 bg-[#0D1117] border border-slate-800 rounded px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading}
                className="bg-slate-800 hover:bg-emerald-600 transition-colors px-3 py-2 rounded text-white"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Row 3: Ingestion Vault Component */}
          <IngestionVault wipeTrigger={wipeTrigger} />
        </div>

        {/* Right Sidebar: Agent Settings (4 Cols) */}
        <div className="lg:col-span-4 bg-[#161B22] border border-slate-800 rounded-lg p-5 flex flex-col justify-between h-full">
          <div className="space-y-6">
            <div className="border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 mb-1">
                <Settings className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-white">Agent Settings</h2>
              </div>
              <p className="text-[11px] text-slate-400">Framework: LangGraph State Machine</p>
            </div>

            {/* Chunking Hyperparameters */}
            <div className="space-y-4">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
                <Layers className="w-3.5 h-3.5" />
                <span>Chunking Hyperparameters</span>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Max Chunk Size</span>
                  <span className="text-emerald-400 font-mono">{maxChunkSize}</span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="4000"
                  step="100"
                  value={maxChunkSize}
                  onChange={(e) => setMaxChunkSize(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Chunk Overlap Block</span>
                  <span className="text-emerald-400 font-mono">{chunkOverlap}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1000"
                  step="50"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>
            </div>

            <hr className="border-slate-800" />

            {/* Agent Parameters */}
            <div className="space-y-4">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
                <Database className="w-3.5 h-3.5" />
                <span>Agent Parameters</span>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Database Chunks (Retrieval K)</span>
                  <span className="text-emerald-400 font-mono">{retrievalK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={retrievalK}
                  onChange={(e) => setRetrievalK(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Generation Temperature</span>
                  <span className="text-emerald-400 font-mono">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Generation Top-K Window</span>
                  <span className="text-emerald-400 font-mono">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Sidebar Bottom Buttons */}
          <div className="space-y-2 pt-6 mt-6 border-t border-slate-800">
            <button
              onClick={handleWipeData}
              className="w-full bg-[#0D1117] hover:bg-red-950/40 border border-slate-800 hover:border-red-900 text-slate-300 hover:text-red-400 text-xs py-2 rounded transition-colors flex items-center justify-center gap-2"
            >
              <Trash2 className="w-3.5 h-3.5 text-red-400" />
              <span>Wipe Knowledge Base Data</span>
            </button>

            <button
              onClick={() => {
                setMessages([]);
                setInspectedNodes([]);
              }}
              className="w-full bg-[#0D1117] hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs py-2 rounded transition-colors flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear Conversation</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}