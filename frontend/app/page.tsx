"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Image as ImageIcon, User, Bot, Loader2, X, Sparkles, Shield, Database, ChevronRight, HelpCircle, ArrowRight, UploadCloud, CheckCircle2, AlertCircle, RefreshCw, Settings, FileText } from "lucide-react";
import axios from "axios";

interface Message {
  role: "user" | "assistant";
  content: string;
  image_url?: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SUGGESTED_PROMPTS = [
  {
    title: "Troubleshoot SuperWidget",
    desc: "Device not connecting & setup",
    prompt: "How do I troubleshoot a SuperWidget 3000 that won't connect?",
    category: "technical"
  },
  {
    title: "Shipping & Rates",
    desc: "Free standard shipping details",
    prompt: "What is your shipping policy and how long does standard shipping take?",
    category: "shipping"
  },
  {
    title: "30-Day Return Policy",
    desc: "Refund eligibility and packaging",
    prompt: "Can I return a product after 20 days for a full refund?",
    category: "returns"
  }
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [department, setDepartment] = useState("general");
  const [showInsights, setShowInsights] = useState(true);
  
  // Dynamic Console States
  const [activeTab, setActiveTab] = useState<"chat" | "console">("chat");
  const [serverHealthy, setServerHealthy] = useState(false);
  const [dbConnected, setDbConnected] = useState(false);
  
  const [oracleTables, setOracleTables] = useState<string[]>([]);
  const [tableMetadata, setTableMetadata] = useState<Record<string, string>>({});
  const [editingMetadata, setEditingMetadata] = useState<Record<string, string>>({});
  
  const [dbConfig, setDbConfig] = useState({
    user: "system",
    password: "",
    host: "localhost",
    port: 1521,
    service_name: "xe"
  });
  
  const [modelConfig, setModelConfig] = useState({
    MODEL_NAME: "qwen2.5",
    EMBEDDING_MODEL: "nomic-embed-text",
    LOCAL_LLM_BASE_URL: "http://localhost:11434/v1",
    LOCAL_EMBEDDING_BASE_URL: "http://localhost:11434/v1",
    VECTOR_DB_TYPE: "qdrant"
  });

  const [backendConfig, setBackendConfig] = useState({
    llm_model: "qwen2.5",
    embedding_model: "nomic-embed-text",
    vector_db: "qdrant"
  });

  const [ragFile, setRagFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchConfigAndTables = async () => {
    try {
      const configRes = await axios.get(`${API_URL}/api/config`);
      if (configRes.data.db_config) {
        setDbConfig(configRes.data.db_config);
      }
      if (configRes.data.model_config) {
        const mConfig = configRes.data.model_config;
        setModelConfig(mConfig);
        setBackendConfig({
          llm_model: mConfig.MODEL_NAME,
          embedding_model: mConfig.EMBEDDING_MODEL,
          vector_db: mConfig.VECTOR_DB_TYPE
        });
      }
      
      const tablesRes = await axios.get(`${API_URL}/api/database/tables`);
      const fetchedTables = tablesRes.data.tables || [];
      setOracleTables(fetchedTables);
      setTableMetadata(tablesRes.data.metadata || {});
      
      // Seed editing text fields
      const initialEditing: Record<string, string> = {};
      fetchedTables.forEach((tbl: string) => {
        initialEditing[tbl] = tablesRes.data.metadata?.[tbl] || "";
      });
      setEditingMetadata(initialEditing);
    } catch (err) {
      console.error("Failed to fetch settings from backend:", err);
    }
  };

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get(`${API_URL}/health`);
        setServerHealthy(true);
        setDbConnected(!!res.data.db_connected);
      } catch (err) {
        setServerHealthy(false);
        setDbConnected(false);
      }
    };
    
    checkHealth();
    fetchConfigAndTables();
    
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSaveDbConfig = async () => {
    try {
      await axios.post(`${API_URL}/api/config`, { db_config: dbConfig });
      alert("Database configuration saved successfully! Re-testing connectivity.");
      setTimeout(fetchConfigAndTables, 1000);
    } catch (err: any) {
      alert("Failed to save database config: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleSaveModelConfig = async () => {
    try {
      await axios.post(`${API_URL}/api/config`, { model_config: modelConfig });
      alert("Model configurations saved successfully!");
      fetchConfigAndTables();
    } catch (err: any) {
      alert("Failed to save model config: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleSaveTableMetadata = async (tableName: string) => {
    const desc = editingMetadata[tableName] || "";
    try {
      await axios.post(`${API_URL}/api/database/tables/metadata`, {
        table_name: tableName,
        description: desc
      });
      setTableMetadata(prev => ({
        ...prev,
        [tableName.toLowerCase()]: desc
      }));
      alert(`Description for '${tableName.toUpperCase()}' updated successfully!`);
    } catch (err: any) {
      alert("Failed to save table metadata: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleRagFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ragFile) return;
    
    setIsUploading(true);
    setUploadMessage("Uploading and indexing document in RAG vector store... This might take a minute.");
    
    const formData = new FormData();
    formData.append("file", ragFile);
    
    try {
      await axios.post(`${API_URL}/api/rag/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setUploadMessage(`Successfully uploaded and indexed '${ragFile.name}'!`);
      setRagFile(null);
      // Refresh insights if needed
      fetchConfigAndTables();
    } catch (err: any) {
      setUploadMessage("Failed to upload and index document: " + (err.response?.data?.detail || err.message));
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSend = async (customPrompt?: string) => {
    const textToSend = customPrompt || input;
    if (!textToSend.trim() && !image) return;

    const newMessage: Message = {
      role: "user",
      content: textToSend,
      image_url: image || undefined,
    };

    const updatedHistory = [...messages, newMessage];
    setMessages(updatedHistory);
    setInput("");
    setImage(null);
    setIsLoading(true);
    setIsThinking(true);

    try {
      // 1. Initialize an empty assistant message to show loading state immediately
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "" },
      ]);

      // 2. Initiate the stream fetch
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedHistory.map((m) => ({
            role: m.role,
            content: m.content,
            image_url: m.image_url,
          })),
          department: department,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No readable stream found in response body.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let isDone = false;
      let streamBuffer = "";

      while (!isDone) {
        const { value, done } = await reader.read();
        isDone = done;

        if (value) {
          streamBuffer += decoder.decode(value, { stream: true });
          
          // Process the stream buffer looking for complete SSE data blocks (separated by double newlines)
          let eventBoundary = streamBuffer.indexOf("\n\n");
          
          while (eventBoundary !== -1) {
            const payload = streamBuffer.slice(0, eventBoundary).trim();
            streamBuffer = streamBuffer.slice(eventBoundary + 2);

            if (payload.startsWith("data: ")) {
              try {
                const dataStr = payload.slice(6);
                const parsed = JSON.parse(dataStr);

                if (parsed.token) {
                  // Turn off thinking mode as soon as first token arrives
                  setIsThinking(false);

                  // Update the LAST message array iteratively using React's setter callback
                  setMessages((prev) => {
                    const updatedMessages = [...prev];
                    const finalIdx = updatedMessages.length - 1;
                    
                    if (finalIdx >= 0 && updatedMessages[finalIdx].role === "assistant") {
                      updatedMessages[finalIdx] = {
                        ...updatedMessages[finalIdx],
                        content: updatedMessages[finalIdx].content + parsed.token,
                      };
                    }
                    return updatedMessages;
                  });
                } else if (parsed.error) {
                  throw new Error(parsed.error);
                }
              } catch (e) {
                console.error("Failed to parse SSE chunk:", e);
              }
            }
            eventBoundary = streamBuffer.indexOf("\n\n");
          }
        }
      }
    } catch (error) {
      console.error("Chat Error:", error);
      setIsThinking(false);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error connecting to the support backend. Please check your connection." },
      ]);
    } finally {
      setIsLoading(false);
      setIsThinking(false);
    }
  };

  // Simple formatter for lists & bold text in messages
  const formatMessageContent = (text: string) => {
    return text.split('\n').map((line, idx) => {
      let content = line;
      // Bold text formatting **text**
      const boldRegex = /\*\*(.*?)\*\*/g;
      const parts = [];
      let lastIndex = 0;
      let match;
      
      while ((match = boldRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push(line.substring(lastIndex, match.index));
        }
        parts.push(<strong key={match.index} className="text-white font-semibold">{match[1]}</strong>);
        lastIndex = boldRegex.lastIndex;
      }
      if (lastIndex < line.length) {
        parts.push(line.substring(lastIndex));
      }

      const formattedLine = parts.length > 0 ? parts : content;

      if (line.startsWith('- ') || line.startsWith('* ')) {
        return <li key={idx} className="ml-4 list-disc text-white/90 my-1">{formattedLine.toString().substring(2)}</li>;
      }
      if (/^\d+\.\s/.test(line)) {
        return <li key={idx} className="ml-4 list-decimal text-white/90 my-1">{line.replace(/^\d+\.\s/, '')}</li>;
      }
      return <p key={idx} className="my-1 min-h-[1rem]">{formattedLine}</p>;
    });
  };

  return (
    <div className="flex h-screen bg-mesh-glow bg-black text-white font-sans overflow-hidden">
      {/* Main Chat Layout */}
      <div className="flex-1 flex flex-col h-full relative z-10 border-r border-white/5">
        
        {/* Top Header */}
        <header className="flex items-center justify-between px-8 py-4 border-b border-white/10 bg-black/40 backdrop-blur-xl z-20">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Bot className="text-white w-5 h-5" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-wide text-gradient">Acme Support Copilot</h1>
              <div className="text-[10px] text-white/40 flex items-center gap-3 mt-0.5">
                <div className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${serverHealthy ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
                  <span>API: {serverHealthy ? "Online" : "Offline"}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${dbConnected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
                  <span>Oracle DB: {dbConnected ? "Connected" : "Disconnected"}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Tab Switches */}
            <div className="bg-white/5 border border-white/10 rounded-lg p-0.5 flex items-center gap-0.5">
              <button
                onClick={() => setActiveTab("chat")}
                className={`text-xs px-3 py-1.5 rounded-md transition-all font-medium cursor-pointer ${
                  activeTab === "chat"
                    ? "bg-white/10 text-white shadow-sm"
                    : "text-white/60 hover:text-white"
                }`}
              >
                Chat Interface
              </button>
              <button
                onClick={() => setActiveTab("console")}
                className={`text-xs px-3 py-1.5 rounded-md transition-all font-medium cursor-pointer ${
                  activeTab === "console"
                    ? "bg-white/10 text-white shadow-sm"
                    : "text-white/60 hover:text-white"
                }`}
              >
                Control Console
              </button>
            </div>

            <select
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg text-xs px-3 py-1.5 focus:outline-none focus:border-indigo-500 text-white/80 cursor-pointer transition-all hover:bg-white/10"
            >
              <option value="general" className="bg-zinc-950 text-white">General Scope</option>
              <option value="sales" className="bg-zinc-950 text-white">Sales & Shipping</option>
              <option value="technical" className="bg-zinc-950 text-white">Technical Support</option>
              <option value="billing" className="bg-zinc-950 text-white">Billing & Returns</option>
            </select>

            <button
              onClick={() => setShowInsights(!showInsights)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                showInsights 
                  ? "bg-white/10 border-white/20 text-white" 
                  : "bg-transparent border-white/10 text-white/60 hover:text-white"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              AI Insights
            </button>
          </div>
        </header>

        {activeTab === "chat" ? (
          <>
            {/* Messages & Conversation Area */}
            <div 
              ref={scrollRef}
              className="flex-1 overflow-y-auto px-6 py-8 space-y-6 scrollbar-thin"
            >
              <AnimatePresence initial={false}>
                {messages.length === 0 && (
                  <motion.div 
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-center justify-center min-h-[75%] max-w-2xl mx-auto text-center px-4 space-y-8"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-indigo-500/10 blur-3xl rounded-full" />
                      <div className="relative w-16 h-16 bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center mx-auto shadow-2xl backdrop-blur-xl">
                        <Sparkles className="w-7 h-7 text-indigo-400 animate-pulse" />
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h2 className="text-2xl font-semibold tracking-tight text-gradient">Welcome to Acme Support</h2>
                      <p className="text-sm text-white/50 max-w-md mx-auto leading-relaxed">
                        Ask me any question about the **SuperWidget 3000**, shipping specifications, or return procedures. I search our corporate knowledge base in real-time.
                      </p>
                    </div>

                    {/* Suggested Prompts Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full pt-4">
                      {SUGGESTED_PROMPTS.map((item, idx) => (
                        <motion.div
                          key={idx}
                          whileHover={{ y: -4, scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => handleSend(item.prompt)}
                          className="cursor-pointer p-4 rounded-xl bg-white/5 border border-white/10 text-left hover:bg-white/10 hover:border-white/20 transition-all shadow-xl group backdrop-blur-md"
                        >
                          <div className="flex justify-between items-start mb-2">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">{item.category}</span>
                            <ArrowRight className="w-3.5 h-3.5 text-white/20 group-hover:text-white transition-colors" />
                          </div>
                          <h3 className="text-xs font-semibold text-white/90 group-hover:text-white mb-1">{item.title}</h3>
                          <p className="text-[11px] text-white/40 leading-snug">{item.desc}</p>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {messages.map((msg, idx) => {
                  if (msg.role === "assistant" && msg.content === "") return null;
                  return (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ type: "spring", stiffness: 300, damping: 25 }}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} max-w-4xl mx-auto`}
                    >
                      <div className={`flex gap-3 max-w-[85%] ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                        
                        {/* Avatar */}
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-lg ${
                          msg.role === "user" 
                            ? "bg-white/10 border border-white/10 text-white" 
                            : "bg-gradient-to-tr from-violet-600 to-indigo-600 text-white"
                        }`}>
                          {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                        </div>

                        {/* Message Bubble */}
                        <div className="flex flex-col gap-1">
                          <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                            msg.role === "user" 
                              ? "bg-white text-black shadow-xl rounded-tr-none" 
                              : "bg-white/5 border border-white/10 text-white/90 rounded-tl-none shadow-2xl backdrop-blur-md"
                          }`}>
                            {msg.content && <div className="space-y-1">{formatMessageContent(msg.content)}</div>}
                            {msg.image_url && (
                              <div className="mt-2.5 overflow-hidden rounded-xl border border-white/10">
                                <img 
                                  src={msg.image_url} 
                                  alt="Uploaded issue screenshot" 
                                  className="max-h-60 w-auto object-cover hover:scale-105 transition-all"
                                />
                              </div>
                            )}
                          </div>
                          <span className="text-[10px] text-white/30 tracking-wider uppercase font-semibold px-2">
                            {msg.role === "user" ? "You" : "Acme Copilot"}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>

              {isThinking && (
                <div className="flex justify-start max-w-4xl mx-auto">
                  <div className="flex gap-3 max-w-[80%]">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-white/5 border border-white/10 px-4 py-3 rounded-2xl rounded-tl-none flex items-center gap-2.5 backdrop-blur-md shadow-xl">
                      <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                      <span className="text-xs text-white/40 font-medium italic">Searching databases & thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Inputs Area */}
            <footer className="p-6 border-t border-white/10 bg-black/40 backdrop-blur-xl z-20">
              <div className="max-w-4xl mx-auto relative">
                
                {/* Image Preview Floating Banner */}
                {image && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute -top-24 left-0 p-2 bg-white/10 rounded-xl border border-white/20 backdrop-blur-md shadow-2xl flex items-center gap-2"
                  >
                    <img src={image} alt="Upload Preview" className="h-16 w-16 object-cover rounded-lg" />
                    <button 
                      onClick={() => setImage(null)}
                      className="bg-black/80 hover:bg-black rounded-full p-1.5 shadow-lg transition-colors border border-white/10"
                    >
                      <X className="w-3 h-3 text-white" />
                    </button>
                  </motion.div>
                )}

                {/* Input Box */}
                <div className="relative flex items-center gap-2 bg-white/5 border border-white/10 rounded-2xl px-4 py-2 focus-within:border-indigo-500/50 focus-within:bg-white/10 transition-all shadow-2xl backdrop-blur-md">
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 text-white/40 hover:text-white transition-colors"
                    title="Upload screenshot of your issue"
                  >
                    <ImageIcon className="w-5 h-5" />
                  </button>
                  <input 
                    type="file" 
                    hidden 
                    ref={fileInputRef} 
                    accept="image/*"
                    onChange={handleImageUpload}
                  />
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Ask about SuperWidget specifications, free shipping limit..."
                    className="flex-1 bg-transparent border-none focus:outline-none focus:ring-0 text-sm py-2.5 placeholder:text-white/25"
                  />
                  <button 
                    onClick={() => handleSend()}
                    disabled={isLoading || (!input.trim() && !image)}
                    className="p-2.5 bg-white hover:bg-white/90 text-black rounded-xl disabled:opacity-30 disabled:hover:bg-white transition-all hover:scale-105 active:scale-95 glow-btn cursor-pointer shadow-lg shadow-white/5 flex items-center justify-center shrink-0"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </footer>
          </>
        ) : (
          /* System Settings & Control Console */
          <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-thin bg-neutral-950/20">
            
            {/* Console Title Banner */}
            <div className="flex items-center gap-3 border-b border-white/5 pb-4">
              <Settings className="w-5 h-5 text-indigo-400" />
              <div>
                <h2 className="text-base font-semibold text-white">System Settings & RAG Control Console</h2>
                <p className="text-xs text-white/50">Manage data sources, connection parameters, agent profiles, and table routing prompts.</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Database Connection Card */}
              <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md space-y-4">
                <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                  <Database className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">Oracle Database Connection</h3>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2 space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Host / IP</label>
                    <input 
                      type="text" 
                      value={dbConfig.host} 
                      onChange={(e) => setDbConfig({...dbConfig, host: e.target.value})}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>
                  
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Port</label>
                    <input 
                      type="number" 
                      value={dbConfig.port} 
                      onChange={(e) => setDbConfig({...dbConfig, port: parseInt(e.target.value) || 1521})}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Service Name / SID</label>
                    <input 
                      type="text" 
                      value={dbConfig.service_name} 
                      onChange={(e) => setDbConfig({...dbConfig, service_name: e.target.value})}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Username</label>
                    <input 
                      type="text" 
                      value={dbConfig.user} 
                      onChange={(e) => setDbConfig({...dbConfig, user: e.target.value})}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Password</label>
                    <input 
                      type="password" 
                      value={dbConfig.password} 
                      onChange={(e) => setDbConfig({...dbConfig, password: e.target.value})}
                      placeholder="••••••••"
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>
                </div>

                <div className="flex justify-between items-center pt-2">
                  <div className="flex items-center gap-1.5 text-[10px]">
                    <span className={`w-1.5 h-1.5 rounded-full ${dbConnected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
                    <span className="text-white/40">{dbConnected ? "Database connected" : "Database disconnected"}</span>
                  </div>
                  <button 
                    onClick={handleSaveDbConfig}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-indigo-500/10 active:scale-95 transition-all cursor-pointer"
                  >
                    Save connection
                  </button>
                </div>
              </div>

              {/* Model Configurations Card */}
              <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md space-y-4">
                <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                  <Bot className="w-4 h-4 text-violet-400" />
                  <h3 className="text-sm font-semibold text-white">AI Engine & Model Configuration</h3>
                </div>

                <div className="space-y-3.5">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">LLM model</label>
                      <input 
                        type="text" 
                        value={modelConfig.MODEL_NAME} 
                        onChange={(e) => setModelConfig({...modelConfig, MODEL_NAME: e.target.value})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Embedding Model</label>
                      <input 
                        type="text" 
                        value={modelConfig.EMBEDDING_MODEL} 
                        onChange={(e) => setModelConfig({...modelConfig, EMBEDDING_MODEL: e.target.value})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                      />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">LLM API Base URL</label>
                    <input 
                      type="text" 
                      value={modelConfig.LOCAL_LLM_BASE_URL} 
                      onChange={(e) => setModelConfig({...modelConfig, LOCAL_LLM_BASE_URL: e.target.value})}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Embedding Base URL</label>
                      <input 
                        type="text" 
                        value={modelConfig.LOCAL_EMBEDDING_BASE_URL} 
                        onChange={(e) => setModelConfig({...modelConfig, LOCAL_EMBEDDING_BASE_URL: e.target.value})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all text-white"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Vector Store Backend</label>
                      <select 
                        value={modelConfig.VECTOR_DB_TYPE} 
                        onChange={(e) => setModelConfig({...modelConfig, VECTOR_DB_TYPE: e.target.value})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 text-white/80 cursor-pointer transition-all hover:bg-white/10"
                      >
                        <option value="qdrant" className="bg-zinc-950 text-white">Qdrant (Clustered)</option>
                        <option value="faiss" className="bg-zinc-950 text-white">FAISS (Flat Index)</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end pt-1">
                  <button 
                    onClick={handleSaveModelConfig}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-indigo-500/10 active:scale-95 transition-all cursor-pointer"
                  >
                    Save settings
                  </button>
                </div>
              </div>
            </div>

            {/* RAG Knowledge Ingestion panel */}
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md space-y-4">
              <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                <UploadCloud className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-white">RAG Knowledge Ingestion Panel</h3>
              </div>

              <form onSubmit={handleRagFileUpload} className="space-y-4">
                <div className="border border-dashed border-white/10 rounded-2xl p-8 text-center bg-white/[0.02] hover:bg-white/[0.04] transition-all relative group">
                  <input 
                    type="file" 
                    id="rag-file-input"
                    accept=".pdf"
                    onChange={(e) => setRagFile(e.target.files?.[0] || null)}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="flex flex-col items-center space-y-2">
                    <div className="p-3 bg-white/5 rounded-xl border border-white/10 text-white/50 group-hover:text-emerald-400 group-hover:border-emerald-500/30 transition-all shadow-md">
                      <FileText className="w-6 h-6" />
                    </div>
                    {ragFile ? (
                      <div className="space-y-1">
                        <p className="text-xs font-semibold text-white">{ragFile.name}</p>
                        <p className="text-[10px] text-white/40">{(ragFile.size / 1024 / 1024).toFixed(2)} MB • PDF File</p>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <p className="text-xs font-semibold text-white/70">Click to browse or drag PDF here</p>
                        <p className="text-[10px] text-white/40">PDF documents undergo layout-aware Docling hierarchical extraction</p>
                      </div>
                    )}
                  </div>
                </div>

                {uploadMessage && (
                  <div className={`p-3 rounded-xl flex items-start gap-2.5 text-xs ${
                    uploadMessage.toLowerCase().includes("fail") 
                      ? "bg-rose-500/10 border border-rose-500/20 text-rose-300"
                      : "bg-emerald-500/10 border border-emerald-500/20 text-emerald-300"
                  }`}>
                    {isUploading ? (
                      <Loader2 className="w-4 h-4 animate-spin text-indigo-400 mt-0.5 shrink-0" />
                    ) : uploadMessage.toLowerCase().includes("fail") ? (
                      <AlertCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                    )}
                    <span className="leading-relaxed">{uploadMessage}</span>
                  </div>
                )}

                <div className="flex justify-end">
                  <button 
                    type="submit"
                    disabled={isUploading || !ragFile}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-xl text-xs font-semibold shadow-lg shadow-emerald-500/10 active:scale-95 transition-all cursor-pointer flex items-center gap-1.5"
                  >
                    {isUploading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                    <span>Upload & Index Document</span>
                  </button>
                </div>
              </form>
            </div>

            {/* Oracle Tables Context / Metadata Panel */}
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">SQL Table Semantic Directory & Metadata Router</h3>
                </div>
                <button
                  onClick={fetchConfigAndTables}
                  className="p-1.5 hover:bg-white/5 border border-white/10 rounded-lg text-white/50 hover:text-white transition-all flex items-center gap-1.5 text-[10px] font-semibold cursor-pointer"
                  title="Reload table schemas from connection"
                >
                  <RefreshCw className="w-3 h-3" />
                  Sync Schemas
                </button>
              </div>

              <p className="text-xs text-white/50 max-w-3xl leading-relaxed">
                Table names in operational systems can be highly confusing (e.g. obscure database names). 
                Provide a natural language summary explaining precisely what columns or semantic entities are contained inside each table. 
                Our backend router uses these descriptions to decide which tables to select for the SQL execution agent.
              </p>

              {oracleTables.length === 0 ? (
                <div className="p-8 border border-white/5 rounded-2xl text-center bg-white/[0.01]">
                  <AlertCircle className="w-8 h-8 text-white/20 mx-auto mb-2" />
                  <p className="text-xs font-medium text-white/40">No user tables found or database connection is offline.</p>
                  <p className="text-[10px] text-white/30 mt-1">Configure the Oracle connection above and click "Sync Schemas" to load tables.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {oracleTables.map((tbl) => {
                    const tblLower = tbl.toLowerCase();
                    return (
                      <div key={tbl} className="p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/10 hover:bg-white/[0.03] transition-all space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-mono font-bold tracking-wide text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-0.5 rounded-md uppercase">
                            {tbl}
                          </span>
                          <span className="text-[10px] text-white/30">
                            {tableMetadata[tblLower] ? "Metadata Active" : "Needs Description"}
                          </span>
                        </div>
                        
                        <div className="space-y-1">
                          <label className="text-[9px] uppercase tracking-wider text-white/40 font-bold">Semantic Description</label>
                          <textarea
                            value={editingMetadata[tbl] || ""}
                            onChange={(e) => setEditingMetadata({
                              ...editingMetadata,
                              [tbl]: e.target.value
                            })}
                            placeholder="Describe what fields or business objects are in this table to help the SQL router..."
                            rows={3}
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 focus:bg-white/10 transition-all placeholder:text-white/20 resize-none leading-relaxed text-white"
                          />
                        </div>

                        <div className="flex justify-end">
                          <button
                            onClick={() => handleSaveTableMetadata(tbl)}
                            className="px-3 py-1.5 bg-white/5 border border-white/10 hover:bg-indigo-600 hover:border-indigo-600 text-white rounded-lg text-[10px] font-semibold active:scale-95 transition-all cursor-pointer"
                          >
                            Update Metadata
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

          </div>
        )}
      </div>

      {/* Right AI & RAG Insights Drawer */}
      <AnimatePresence>
        {showInsights && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 340, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 right-0 lg:relative flex flex-col h-full bg-neutral-950/95 lg:bg-black/60 backdrop-blur-2xl border-l border-white/10 z-50 lg:z-10 overflow-hidden"
          >
            <div className="p-6 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-indigo-400" />
                <h2 className="text-sm font-semibold text-gradient">RAG & Model Insights</h2>
              </div>
              <button 
                onClick={() => setShowInsights(false)}
                className="p-1 hover:bg-white/5 rounded-lg text-white/40 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              {/* Active LLM and Embeddings info */}
              <div className="space-y-4">
                <h3 className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Active Models</h3>
                
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-white/40">LLM Model</span>
                    <span className="bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded text-[10px] font-mono border border-indigo-500/20">{backendConfig.llm_model}</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-white/40">Embedding Model</span>
                    <span className="bg-violet-500/10 text-violet-300 px-2 py-0.5 rounded text-[10px] font-mono border border-violet-500/20">{backendConfig.embedding_model}</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-white/40">Vector Database</span>
                    <span className="text-white/80 font-mono text-[10px] uppercase">{backendConfig.vector_db}</span>
                  </div>
                </div>
              </div>

              {/* RAG Parameters */}
              <div className="space-y-4">
                <h3 className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Database State</h3>
                
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-3.5">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/40">KB Collection</span>
                      <span className="text-white/90 text-xs font-mono">customer_support_kb</span>
                    </div>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/40">Search Strategy</span>
                      <span className="text-white/90 text-xs font-medium">Similarity (k=3)</span>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/40">Source Count</span>
                      <span className="text-emerald-400 font-bold text-xs">1 Active File</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Verified Sources */}
              <div className="space-y-4">
                <h3 className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Active Knowledge Base</h3>
                <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 flex items-start gap-3">
                  <HelpCircle className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <h4 className="text-xs font-semibold text-emerald-300">company_info.txt</h4>
                    <p className="text-[10px] text-white/40 leading-relaxed">
                      Contains official returns policies, SuperWidget troubleshooting protocols, and free shipping requirements.
                    </p>
                  </div>
                </div>
              </div>

            </div>

            <div className="p-6 border-t border-white/10 bg-black/40">
              <div className="flex items-center gap-2 text-xs text-white/40">
                <Shield className="w-3.5 h-3.5" />
                Security Layer Active
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
