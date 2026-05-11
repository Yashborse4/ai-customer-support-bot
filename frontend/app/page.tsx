"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Image as ImageIcon, User, Bot, Loader2, X, Sparkles, Shield, Database, ChevronRight, HelpCircle, ArrowRight } from "lucide-react";
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
  const [showInsights, setShowInsights] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setImage(null);
    setIsLoading(true);

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
          messages: [...messages, newMessage].map((m) => ({
            role: m.role,
            content: m.content,
            image_url: m.image_url,
          })),
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
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error connecting to the support backend. Please check your connection." },
      ]);
    } finally {
      setIsLoading(false);
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
              <div className="text-[10px] text-white/40 flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                Active & Connected
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowInsights(!showInsights)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${
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

            {messages.map((msg, idx) => (
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
            ))}
          </AnimatePresence>

          {isLoading && (
            <div className="flex justify-start max-w-4xl mx-auto">
              <div className="flex gap-3 max-w-[80%]">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-white/5 border border-white/10 px-4 py-3 rounded-2xl rounded-tl-none flex items-center gap-2.5 backdrop-blur-md shadow-xl">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                  <span className="text-xs text-white/40 font-medium italic">Searching context & thinking...</span>
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
      </div>

      {/* Right AI & RAG Insights Drawer */}
      <AnimatePresence>
        {showInsights && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 340, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="hidden lg:flex flex-col h-full bg-black/60 backdrop-blur-2xl border-l border-white/10 z-10 overflow-hidden"
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
                    <span className="bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded text-[10px] font-mono border border-indigo-500/20">GPT-4o</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-white/40">Embedding Model</span>
                    <span className="bg-violet-500/10 text-violet-300 px-2 py-0.5 rounded text-[10px] font-mono border border-violet-500/20">Text-Embedding-3</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-white/40">Vector Database</span>
                    <span className="text-white/80 font-mono text-[10px]">ChromaDB</span>
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
