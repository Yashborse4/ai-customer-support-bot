
"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Image as ImageIcon, User, Bot, Loader2, X } from "lucide-react";
import axios from "axios";

interface Message {
  role: "user" | "assistant";
  content: string;
  image_url?: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
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

  const handleSend = async () => {
    if (!input.trim() && !image) return;

    const newMessage: Message = {
      role: "user",
      content: input,
      image_url: image || undefined,
    };

    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setImage(null);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        messages: [...messages, newMessage].map((m) => ({
          role: m.role,
          content: m.content,
          image_url: m.image_url,
        })),
      });

      const botMessage: Message = {
        role: "assistant",
        content: response.data.response,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error. Please check your connection." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-black text-white font-sans overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-4 border-b border-white/10 bg-black/50 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
            <Bot className="text-black w-5 h-5" />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Acme Corp Support</h1>
        </div>
        <div className="text-xs text-white/40 flex items-center gap-2">
          <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
          System Online
        </div>
      </header>

      {/* Chat Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-8 space-y-6 scrollbar-hide"
      >
        <AnimatePresence initial={false}>
          {messages.length === 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col items-center justify-center h-full text-center space-y-4"
            >
              <Bot className="w-12 h-12 text-white/20" />
              <div className="space-y-2">
                <h2 className="text-xl font-medium text-white/80">How can I help you?</h2>
                <p className="text-sm text-white/40 max-w-sm">
                  Ask me about products, shipping, or upload a screenshot of an issue.
                </p>
              </div>
            </motion.div>
          )}

          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: msg.role === "user" ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`max-w-[80%] flex flex-col gap-2 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`px-4 py-3 rounded-2xl ${
                  msg.role === "user" 
                    ? "bg-white text-black" 
                    : "bg-white/5 border border-white/10 text-white/90"
                }`}>
                  {msg.content && <p className="text-sm leading-relaxed">{msg.content}</p>}
                  {msg.image_url && (
                    <img 
                      src={msg.image_url} 
                      alt="uploaded" 
                      className="mt-2 rounded-lg max-w-full h-auto border border-white/10"
                    />
                  )}
                </div>
                <div className="flex items-center gap-2 px-1 text-[10px] uppercase tracking-widest text-white/30 font-medium">
                  {msg.role === "user" ? (
                    <>You <User className="w-3 h-3" /></>
                  ) : (
                    <><Bot className="w-3 h-3" /> Bot</>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white/5 border border-white/10 px-4 py-3 rounded-2xl flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-white/40" />
              <span className="text-xs text-white/40 font-medium italic">Thinking...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <footer className="p-6 border-t border-white/10 bg-black/50 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto relative">
          {image && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute -top-24 left-0 p-2 bg-white/10 rounded-xl border border-white/20 backdrop-blur-md"
            >
              <img src={image} alt="preview" className="h-16 w-16 object-cover rounded-lg" />
              <button 
                onClick={() => setImage(null)}
                className="absolute -top-2 -right-2 bg-red-500 rounded-full p-1 shadow-lg"
              >
                <X className="w-3 h-3 text-white" />
              </button>
            </motion.div>
          )}

          <div className="relative flex items-center gap-2 bg-white/5 border border-white/10 rounded-2xl px-4 py-2 focus-within:border-white/30 transition-all shadow-2xl">
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-white/40 hover:text-white transition-colors"
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
              placeholder="Type your message..."
              className="flex-1 bg-transparent border-none focus:ring-0 text-sm py-2 placeholder:text-white/20"
            />
            <button 
              onClick={handleSend}
              disabled={isLoading || (!input.trim() && !image)}
              className="p-2 bg-white text-black rounded-xl disabled:opacity-30 disabled:grayscale transition-all hover:scale-105 active:scale-95"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
