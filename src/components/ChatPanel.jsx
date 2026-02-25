import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const CHAT_STORAGE_KEY = "keanglobal_chat_messages";


function ChatPanel({ setShowMap, setRouteRequest }) {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem(CHAT_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const abortControllerRef = useRef(null);
  const pendingUserMessageRef = useRef("");

  useEffect(() => {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  async function sendMessage() {
    const userMessage = input.trim();
    if (!userMessage || loading) return;

    setMessages(prev => [...prev, { text: userMessage, sender: "user" }]);
    setInput("");
    setLoading(true);
    pendingUserMessageRef.current = userMessage;
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
        signal: abortControllerRef.current.signal
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data?.detail || "Request failed.");
      }

      const answerText = data.answer || data.reply || "No response from backend.";
      setMessages(prev => [...prev, { text: answerText, sender: "bot" }]);
      setShowMap(data.intent === "location");
      if (data.intent === "location") {
        setRouteRequest({
          destinationId: data.destination_id || null,
          useCurrentLocation: Boolean(data.use_current_location)
        });
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }

      const errorText =
        error instanceof Error
          ? error.message
          : "Backend unavailable. Start FastAPI and Ollama.";

      setMessages(prev => [
        ...prev,
        {
          text: `Error: ${errorText}`,
          sender: "bot"
        }
      ]);
      setShowMap(false);
      setRouteRequest(null);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
      pendingUserMessageRef.current = "";
    }
  }

  function cancelSend() {
    if (!loading) return;

    const pendingMessage = pendingUserMessageRef.current;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    setMessages(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.sender === "user" && last.text === pendingMessage) {
        return prev.slice(0, -1);
      }
      return prev;
    });
    setInput(pendingMessage);
    setLoading(false);
    abortControllerRef.current = null;
    pendingUserMessageRef.current = "";
  }

  function clearHistory() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setMessages([]);
    setInput("");
    setLoading(false);
    setShowMap(false);
    setRouteRequest(null);
    localStorage.removeItem(CHAT_STORAGE_KEY);
    abortControllerRef.current = null;
    pendingUserMessageRef.current = "";
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  }

  return (
    <div className="panel chat-panel">
      <div className="map-header">
        <h3 className="panel-title">Campus Concierge</h3>
        <button className="btn-secondary" type="button" onClick={clearHistory}>
          Clear Chat
        </button>
      </div>

      <div className="chat-box">
        {loading && <div className="spinner"></div>}

        {messages.map((m, i) => (
          <div key={i} className={m.sender === "user" ? "msg-user" : "msg-bot"}>
            {m.text}
          </div>
        ))}
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Kean University..."
        />

        <button className="btn-primary" onClick={sendMessage} disabled={loading}>
          Send
        </button>
        {loading && (
          <button className="btn-secondary" type="button" onClick={cancelSend}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

export default ChatPanel;
