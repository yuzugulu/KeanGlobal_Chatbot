import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function ChatPanel({ setShowMap, setRouteRequest }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    const userMessage = input.trim();
    if (!userMessage || loading) return;

    setMessages(prev => [...prev, { text: userMessage, sender: "user" }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Request failed.");
      }

      setMessages(prev => [...prev, { text: data.answer, sender: "bot" }]);
      setShowMap(data.intent === "location");
      if (data.intent === "location") {
        setRouteRequest({
          destinationId: data.destination_id || null,
          useCurrentLocation: Boolean(data.use_current_location)
        });
      }
    } catch (error) {
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
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  }

  return (
    <div className="panel chat-panel">
      <h3 className="panel-title">Campus Concierge</h3>

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
      </div>
    </div>
  );
}

export default ChatPanel;
