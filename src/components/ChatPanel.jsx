import { useState } from "react";

function ChatPanel({ setShowMap }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Temporary mock AI logic
  function mockAIResponse(userText) {
    const text = userText.toLowerCase();

    if (
      text.includes("where") ||
      text.includes("location") ||
      text.includes("building")
    ) {
      return {
        answer: "Kean Hall is highlighted on the map.",
        intent: "location",
        building: "Kean Hall"
      };
    }

    return {
      answer: "KeanGlobal prototype running.",
      intent: "policy"
    };
  }

  function sendMessage() {
    if (!input) return;

    const userMessage = input;

    // Show user's message
    setMessages(prev => [
      ...prev,
      { text: userMessage, sender: "user" }
    ]);

    setInput("");
    setLoading(true); // Spinner ON

    setTimeout(() => {
      const ai = mockAIResponse(userMessage);

      // Show bot response
      setMessages(prev => [
        ...prev,
        { text: ai.answer, sender: "bot" }
      ]);

      // Show map only if location intent
      if (ai.intent === "location") {
        setShowMap(true);
      } else {
        setShowMap(false);
      }

      setLoading(false); // Spinner OFF
    }, 800);
  }

  return (
    <div className="panel chat-panel">
      <h3 className="panel-title">Campus Concierge</h3>

      <div className="chat-box">
        {loading && <div className="spinner"></div>}

        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === "user" ? "msg-user" : "msg-bot"}
          >
            {m.text}
          </div>
        ))}
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about Kean University..."
        />

        <button className="btn-primary" onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;
