import { useNavigate } from "react-router-dom";

export default function HomePage() {
  const navigate = useNavigate();

  return (
        <div className="home">
        <section className="hero">
            <div className="hero-full">
                <div className="hero-video-wrap">
                <video
                    className="hero-video"
                    src="/media/hero.mp4"
                    autoPlay
                    loop
                    muted
                    playsInline
                    preload="auto"
                />
                <div className="hero-overlay">
                    <div className="hero-actions">
                    <button className="hero-btn" onClick={() => navigate("/programs")}>
                        Majors &amp; Degree Programs
                    </button>

                    <button className="hero-btn" onClick={() => navigate("/chat")}>
                        Get Direction
                    </button>
                    </div>
                </div>
                </div>
            </div>
        </section>

        <div className="panel" style={{ margin: 20, borderRadius: 12 }}>
            <h2 className="panel-title">KeanGlobal Home</h2>
            <p style={{ color: "var(--text-secondary)" }}>
                Welcome. Choose a feature from the navigation bar, or start with Chat.
            </p>

            <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
                <button className="btn-secondary" onClick={() => navigate("/chat")}>
                Go to Chat
                </button>
            </div>
        </div>
    </div>
  );
}
