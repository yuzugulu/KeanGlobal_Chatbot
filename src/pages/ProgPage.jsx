import { useNavigate } from "react-router-dom";
import { useMemo, useState, useEffect } from "react";

export default function ProgPage() {
  const navigate = useNavigate();
  
  // State for fetched data, search, and filters
  const [programsList, setProgramsList] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [q, setQ] = useState("");
  const [level, setLevel] = useState("All");

  // Fetch all programs on mount
  useEffect(() => {
    const fetchPrograms = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/programs');
        if (!response.ok) throw new Error("Network response was not ok");
        
        const data = await response.json();
        
        // Transform JSON object into an array for rendering
        const formattedPrograms = Object.entries(data.programs).map(([key, pData]) => {
          const isGrad = pData.metadata.full_name.match(/(M\.S\.|M\.A\.|Ph\.D\.|Post)/i);
          
          return {
            id: key,
            name: pData.metadata.full_name,
            level: isGrad ? "Graduate" : "Undergraduate",
            area: "Kean Program",
            tags: [
              pData.metadata.coordinator ? "Coordinator Info" : "",
              Object.keys(pData.curriculum?.core_courses || {}).length > 0 ? "Has Courses" : ""
            ].filter(Boolean),
            ...pData
          };
        });

        setProgramsList(formattedPrograms);
      } catch (error) {
        console.error("Error fetching programs:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPrograms();
  }, []);

  // Filter logic
  const filtered = useMemo(() => {
    const keyword = q.trim().toLowerCase();
    return programsList.filter((p) => {
      const matchQ =
        !keyword ||
        p.name.toLowerCase().includes(keyword) ||
        p.tags.some((t) => t.toLowerCase().includes(keyword)) ||
        p.area.toLowerCase().includes(keyword);

      const matchLevel = level === "All" || p.level === level;
      return matchQ && matchLevel;
    });
  }, [q, level, programsList]);

  if (isLoading) {
    return <div className="p-8 text-center text-gray-500">Loading programs...</div>;
  }

  return (
    <div className="porgrams">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-full">
          <div className="hero-video-wrap">
            <video
              className="hero-video"
              src="/media/hero.mp4"
              autoPlay loop muted playsInline preload="auto"
            />
            <div className="hero-overlay">
              <h1 className="hero-title">Majors &amp; Degree Programs</h1>
              <div className="hero-actions">
                <button className="hero-btn" onClick={() => navigate("/programs")}>
                  Explore Programs
                </button>
                <button className="hero-btn" onClick={() => navigate("/chat")}>
                  Help Me Choose
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Header & Controls */}
      <div className="programs-header">
        <h1 className="programs-title">Majors & Degree Programs</h1>
        <p className="programs-subtitle">
          Browse {programsList.length} programs. Use search to filter by name or keywords.
        </p>

        <div className="programs-controls">
          <input
            className="programs-search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search (e.g., Computer Science, Data)..."
          />
          <select
            className="programs-select"
            value={level}
            onChange={(e) => setLevel(e.target.value)}
          >
            <option value="All">All levels</option>
            <option value="Undergraduate">Undergraduate</option>
            <option value="Graduate">Graduate</option>
          </select>
        </div>
      </div>

      {/* Program Grid */}
      <div className="programs-grid">
        {filtered.map((p) => (
          <div key={p.id} className="program-card">
            <div className="program-card-top">
              <div className="program-name">{p.name}</div>
              <div className="program-meta">
                <span className="pill">{p.level}</span>
              </div>
            </div>

            {p.metadata.note && (
              <div style={{ color: "red", fontSize: "0.8rem", marginBottom: "8px" }}>
                * {p.metadata.note}
              </div>
            )}

            <div className="program-tags">
              {p.tags.map((t) => (
                <span key={t} className="tag">{t}</span>
              ))}
            </div>

            <div className="program-actions">
              <button
                className="btn-secondary"
                onClick={() => navigate(`/program/${p.id}`)}
              >
                View Details
              </button>
              <button
                className="btn-secondary"
                onClick={() => {
                  const email = p.metadata.contact.email;
                  if (email) {
                    window.location.href = `mailto:${email}?subject=Inquiry about ${p.name}`;
                  } else {
                    alert("No contact email available for this program.");
                  }
                }}
              >
                Contact
              </button>
            </div>
          </div>
        ))}

        {filtered.length === 0 && !isLoading && (
          <div className="program-empty">
            No programs found. Try another keyword.
          </div>
        )}
      </div>
    </div>
  );
}