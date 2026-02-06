import { useState } from "react";

function ThemeToggle() {
  const [dark, setDark] = useState(false);

  function toggleTheme() {
    setDark(!dark);
    document.body.classList.toggle("dark");
  }

  return (
    <button className="theme-btn" onClick={toggleTheme}>
      {dark ? "☀️ Light" : "🌙 Dark"}
    </button>
  );
}

export default ThemeToggle;
