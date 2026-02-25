import ThemeToggle from "./ThemeToggle";
import keanLogo from "../assets/Kean_Seal-2018-white.gif";
import { NavLink } from "react-router-dom";

function Header() {
  return (
    <header className="header">
      <div className="logo">
        <img src={keanLogo} alt="Kean University Logo" className="kean-logo" />
        <h2>KeanGlobal</h2>

        <nav className="nav">
          <NavLink to="/" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Home
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Map/Chat
          </NavLink>
          <NavLink to="/programs" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Programs
          </NavLink>
        </nav>
      </div>

      <ThemeToggle />
    </header>
  );
}

export default Header;
