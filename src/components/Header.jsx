import ThemeToggle from "./ThemeToggle";
import keanLogo from "../assets/Kean_Seal-2018-white.gif";

function Header() {
  return (
    <header className="header">
      <div className="logo">
        <img
          src={keanLogo}
          alt="Kean University Logo"
          className="kean-logo"
        />
        <h2>KeanGlobal</h2>
      </div>

      <ThemeToggle />
    </header>
  );
}

export default Header;
