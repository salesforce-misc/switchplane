import Hero from "./components/Hero";
import Comparison from "./components/Comparison";
import Quickstart from "./components/Quickstart";
import TerminalRecording from "./components/TerminalRecording";
import LocalFirst from "./components/LocalFirst";
import Pillars from "./components/Pillars";
import Features from "./components/Features";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="min-h-screen">
      <Hero />
      <Comparison />
      <Quickstart />
      <TerminalRecording />
      <LocalFirst />
      <Pillars />
      <Features />
      <Footer />
    </div>
  );
}

export default App;
