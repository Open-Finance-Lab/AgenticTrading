import { Navbar } from "../components/home/Navbar";
import { MarketTicker } from "../components/home/MarketTicker";
import { Hero } from "../components/home/Hero";
import { WhyCare } from "../components/home/WhyCare";
import { Talk } from "../components/home/Talk";
import { Test } from "../components/home/Test";
import { Race } from "../components/home/Race";
import { FooterCTA } from "../components/home/FooterCTA";
import { LeaderboardProvider } from "@/lib/useLeaderboard";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <div className="landing-chrome">
        <Navbar />
        <MarketTicker />
      </div>
      <LeaderboardProvider>
        <main>
          <Hero />
          <WhyCare />
          <Talk />
          <Test />
          <Race />
        </main>
      </LeaderboardProvider>
      <FooterCTA />
    </div>
  );
}
