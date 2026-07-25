import { Button } from "@/components/ui/button";
import { SIGNUP_HREF } from "@/lib/cta";

export function FooterCTA() {
  return (
    <footer className="py-24 relative overflow-hidden text-center border-t border-border">
      <div className="absolute inset-0 bg-grid-pattern opacity-10 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
      <div className="container mx-auto px-6 relative z-10">
        <p className="text-sm font-mono uppercase tracking-widest text-muted-foreground mb-4">
          Talk → Test → Race
        </p>
        <h2 className="text-4xl md:text-5xl font-bold tracking-tighter mb-6">Ready to run your first idea?</h2>
        <p className="text-xl text-muted-foreground mb-10 max-w-xl mx-auto">
          Create a free account. Prove an idea in a backtest. Climb the board.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button size="lg" className="w-full sm:w-auto bg-primary text-primary-foreground hover:bg-primary/90 text-base h-12 px-8" asChild>
            <a href={SIGNUP_HREF}>Start Free</a>
          </Button>
        </div>

        <div className="mt-24 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
          <div>© 2026 Agentic Trading Lab. All rights reserved.</div>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a href="#" className="hover:text-foreground">Terms</a>
            <a href="#" className="hover:text-foreground">Privacy</a>
            <a href="#" className="hover:text-foreground">Documentation</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
