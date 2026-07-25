import { Route, Switch, Router as WouterRouter } from 'wouter';
import { Analytics } from '@vercel/analytics/react';
import LandingPage from '@/pages/landing-page';

// Simplified fallback for NotFound just in case
function NotFound() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-background text-foreground">
      <div className="text-center font-mono">
        <h1 className="text-4xl font-bold text-primary mb-4">404</h1>
        <p className="text-muted-foreground">Page not found</p>
      </div>
    </div>
  );
}

function Router() {
  return (
    <Switch>
      <Route path="/" component={LandingPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
      <Router />
      <Analytics />
    </WouterRouter>
  );
}

export default App;