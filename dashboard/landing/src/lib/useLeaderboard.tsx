import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { fetchLeaderboard, type BoardData } from "@/lib/leaderboard";

/** THREE states, and they must be distinguishable on screen.
 *
 *  Not two plus a fallback. A silent fallback to sample curves would make "the
 *  backend is down" and "the backend is fine" render near-identically -- the
 *  failure shape CLAUDE.md's fail-closed-is-not-fail-visible section is about,
 *  and the same one that degraded the news panel in prod for hours with a green
 *  test suite. Render's free tier cold-starts in 30-60s, so `loading` is a
 *  routine first-visit occurrence rather than an edge case. */
export type BoardState =
  | { status: "loading" }
  | { status: "ready"; data: BoardData }
  | { status: "error"; message: string };

const LeaderboardContext = createContext<BoardState>({ status: "loading" });

/** Turns a caught fetch rejection into the `{status: "error"}` half of
 *  `BoardState`. Factored out of the effect below so this branch -- the one
 *  place a failure splits into "we gave up waiting" vs. "the request itself
 *  failed" -- is plain, DOM-free logic that can be run and asserted on under
 *  node, the same way `leaderboard.ts`'s pure functions are. Everything else
 *  in this file needs a real React render (a mounted provider, an effect, a
 *  commit) to mean anything, which is why this is the one piece pulled out.
 *
 *  SHOWING `err.message` VERBATIM IS SAFE ONLY BECAUSE `fetchLeaderboard`
 *  SANITISES AT ITS OWN BOUNDARY -- it maps the browser's `TypeError: Failed to
 *  fetch` and a `<!DOCTYPE`-shaped 2xx body onto visitor-facing sentences, and
 *  re-throws AbortError untouched so the branch below can still see it. Do not
 *  reintroduce a raw-message path here on the grounds that this function is
 *  where the string is chosen; the sanitising has to happen where the cause is
 *  known. */
export function classifyFetchFailure(
  err: unknown,
  aborted: boolean,
): { status: "error"; message: string } {
  if (aborted && !(err instanceof Error && err.name !== "AbortError")) {
    return { status: "error", message: "Timed out waiting for the board." };
  }
  return {
    status: "error",
    message: err instanceof Error ? err.message : "Unknown error",
  };
}

/** One fetch for the page. The hero and the Race standings are four screens
 *  apart and render the same board; fetching twice doubles the load on a
 *  cold-starting free-tier backend and lets the two disagree. */
export function LeaderboardProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<BoardState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    // A THIRD CAUSE OF `signal.aborted`, AND THE ONE THAT IS NOT A FAILURE.
    // `classifyFetchFailure` splits an abort into "the 45s ceiling fired" and
    // "the request failed on its own" -- but the cleanup below aborts too, and
    // an unmount reaching that branch is reported to the user as "Timed out
    // waiting for the board." Today the setState lands on an unmounted provider
    // and is a no-op, so the bug is latent; add <StrictMode> to main.tsx and
    // React 18 double-invokes this effect, aborting the first fetch on a
    // component whose state survives -- so the first thing a visitor sees is
    // the timeout copy, before the second fetch resolves over it. Intent is
    // tracked explicitly rather than inferred from `signal.aborted`, which
    // cannot tell the three cases apart.
    let cancelled = false;
    // Generous, because a free-tier cold start is 30-60s and giving up at 10
    // would report a failure to every first visitor of the day. Same ceiling
    // MarketTicker already uses.
    const timeout = setTimeout(() => controller.abort(), 45_000);
    fetchLeaderboard(controller.signal)
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) setState(classifyFetchFailure(err, controller.signal.aborted));
      })
      .finally(() => clearTimeout(timeout));
    return () => {
      cancelled = true;
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  return (
    <LeaderboardContext.Provider value={state}>{children}</LeaderboardContext.Provider>
  );
}

export function useLeaderboard(): BoardState {
  return useContext(LeaderboardContext);
}
