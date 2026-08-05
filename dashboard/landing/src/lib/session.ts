import { useEffect, useState } from "react";

const AUTH_USER_KEY = "auth-user";

/** True when a cached auth-user profile is present (session cookie is HttpOnly). */
export function hasAuthToken(): boolean {
  try {
    return Boolean(localStorage.getItem(AUTH_USER_KEY));
  } catch {
    return false;
  }
}

/** Reactive signed-in flag (landing CTAs no longer branch on this). */
export function useSignedIn(): boolean {
  const [signedIn, setSignedIn] = useState(hasAuthToken);

  useEffect(() => {
    const sync = () => setSignedIn(hasAuthToken());
    sync();
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  return signedIn;
}
