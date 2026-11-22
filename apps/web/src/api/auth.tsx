/** Session state on top of the API client: restore, login, register, logout. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, type TokenOut, type UserOut } from "./client";

export type AuthStatus = "loading" | "authed" | "anon";

interface AuthContextValue {
  status: AuthStatus;
  user: UserOut | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  /** Adopt a token pair issued elsewhere (the OAuth exchange) and load the profile. */
  adoptSession: (tokens: TokenOut) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): ReactNode {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<UserOut | null>(null);

  useEffect(() => {
    let alive = true;
    api.onAuthFailure = () => {
      if (alive) {
        setUser(null);
        setStatus("anon");
      }
    };
    if (!api.hasRefreshToken) {
      setStatus("anon");
      return () => {
        alive = false;
        api.onAuthFailure = null;
      };
    }
    void (async () => {
      if (await api.refreshNow()) {
        try {
          const me = await api.me();
          if (alive) {
            setUser(me);
            setStatus("authed");
          }
        } catch {
          if (alive) setStatus("anon");
        }
      } else {
        if (alive) setStatus("anon");
      }
    })();
    return () => {
      alive = false;
      api.onAuthFailure = null;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    await api.login(email, password);
    setUser(await api.me());
    setStatus("authed");
  }, []);

  const register = useCallback(async (email: string, password: string, name: string) => {
    // Registration is not a login: the address must be verified from the inbox first,
    // so the page shows "check your inbox" and the session stays anonymous.
    await api.register(email, password, name);
  }, []);

  const adoptSession = useCallback(async (tokens: TokenOut) => {
    // Same path the restore effect takes: store the pair, then confirm it against /me.
    api.adoptTokens(tokens);
    try {
      const me = await api.me();
      setUser(me);
      setStatus("authed");
    } catch (error) {
      api.clearTokens();
      setUser(null);
      setStatus("anon");
      throw error;
    }
  }, []);

  const logout = useCallback(async () => {
    await api.logout();
    setUser(null);
    setStatus("anon");
  }, []);

  const value = useMemo(
    () => ({ status, user, login, register, adoptSession, logout }),
    [status, user, login, register, adoptSession, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}
