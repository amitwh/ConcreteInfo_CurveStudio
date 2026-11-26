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
import { api, type MeOut, type TokenOut } from "./client";
import { clear as clearWorkspaceCache } from "../workspace/workspaceCache";

export type AuthStatus = "loading" | "authed" | "anon";

interface AuthContextValue {
  status: AuthStatus;
  user: MeOut | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  /** Adopt a token pair issued elsewhere (the OAuth exchange) and load the profile. */
  adoptSession: (tokens: TokenOut) => Promise<void>;
  logout: () => Promise<void>;
  /** Replace the held profile after a change, so the header shows a new name at once. */
  updateUser: (user: MeOut) => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): ReactNode {
  // Without a stored refresh token there is no session to restore, so a visitor is known
  // from the first render; "loading" means exactly that an account may be signed in.
  const [status, setStatus] = useState<AuthStatus>(() =>
    api.hasRefreshToken ? "loading" : "anon",
  );
  const [user, setUser] = useState<MeOut | null>(null);

  useEffect(() => {
    let alive = true;
    api.onAuthFailure = () => {
      // The session died: no cached workspace answer may outlive it.
      clearWorkspaceCache();
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
    // The store belongs to the session that filled it: a new one starts empty.
    clearWorkspaceCache();
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
    // The store belongs to the session that filled it: a new one starts empty.
    clearWorkspaceCache();
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
    // The session ended; nothing cached about its workspace may survive it.
    clearWorkspaceCache();
    setUser(null);
    setStatus("anon");
  }, []);

  const updateUser = useCallback((next: MeOut) => setUser(next), []);

  const value = useMemo(
    () => ({ status, user, login, register, adoptSession, logout, updateUser }),
    [status, user, login, register, adoptSession, logout, updateUser],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}
