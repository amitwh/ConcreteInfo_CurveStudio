/**
 * The browser's "install this app" event, kept until the user asks for it.
 *
 * Chromium fires `beforeinstallprompt` once the app qualifies; showing the button only
 * then means no dead control on browsers that do not install web apps. The prompt can be
 * used once, so it is cleared after the user answers it.
 */

import { useEffect, useState } from "react";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

export interface InstallPrompt {
  canInstall: boolean;
  install: () => Promise<void>;
}

export function useInstallPrompt(): InstallPrompt {
  const [event, setEvent] = useState<BeforeInstallPromptEvent | null>(null);

  useEffect(() => {
    const onPrompt = (promptEvent: Event): void => {
      promptEvent.preventDefault();
      setEvent(promptEvent as BeforeInstallPromptEvent);
    };
    const onInstalled = (): void => setEvent(null);
    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  return {
    canInstall: event !== null,
    install: async () => {
      if (event === null) return;
      await event.prompt();
      setEvent(null);
    },
  };
}
