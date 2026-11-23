/**
 * Tells the error reporter whose consent counts, and renders nothing. Signed in, the
 * account's error-report preference decides; signed out, the visitor's own switch does.
 * While the session is still being restored the account's choice is unknown, and nothing is
 * collected until it is known: a preference switched off must hold from the first moment.
 */

import { useEffect } from "react";
import { useAuth } from "../api/auth";
import { setConsentResolver } from "../lib/errorReporter";

export function ErrorReportConsent(): null {
  const { status, user } = useAuth();

  useEffect(() => {
    if (status === "authed" && user !== null) {
      const accountSetting = user.preferences.error_reports;
      setConsentResolver(() => accountSetting);
    } else if (status === "loading") {
      setConsentResolver(() => false);
    } else {
      setConsentResolver(null);
    }
    return () => setConsentResolver(null);
  }, [status, user]);

  return null;
}
