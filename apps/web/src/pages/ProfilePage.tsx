/**
 * /profile — the signed-in account: identity, plan and usage, report and privacy
 * preferences, sessions and deletion. The shell opens it as an overlay over the workspace,
 * so it is a plain column of sections with no page chrome of its own.
 *
 * The profile is server state (TanStack Query, key ["me"]). Every change sends its
 * request, writes the profile the server answers with back into that cache and into the
 * session, so the header shows a new name at once. A section never has two requests in
 * flight: while one is running its controls are disabled and further ones are refused.
 */

import {
  useCallback,
  useId,
  useRef,
  useState,
  type FormEvent,
  type ReactElement,
  type ReactNode,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../api/auth";
import { api, type MeOut } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Switch } from "../components/Switch";
import { ErrorBox, Loading, apiErrorText } from "../components/States";
import { fmtBytes, fmtInt } from "../format";

/** The profile's key in the query cache; every change below writes its answer here. */
const ME_KEY = ["me"] as const;

export function ProfilePage(): ReactElement {
  const meQuery = useQuery({ queryKey: ME_KEY, queryFn: () => api.me() });

  if (meQuery.isPending) return <Loading label="Loading your profile…" />;
  if (meQuery.isError) {
    return (
      <div className="profile-page">
        <ErrorBox error={meQuery.error} />
        <button type="button" className="btn" onClick={() => void meQuery.refetch()}>
          Retry
        </button>
      </div>
    );
  }
  return <Profile me={meQuery.data} />;
}

function Profile({ me }: { me: MeOut }): ReactElement {
  const named = me.name.trim() !== "";

  return (
    <div className="profile-page">
      <AccountSection me={me} />
      <PlanSection me={me} />
      <PreferenceSection
        title="Reports"
        label="Show my name on PDF reports"
        value={me.preferences.report_byline}
        disabled={!named}
        change={async (report_byline) => api.updatePreferences({ report_byline })}
        hint={
          named ? (
            <>
              Your name appears as “Prepared by {me.name.trim()}” on the PDF reports you
              export.
            </>
          ) : (
            <>
              A name is needed first: add your name under Account, and it appears as
              “Prepared by …” on the PDF reports you export.
            </>
          )
        }
      />
      <PreferenceSection
        title="Privacy"
        label="Send error reports"
        value={me.preferences.error_reports}
        change={async (error_reports) => api.updatePreferences({ error_reports })}
        hint={
          <>
            When something goes wrong, an anonymous report helps us fix it. Reports never
            include your account, your projects or your data.
          </>
        }
      />
      <SessionsSection />
      <DangerSection me={me} />
    </div>
  );
}

/** Write a profile the server just answered with into the cache and the session. */
function useApply(): (me: MeOut) => void {
  const queryClient = useQueryClient();
  const { updateUser } = useAuth();
  return useCallback(
    (me: MeOut) => {
      queryClient.setQueryData(ME_KEY, me);
      updateUser(me);
    },
    [queryClient, updateUser],
  );
}

interface Section {
  /** A request of this section is in flight: its controls are disabled. */
  locked: boolean;
  notice: string | null;
  error: string | null;
  /**
   * Run the section's request. Refused (and resolves false) while one is already in
   * flight. `task` returns the confirmation to show, or null for none; what it throws is
   * shown as the section's error.
   */
  run: (task: () => Promise<string | null>) => Promise<boolean>;
  /** Show an error without a request (a rule the page can check itself). */
  fail: (message: string) => void;
  /** Forget the previous confirmation and error. */
  clear: () => void;
}

/** The state every section shares: one request at a time, its confirmation and its error. */
function useSection(): Section {
  const [locked, setLocked] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // A ref, not the state: two clicks in one tick must not both get through before React
  // has re-rendered the controls as disabled.
  const inflight = useRef(false);

  const run = useCallback(async (task: () => Promise<string | null>): Promise<boolean> => {
    if (inflight.current) return false;
    inflight.current = true;
    setLocked(true);
    setNotice(null);
    setError(null);
    try {
      setNotice(await task());
      return true;
    } catch (err) {
      setError(apiErrorText(err, "something went wrong"));
      return false;
    } finally {
      inflight.current = false;
      setLocked(false);
    }
  }, []);

  const fail = useCallback((message: string): void => {
    setNotice(null);
    setError(message);
  }, []);

  const clear = useCallback((): void => {
    setNotice(null);
    setError(null);
  }, []);

  return { locked, notice, error, run, fail, clear };
}

/** A section's confirmation (a status region) and its error, each when present. */
function SectionFeedback({
  notice,
  error,
}: {
  notice: string | null;
  error: string | null;
}): ReactElement | null {
  if (notice === null && error === null) return null;
  return (
    <>
      {notice !== null && (
        <p className="alert alert--success" role="status">
          {notice}
        </p>
      )}
      {error !== null && (
        <p className="alert alert--error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}

/** A section with its heading; `children` follow the feedback. */
function SectionFrame({
  title,
  feedback,
  children,
}: {
  title: string;
  feedback: ReactElement | null;
  children: ReactNode;
}): ReactElement {
  const headingId = useId();
  return (
    <section className="profile-section" aria-labelledby={headingId}>
      <h3 id={headingId}>{title}</h3>
      {feedback}
      {children}
    </section>
  );
}

/* ---- Account -------------------------------------------------------------------------- */

function AccountSection({ me }: { me: MeOut }): ReactElement {
  const section = useSection();
  const onSaved = useApply();
  const ids = useId();
  const [name, setName] = useState(me.name);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const { locked } = section;

  const saveName = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") {
      section.fail("Enter a name.");
      return;
    }
    await section.run(async () => {
      const next = await api.updateMe(trimmed);
      setName(next.name);
      onSaved(next);
      return "Name saved.";
    });
  };

  const resend = (): void =>
    void section.run(async () => {
      await api.resendVerification(me.email);
      return "If the address is unverified, a new link is on its way.";
    });

  const changePassword = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const changed = await section.run(async () => {
      const next = await api.changePassword(currentPassword, newPassword);
      onSaved(next);
      return "Password changed. Other devices were signed out.";
    });
    if (changed) {
      setCurrentPassword("");
      setNewPassword("");
    }
  };

  const sendPasswordLink = (): void =>
    void section.run(async () => {
      await api.forgotPassword(me.email);
      return "If your address is registered, a link to set a password is on its way.";
    });

  return (
    <SectionFrame
      title="Account"
      feedback={<SectionFeedback notice={section.notice} error={section.error} />}
    >
      <dl className="profile-facts">
        <div>
          <dt>Email</dt>
          <dd>{me.email}</dd>
        </div>
      </dl>
      {!me.email_verified && (
        <p className="alert alert--warning profile-verify">
          Your email is not verified.
          <button type="button" className="btn btn--small" disabled={locked} onClick={resend}>
            Resend verification email
          </button>
        </p>
      )}
      <form className="profile-form" onSubmit={(event) => void saveName(event)} noValidate>
        <div className="field">
          <label htmlFor={`${ids}-name`}>Name</label>
          <input
            id={`${ids}-name`}
            type="text"
            value={name}
            maxLength={100}
            autoComplete="name"
            disabled={locked}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <button type="submit" className="btn btn--primary" disabled={locked}>
          Save
        </button>
      </form>
      <form
        className="profile-form"
        onSubmit={(event) => void changePassword(event)}
        noValidate
      >
        <div className="field">
          <label htmlFor={`${ids}-current`}>Current password</label>
          <input
            id={`${ids}-current`}
            type="password"
            autoComplete="current-password"
            value={currentPassword}
            disabled={locked}
            onChange={(event) => setCurrentPassword(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${ids}-new`}>New password</label>
          <input
            id={`${ids}-new`}
            type="password"
            autoComplete="new-password"
            value={newPassword}
            disabled={locked}
            onChange={(event) => setNewPassword(event.target.value)}
          />
        </div>
        <div className="btn-row">
          <button type="submit" className="btn btn--primary" disabled={locked}>
            Change password
          </button>
          <button type="button" className="btn" disabled={locked} onClick={sendPasswordLink}>
            Set a password by e-mail
          </button>
        </div>
        <p className="hint">
          For an account created with Google or Microsoft, or if you have forgotten your
          password.
        </p>
      </form>
    </SectionFrame>
  );
}

/* ---- Plan and usage ------------------------------------------------------------------- */

function PlanSection({ me }: { me: MeOut }): ReactElement {
  const { plan, usage } = me;
  return (
    <SectionFrame title="Plan and usage" feedback={null}>
      <p className="profile-plan">
        Plan: <strong>{plan.name}</strong>
      </p>
      <div className="meters">
        <Meter label="Datasets" used={usage.datasets} limit={plan.max_datasets} />
        <Meter
          label="Storage"
          used={usage.dataset_bytes}
          limit={plan.max_dataset_bytes}
          format={fmtBytes}
        />
        <Meter label="Stored fits" used={usage.fits} limit={plan.max_fits} />
        <Meter
          label="Auto-fit runs today"
          used={usage.autofits_today}
          limit={plan.max_autofits_per_day}
        />
      </div>
    </SectionFrame>
  );
}

interface MeterProps {
  label: string;
  used: number;
  limit: number;
  /** How to print both numbers; counts by default, bytes for storage. */
  format?: (value: number) => string;
}

/** "used of limit", a progress bar and the percentage; over 90 % is said in words. */
function Meter({ label, used, limit, format = fmtInt }: MeterProps): ReactElement {
  const percent = limit > 0 ? (used / limit) * 100 : used > 0 ? 100 : 0;
  const shown = Math.round(percent);
  const over = shown > 90;
  const count = `${format(used)} of ${format(limit)}`;
  return (
    <div className="meter">
      <div className="meter__head">
        <span className="meter__label">{label}</span>
        <span className="meter__count">{count}</span>
      </div>
      <div
        className="meter__track"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        // Usage can pass a limit that was lowered; the bar fills, the numbers stay true.
        aria-valuemax={Math.max(limit, used, 0)}
        aria-valuenow={used}
        aria-valuetext={count}
      >
        <div
          className={over ? "meter__fill meter__fill--over" : "meter__fill"}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
      <div className="meter__foot">
        <span>{shown}%</span>
        {over && <span className="badge badge--bad">Over 90% used</span>}
      </div>
    </div>
  );
}

/* ---- Reports and Privacy (the two switches) ------------------------------------------- */

function PreferenceSection({
  title,
  label,
  hint,
  value,
  disabled = false,
  change,
}: {
  title: string;
  label: string;
  hint: ReactNode;
  value: boolean;
  disabled?: boolean;
  /** Apply the new value on the server; resolves with the profile it answers with. */
  change: (value: boolean) => Promise<MeOut>;
}): ReactElement {
  const section = useSection();
  const onSaved = useApply();
  // The switch flips at once and comes back if the server refuses.
  const [optimistic, setOptimistic] = useState<boolean | null>(null);

  const toggle = (): void =>
    void section.run(async () => {
      setOptimistic(!value);
      try {
        onSaved(await change(!value));
      } finally {
        setOptimistic(null);
      }
      return "Preference saved.";
    });

  return (
    <SectionFrame
      title={title}
      feedback={<SectionFeedback notice={section.notice} error={section.error} />}
    >
      <Switch
        label={label}
        checked={optimistic ?? value}
        disabled={section.locked || disabled}
        onChange={toggle}
      />
      <p className="hint">{hint}</p>
    </SectionFrame>
  );
}

/* ---- Sessions ------------------------------------------------------------------------- */

function SessionsSection(): ReactElement {
  const section = useSection();
  const onSaved = useApply();
  const [confirming, setConfirming] = useState(false);
  const { locked, error } = section;

  const close = useCallback((): void => setConfirming(false), []);

  const revoke = (): void =>
    void section.run(async () => {
      onSaved(await api.revokeAllSessions());
      close();
      return "Other devices were signed out.";
    });

  // While the confirmation is open, its result is shown there, not twice.
  const feedback = confirming ? null : (
    <SectionFeedback notice={section.notice} error={section.error} />
  );

  return (
    <SectionFrame title="Sessions" feedback={feedback}>
      <p>You are signed in on this device.</p>
      <button
        type="button"
        className="btn"
        disabled={locked}
        onClick={() => {
          section.clear();
          setConfirming(true);
        }}
      >
        Sign out everywhere
      </button>
      {confirming && (
        <ConfirmDialog
          title="Sign out everywhere?"
          message="Every other device will be signed out. This one stays signed in."
          confirmLabel="Sign out everywhere"
          busy={locked}
          error={error}
          onConfirm={revoke}
          onClose={close}
        />
      )}
    </SectionFrame>
  );
}

/* ---- Danger zone ---------------------------------------------------------------------- */

/** The part of an e-mail address before the @: what the delete must be confirmed with. */
function localPart(email: string): string {
  const at = email.indexOf("@");
  return at === -1 ? email : email.slice(0, at);
}

function DangerSection({ me }: { me: MeOut }): ReactElement {
  const section = useSection();
  const { logout } = useAuth();
  const [confirming, setConfirming] = useState(false);
  const { locked, error } = section;
  const phrase = localPart(me.email);

  const close = useCallback((): void => setConfirming(false), []);

  const remove = (typed: string): void =>
    void section.run(async () => {
      await api.deleteMe(typed);
      // The delete already revoked every session; this only clears the local one (its own
      // request is refused), and the app returns to the public pages.
      await logout();
      return null;
    });

  // While the confirmation is open, its result is shown there, not twice.
  const feedback = confirming ? null : (
    <SectionFeedback notice={section.notice} error={section.error} />
  );

  return (
    <SectionFrame title="Danger zone" feedback={feedback}>
      <p>
        Deleting your account removes your datasets, fits and jobs. The audit trail is kept,
        but without your account's identity. This cannot be undone.
      </p>
      <button
        type="button"
        className="btn btn--danger"
        disabled={locked}
        onClick={() => {
          section.clear();
          setConfirming(true);
        }}
      >
        Delete account
      </button>
      {confirming && (
        <ConfirmDialog
          title="Delete account?"
          message="Your datasets, fits and jobs will be deleted. This cannot be undone."
          confirmLabel="Delete account"
          danger
          phrase={phrase}
          busy={locked}
          error={error}
          onConfirm={remove}
          onClose={close}
        />
      )}
    </SectionFrame>
  );
}
