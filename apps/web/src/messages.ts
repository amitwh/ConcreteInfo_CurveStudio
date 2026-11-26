/**
 * The engine's codes as sentences (design item 16): the copy of the core's
 * `curvestudio.messages` mapping, generated into `src/content/engine-messages.json`
 * by `scripts/export_engine_messages.py` and pinned to it by a core test. A status or
 * error code never reaches the user as a bare token — the sentence leads and the
 * server's own message follows it.
 */

import engineMessages from "./content/engine-messages.json";

/** The code → sentence mapping, exactly the core's `curvestudio.messages.SENTENCES`. */
export const ENGINE_SENTENCES: Readonly<Record<string, string>> = engineMessages;

/** The one plain sentence for `code`, or null when it has none. */
export function engineSentence(code: string): string | null {
  const sentence = ENGINE_SENTENCES[code];
  return sentence !== undefined && sentence !== "" ? sentence : null;
}

/** A code and its own message as the user reads them: the sentence first (see the core). */
export function explainCode(
  code: string | null | undefined,
  message: string | null | undefined,
  fallback = "something went wrong",
): string {
  const sentence = code != null ? engineSentence(code) : null;
  const detail = message?.trim() ?? "";
  if (sentence === null) return detail !== "" ? detail : fallback;
  if (detail === "") return sentence;
  return `${sentence} — ${detail}`;
}
