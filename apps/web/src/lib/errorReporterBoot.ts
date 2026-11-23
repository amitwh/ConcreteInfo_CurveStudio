/**
 * Imported first by main.tsx, so that its body runs before any other module of the app is
 * evaluated: a throw while the app's modules load is caught by the reporter's hooks, and so
 * still reaches the server. No recovery screen is possible then; the report is the point.
 *
 * Keep it to this. The reporter and the API base are leaves, and nothing here may be able to
 * fail before the hooks are in. The `installErrorReporter` call that main.tsx makes later is
 * the same installation, and changes nothing.
 */
import { API_BASE } from "../api/apiBase";
import { installErrorReporter } from "./errorReporter";

installErrorReporter({ apiBase: API_BASE });
