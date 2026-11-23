# Enabling social login (Google · Microsoft · LinkedIn)

CurveStudio supports OpenID Connect sign-in for Google, Microsoft and LinkedIn.
The code is already deployed; a provider appears on the home page and `/login` **only
when its client ID and secret are configured**. No code change is needed to enable one.

The flow is OAuth 2.0 authorization code + PKCE (S256), with a signed state cookie and a
single-use 120-second exchange code. Accounts are matched by **verified email**, so a
person who registered with a password and later uses Google with the same address lands in
the same account (and is marked verified). Accounts created through a provider cannot use
password sign-in.

## Redirect URIs (register these exactly)

```
https://curvestudio.concreteinfo.cloud/api/v1/auth/oauth/google/callback
https://curvestudio.concreteinfo.cloud/api/v1/auth/oauth/microsoft/callback
https://curvestudio.concreteinfo.cloud/api/v1/auth/oauth/linkedin/callback
```

For a local run, register `http://localhost:5173/api/v1/auth/oauth/{provider}/callback`
as well and set `CURVESTUDIO_APP_URL=http://localhost:5173`.

## 1. Google

1. <https://console.cloud.google.com/> → create/select a project.
2. **APIs & Services → OAuth consent screen**: User type External; app name `CurveStudio`;
   support email and developer contact; scopes `openid`, `email`, `profile` (all basic,
   no verification review needed).
3. **Credentials → Create credentials → OAuth client ID → Web application**.
   Authorized redirect URI: the Google line above.
4. Copy the **Client ID** and **Client secret**.

Env:
```
CURVESTUDIO_OAUTH_GOOGLE_CLIENT_ID=...
CURVESTUDIO_OAUTH_GOOGLE_CLIENT_SECRET=...
```

## 2. Microsoft (Entra ID)

1. <https://entra.microsoft.com/> → **App registrations → New registration**.
2. Name `CurveStudio`; Supported account types: *Accounts in any organizational directory
   and personal Microsoft accounts* (multitenant + personal) unless you want to restrict it.
3. **Redirect URI**: platform **Web**, the Microsoft line above.
4. **Certificates & secrets → New client secret** → copy the value (not the ID).
5. Tenant: leave `common` for multitenant + personal, or set your directory ID to restrict.

Env:
```
CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_ID=...
CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_SECRET=...
CURVESTUDIO_OAUTH_MICROSOFT_TENANT=common   # or the directory (tenant) id
```

## 3. LinkedIn

1. <https://www.linkedin.com/developers/apps> → **Create app**; associate it with a
   LinkedIn Page you administer; verify the app.
2. **Products** → request **Sign In with LinkedIn using OpenID Connect**
   (scopes `openid profile email`).
3. **Auth** tab → Authorized redirect URLs: the LinkedIn line above.
4. Copy **Client ID** and **Primary Client Secret** (Auth tab).

Env:
```
CURVESTUDIO_OAUTH_LINKEDIN_CLIENT_ID=...
CURVESTUDIO_OAUTH_LINKEDIN_CLIENT_SECRET=...
```

## 4. Set the variables

**Production (Coolify):** app `curvestudio-vps` → Environment Variables. Add the keys with
the `CURVESTUDIO_` prefix. If the app has a preview environment, add them there too.
Redeploy (or restart) — settings are read at start-up.

**Local:** put the same keys in `apps/api/.env` (never commit them).

## 5. Verify

```bash
curl -s https://curvestudio.concreteinfo.cloud/api/v1/auth/oauth/providers
# {"providers":["google","microsoft","linkedin"]}   (only the configured ones)
```

Open the home page: a “Continue with …” button appears under the sign-in card (and on
`/login`). Click one, complete the provider prompt, and you land back in the workspace.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `redirect_uri_mismatch` | The URI in the provider console differs by scheme, host, port or trailing slash. Copy the exact callback line. |
| Provider button missing | Client ID or secret empty in the running container (settings are frozen at start-up → redeploy). |
| `oauth_error=state` on return | The state cookie expired (10 min) or the flow started in another browser/tab. Retry. |
| `oauth_error=provider` | Token exchange or userinfo failed (wrong secret, disabled provider app, unverified email). Check the API log for `oauth` entries — for Microsoft, the `AADSTS…` code it names is decoded in the table below. |
| Signed in to a different account | The provider's email differs from the one you registered with. Accounts are matched by email, not by provider identity. |
| “invalid or expired” after sign-in | The app exchange code is single-use and valid for 120 s; reloading the callback page twice triggers this. Start again. |

After any failed sign-in, the API log's `oauth microsoft callback failed:` line names the code, and the application's container log (Coolify) is the first place to look.

*Microsoft (Entra ID) `AADSTS` error codes, each with the fix for this deployment. Verified against Microsoft's error-code reference, accessed 23/11/2026 (IST).*

| Code | What Microsoft is saying | The fix here |
|---|---|---|
| `AADSTS50011` | The reply (redirect) address is missing, misconfigured, or doesn't match the reply addresses configured for the app. | The registration must contain the exact callback `https://curvestudio.concreteinfo.cloud/api/v1/auth/oauth/microsoft/callback` — same scheme, host and path, no trailing slash (add the `localhost` variant too for local runs). |
| `AADSTS700016` | The application wasn't found in the directory/tenant; the identifier value may be misconfigured, or the request was sent to the wrong tenant. | Re-copy the Application (client) ID from the registration's Overview page into `CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_ID` (not the Object ID or Directory ID), make `CURVESTUDIO_OAUTH_MICROSOFT_TENANT` match the directory that owns the app, and redeploy. |
| `AADSTS7000215` | An invalid client secret was provided with the request. | The value in `CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_SECRET` is wrong — most often the secret's **Secret ID** was copied instead of its **Value**. Copy the Value from *Certificates & secrets* and redeploy. |
| `AADSTS7000218` | The request body must contain the parameter `client_assertion` or `client_secret`. | `CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_SECRET` is empty in the running container. Set it in Coolify and redeploy — settings are read at start-up. |
| `AADSTS7000222` | The provided client secret keys are expired; create new keys for the app. | The secret passed its expiry date (shown when it was created). Create a new client secret, copy its **Value**, update the env var and redeploy. |
| `AADSTS50194` | The app isn't configured as a multitenant application, so the `/common` endpoint isn't supported for it. | The registration's supported account types and `CURVESTUDIO_OAUTH_MICROSOFT_TENANT` disagree. Keep `common` only with a multitenant registration (*Accounts in any organizational directory and personal Microsoft accounts*), or set the tenant to the directory (tenant) ID for a single-tenant app. |
| `AADSTS65001` | The user or administrator hasn't consented to use the application. | A work/school tenant hasn't granted consent (or disallows user consent). Its admin grants it under **API permissions → Grant admin consent**; personal accounts consent at the sign-in prompt. |
| `AADSTS53003` | Access has been blocked by Conditional Access policies; the policy doesn't allow token issuance. | The user's tenant policy (MFA, compliant device, location) blocks this app. The user completes the challenge (e.g. MFA) or their admin excludes the app — nothing to change in this deployment. |
| `AADSTS54005` | The OAuth2 authorization code was already redeemed; retry with a new valid code. | The one-time authorization code was replayed (reload or Back on the callback URL). Start the sign-in again from the home page. |
| `AADSTS9002313` | The request is malformed or invalid. | Generic `invalid_grant` from the token exchange. Start a fresh sign-in; if it persists, check the tenant/account-type pairing (`AADSTS50194` above). |

## Security notes

- Client secrets live only in server environment variables; nothing is exposed to the SPA.
- PKCE, an HttpOnly `SameSite=Lax` state cookie and a single-use exchange code protect the
  redirect leg; the access/refresh tokens are only issued on `POST /auth/oauth/exchange`.
- Google and LinkedIn must report `email_verified: true`; Microsoft identities must carry
  an email. A provider identity without a verified email is rejected.
- Revoking access in the provider console (or deleting the client) disables the button only
  after the env vars are removed/emptied and the app redeploys.
