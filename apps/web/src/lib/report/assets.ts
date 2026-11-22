/**
 * Browser-only loaders for the images the fit report embeds. Everything here touches
 * `fetch` and `FileReader`, so it is dynamically imported by `export.ts` and never by the
 * pure report-model tests.
 *
 * Images are turned into data URLs because react-pdf must be able to render the document
 * without a second network round-trip for every page.
 */

const LOGO_URL = "/curvestudio-logo.png";
const WATERMARK_URL = "/brand/watermark.png";

function blobToDataUrl(blob: Blob): Promise<string | null> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(blob);
  });
}

async function fetchAsDataUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await blobToDataUrl(await response.blob());
  } catch {
    return null;
  }
}

/** The CurveStudio logo printed in every page header, or null when unavailable. */
export async function loadLogoDataUrl(): Promise<string | null> {
  return fetchAsDataUrl(LOGO_URL);
}

/** The pre-rotated faint watermark, or null when unavailable. */
export async function loadWatermarkDataUrl(): Promise<string | null> {
  return fetchAsDataUrl(WATERMARK_URL);
}
