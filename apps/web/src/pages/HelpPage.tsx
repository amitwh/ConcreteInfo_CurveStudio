/**
 * /help — the canonical CurveStudio help (docs/help-content.md) rendered as
 * Markdown through react-markdown + remark-gfm, so the web shows the same text
 * the desktop app renders from the one source. The Markdown stack is imported
 * here only: this route is lazy, keeping it out of every other chunk. Raw HTML
 * stays disabled — react-markdown's default sanitisation is left untouched.
 *
 * A section index sits beside the content on wide screens and above it on narrow
 * ones; the search box filters the index and jumps to the first match.
 */

import {
  useEffect,
  isValidElement,
  useMemo,
  useState,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
  type SubmitEvent,
} from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, Search } from "lucide-react";
import { HELP_SECTIONS, type HelpSection } from "../content/helpDocument";

const SEARCHABLE = HELP_SECTIONS.map((section) => ({ section, text: section.searchText }));

type ImageProps = ComponentPropsWithoutRef<"img">;

function isImage(node: ReactNode): node is ReactElement<ImageProps> {
  return isValidElement(node) && typeof (node.props as ImageProps).src === "string";
}

/** The canonical document references its screenshots as `images/…`; the web ships them at /help/…. */
function helpAsset(src: string | undefined): string | undefined {
  if (src === undefined || !src.startsWith("images/")) return src;
  return `/help/${src.slice("images/".length)}`;
}

/**
 * The Markdown rendering details: external links open a new tab, screenshots
 * resolve to the app's own /help assets (a lone image in a paragraph becomes a
 * captioned figure), and tables keep the app's wrapping and styles.
 */
const components: Components = {
  a({ node: _node, children, href, ...rest }) {
    const external = href !== undefined && !href.startsWith("/");
    return (
      <a
        href={href}
        {...rest}
        rel={external ? "noopener noreferrer" : undefined}
        target={external ? "_blank" : undefined}
      >
        {children}
      </a>
    );
  },
  img({ node: _node, src, alt, ...rest }) {
    return <img src={helpAsset(src)} alt={alt} loading="lazy" {...rest} />;
  },
  p({ node: _node, children, ...rest }) {
    const content = Array.isArray(children) ? children : [children];
    const onlyImage =
      content.length >= 1 &&
      content.every((child) => isImage(child) || (typeof child === "string" && child.trim() === ""));
    if (onlyImage) {
      const image = content.find((child) => isImage(child));
      const caption = image === undefined ? undefined : (image.props as ImageProps).alt;
      return (
        <figure className="help-figure">
          {image}
          {typeof caption === "string" && caption !== "" ? (
            <figcaption>{caption}</figcaption>
          ) : null}
        </figure>
      );
    }
    return <p {...rest}>{children}</p>;
  },
  table({ node: _node, children, ...rest }) {
    return (
      <div className="table-wrap">
        <table className="table help-table" {...rest}>
          {children}
        </table>
      </div>
    );
  },
};

function Section({ section }: { section: HelpSection }): ReactElement {
  return (
    <section id={section.id} className="help-section" aria-labelledby={`${section.id}-title`}>
      <h2 id={`${section.id}-title`}>{section.title}</h2>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {section.body}
      </Markdown>
    </section>
  );
}

/** The index shows the heading's number as its own chip, as the desktop's list does. */
function displayTitle(section: HelpSection): string {
  return section.title.replace(/^\d+\.\s+/, "");
}

/** Scroll to a section without touching focus (the search box keeps the caret). */
function jumpTo(id: string): void {
  const element = document.getElementById(id);
  if (element === null) return;
  if (typeof element.scrollIntoView === "function") {
    element.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

export function HelpPage(): ReactElement {
  const [query, setQuery] = useState("");
  const trimmed = query.trim().toLowerCase();

  const matches = useMemo(
    () =>
      trimmed === ""
        ? HELP_SECTIONS
        : SEARCHABLE.filter((entry) => entry.text.includes(trimmed)).map((entry) => entry.section),
    [trimmed],
  );

  const firstMatch = matches[0];

  // A page loaded at /help#keyboard-shortcuts has no section to scroll to until React
  // has mounted; scroll once the content exists (the anchor ids are in the markup).
  useEffect(() => {
    const hash = window.location.hash.replace(/^#/, "");
    if (hash !== "") jumpTo(hash);
  }, []);

  const onChange = (value: string): void => {
    setQuery(value);
    const next = value.trim().toLowerCase();
    if (next === "") return;
    const first = SEARCHABLE.find((entry) => entry.text.includes(next));
    if (first !== undefined) jumpTo(first.section.id);
  };

  const onSubmit = (event: SubmitEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (firstMatch !== undefined) jumpTo(firstMatch.id);
  };

  return (
    <main className="help-page">
      <header className="help-header">
        <h1>
          <BookOpen className="icon icon--lg" aria-hidden="true" focusable="false" />
          Help
        </h1>
        <p className="muted">
          The same guidance the desktop app shows. Search the sections, or read them in
          order.
        </p>
      </header>

      <div className="help-layout">
        <aside className="help-index" aria-label="Help sections">
          <form role="search" className="help-search" onSubmit={onSubmit}>
            <label htmlFor="help-search">Search help</label>
            <span className="search-field">
              <Search className="icon" aria-hidden="true" focusable="false" />
              <input
                id="help-search"
                type="search"
                placeholder="Filter sections…"
                value={query}
                onChange={(event) => onChange(event.target.value)}
                autoComplete="off"
              />
            </span>
          </form>

          {matches.length === 0 ? (
            <p className="help-index__empty" role="status">
              No section matches “{query.trim()}”.
            </p>
          ) : (
            <nav aria-label="Sections">
              <ol className="help-index__list">
                {matches.map((section) => (
                  <li key={section.id}>
                    <a href={`#${section.id}`}>
                      <span className="help-index__num" aria-hidden="true">
                        {HELP_SECTIONS.indexOf(section) + 1}
                      </span>
                      {displayTitle(section)}
                    </a>
                  </li>
                ))}
              </ol>
            </nav>
          )}
        </aside>

        <div className="help-content">
          {HELP_SECTIONS.map((section) => (
            <Section key={section.id} section={section} />
          ))}
        </div>
      </div>
    </main>
  );
}
