/**
 * /help — the canonical CurveStudio help text (the desktop app renders the same
 * content). A section index sits beside the content on wide screens and above it on
 * narrow ones; the search box filters the index and jumps to the first match.
 */

import {
  useEffect,
  useMemo,
  useState,
  type SubmitEvent,
  type ReactElement,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import { BookOpen, Search } from "lucide-react";
import {
  HELP_SECTIONS,
  sectionSearchText,
  type HelpBlock,
  type HelpSection,
  type Inline,
} from "../content/help";

const SEARCHABLE = HELP_SECTIONS.map((section) => ({
  section,
  text: sectionSearchText(section),
}));

function InlineNodes({ nodes }: { nodes: Inline[] }): ReactElement {
  return <>{nodes.map((node, index) => renderInline(node, index))}</>;
}

function renderInline(node: Inline, index: number): ReactNode {
  if (typeof node === "string") return node;
  switch (node.kind) {
    case "code":
      return <code key={index}>{node.text}</code>;
    case "strong":
      return <strong key={index}>{node.text}</strong>;
    case "em":
      return <em key={index}>{node.text}</em>;
    case "link":
      return node.href.startsWith("/") ? (
        <Link key={index} to={node.href}>
          {node.text}
        </Link>
      ) : (
        <a key={index} href={node.href} rel="noopener noreferrer" target="_blank">
          {node.text}
        </a>
      );
  }
}

function Block({ block }: { block: HelpBlock }): ReactElement {
  switch (block.kind) {
    case "p":
      return (
        <p>
          <InlineNodes nodes={block.content} />
        </p>
      );
    case "h3":
      return <h3>{block.text}</h3>;
    case "ul":
      return (
        <ul>
          {block.items.map((item, index) => (
            <li key={index}>
              <InlineNodes nodes={item} />
            </li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol>
          {block.items.map((item, index) => (
            <li key={index}>
              <InlineNodes nodes={item} />
            </li>
          ))}
        </ol>
      );
    case "figure":
      return (
        <figure className="help-figure">
          <img src={block.src} alt={block.alt} loading="lazy" />
          <figcaption>{block.caption}</figcaption>
        </figure>
      );
    case "table":
      return (
        <div className="table-wrap">
          <table className="table help-table">
            <thead>
              <tr>
                {block.headers.map((header) => (
                  <th key={header} scope="col">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>
                      <InlineNodes nodes={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
}

function Section({ section }: { section: HelpSection }): ReactElement {
  return (
    <section id={section.id} className="help-section" aria-labelledby={`${section.id}-title`}>
      <h2 id={`${section.id}-title`}>{section.title}</h2>
      {section.blocks.map((block, index) => (
        <Block key={index} block={block} />
      ))}
    </section>
  );
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
                      {section.title}
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
