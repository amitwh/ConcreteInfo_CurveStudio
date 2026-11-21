/**
 * Left sidebar: the dataset list and the upload flow. Uploads accept csv/tsv/txt/xlsx/
 * json, pre-check the size against the server's limit before sending, show progress and
 * can be aborted. When the server answers that x/y columns must be chosen, the column
 * pickers appear and the same file is re-sent with them.
 */

import { useRef, useState, type ChangeEvent, type DragEvent, type ReactElement } from "react";
import { Database, Trash2, Upload } from "lucide-react";
import {
  ApiError,
  MAX_UPLOAD_BYTES,
  api,
  type DatasetOut,
  type IssueOut,
} from "../api/client";
import { ErrorBox, Loading } from "../components/States";
import { fmtInt } from "../format";

export interface DatasetsSidebarProps {
  datasets: DatasetOut[];
  selectedId: string | null;
  loading: boolean;
  loadError: unknown;
  onSelect: (id: string) => void;
  onUploaded: (dataset: DatasetOut) => void;
  onDeleted: (id: string) => void;
}

const ACCEPT = ".csv,.tsv,.txt,.xlsx,.json";
const MB = 1_000_000;

interface ColumnChoice {
  file: File;
  columns: string[];
  x: string;
  y: string;
  weight: string;
}

function IssuesList({ issues }: { issues: IssueOut[] }): ReactElement | null {
  if (issues.length === 0) return null;
  return (
    <ul className="issues">
      {issues.map((issue, index) => (
        <li key={`${issue.code}-${index}`} className={`issue issue--${issue.severity}`}>
          <strong>{issue.code}</strong> {issue.message}
        </li>
      ))}
    </ul>
  );
}

export function DatasetsSidebar({
  datasets,
  selectedId,
  loading,
  loadError,
  onSelect,
  onUploaded,
  onDeleted,
}: DatasetsSidebarProps): ReactElement {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadIssues, setUploadIssues] = useState<IssueOut[] | null>(null);
  const [columnChoice, setColumnChoice] = useState<ColumnChoice | null>(null);
  const [columnError, setColumnError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<unknown>(null);

  const upload = async (file: File, columns?: Partial<ColumnChoice>): Promise<void> => {
    setUploadError(null);
    setUploadIssues(null);
    setColumnChoice(null);
    setColumnError(null);
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError(
        `the file is ${(file.size / MB).toFixed(1)} MB; the limit is ${(MAX_UPLOAD_BYTES / MB).toFixed(1)} MB`,
      );
      return;
    }
    setUploading(true);
    setProgress(0);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const dataset = await api.uploadDataset(
        file,
        {
          xColumn: columns?.x,
          yColumn: columns?.y,
          weightColumn: columns?.weight || undefined,
        },
        (p) => setProgress(p.total > 0 ? p.loaded / p.total : 0),
        controller.signal,
      );
      setUploadIssues(dataset.issues);
      onUploaded(dataset);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setUploadError("upload cancelled");
      } else if (error instanceof ApiError) {
        setUploadError(error.message);
        const detail = error.detail as
          | { code?: string; columns?: string[]; issues?: IssueOut[] }
          | null;
        if (
          error.status === 422 &&
          detail?.code === "COLUMNS_REQUIRED" &&
          Array.isArray(detail.columns)
        ) {
          setColumnChoice({
            file,
            columns: detail.columns,
            x: detail.columns[0] ?? "",
            y: detail.columns[1] ?? "",
            weight: "",
          });
          if (detail.issues) setUploadIssues(detail.issues);
        }
      } else {
        setUploadError("the upload could not be sent; check your connection");
      }
    } finally {
      setUploading(false);
      abortRef.current = null;
    }
  };

  const onDrop = (event: DragEvent<HTMLElement>): void => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file !== undefined) void upload(file);
  };

  const onPick = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    if (file !== undefined) void upload(file);
    event.target.value = "";
  };

  const onDelete = async (id: string): Promise<void> => {
    setDeleteError(null);
    try {
      await api.request(`/datasets/${id}`, { method: "DELETE" });
      setConfirmDelete(null);
      onDeleted(id);
    } catch (error) {
      setDeleteError(error);
    }
  };

  return (
    <section className="panel" aria-label="Datasets">
      <h2 className="panel-title">Datasets</h2>

      <div
        className={`dropzone${dragOver ? " dropzone--over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <p className="muted">Drag a file here, or</p>
        <button
          type="button"
          className="btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          <Upload className="icon" aria-hidden="true" focusable="false" />
          Choose file
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPT}
          onChange={onPick}
          hidden
          aria-hidden="true"
          tabIndex={-1}
        />
        <p className="hint">
          csv, tsv, txt, xlsx or json · up to {Math.floor(MAX_UPLOAD_BYTES / MB)} MB
        </p>
      </div>

      {uploading && (
        <div className="upload-progress" role="status">
          <progress value={progress} max={1} aria-label="upload progress" />
          <button type="button" className="btn btn--small" onClick={() => abortRef.current?.abort()}>
            Cancel upload
          </button>
        </div>
      )}

      {uploadError !== null && (
        <p className="alert alert--error" role="alert">
          {uploadError}
        </p>
      )}

      {columnChoice !== null && (
        <form
          className="column-picker"
          onSubmit={(e) => {
            e.preventDefault();
            if (columnChoice.x === columnChoice.y) {
              setColumnError("Choose two different columns");
              return;
            }
            setColumnError(null);
            void upload(columnChoice.file, columnChoice);
          }}
        >
          <h3>Choose columns</h3>
          {columnError !== null && (
            <p className="alert alert--error" role="alert">
              {columnError}
            </p>
          )}
          <div className="field">
            <label htmlFor="col-x">X column</label>
            <select
              id="col-x"
              value={columnChoice.x}
              onChange={(e) => setColumnChoice({ ...columnChoice, x: e.target.value })}
            >
              {columnChoice.columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="col-y">Y column</label>
            <select
              id="col-y"
              value={columnChoice.y}
              onChange={(e) => setColumnChoice({ ...columnChoice, y: e.target.value })}
            >
              {columnChoice.columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="col-w">Weight column (optional)</label>
            <select
              id="col-w"
              value={columnChoice.weight}
              onChange={(e) => setColumnChoice({ ...columnChoice, weight: e.target.value })}
            >
              <option value="">none</option>
              {columnChoice.columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <button type="submit" className="btn btn--primary" disabled={uploading}>
            <Upload className="icon" aria-hidden="true" focusable="false" />
            Upload with these columns
          </button>
        </form>
      )}

      {uploadIssues !== null && <IssuesList issues={uploadIssues} />}

      {loading && <Loading label="Loading datasets…" />}
      {loadError !== null && <ErrorBox error={loadError} />}
      {deleteError !== null && <ErrorBox error={deleteError} />}

      {!loading && loadError === null && datasets.length === 0 && (
        <p className="state state--empty">
          <Database className="icon icon--lg" aria-hidden="true" focusable="false" />
          No datasets yet — upload one above.
        </p>
      )}

      <ul className="dataset-list">
        {datasets.map((dataset) => (
          <li
            key={dataset.id}
            className={dataset.id === selectedId ? "dataset dataset--selected" : "dataset"}
          >
            <button
              type="button"
              className="dataset__select"
              aria-current={dataset.id === selectedId ? "true" : undefined}
              onClick={() => onSelect(dataset.id)}
              title={dataset.name}
            >
              <span className="dataset__name">{dataset.name}</span>
              <span className="dataset__meta">
                {fmtInt(dataset.n_rows)} rows
                {dataset.weight_column != null && ` · w: ${dataset.weight_column}`}
              </span>
            </button>
            {confirmDelete === dataset.id ? (
              <span className="dataset__confirm">
                <span>Delete “{dataset.name}” and its fits?</span>
                <button type="button" className="btn btn--small btn--danger" onClick={() => void onDelete(dataset.id)}>
                  <Trash2 className="icon" aria-hidden="true" focusable="false" />
                  Delete
                </button>
                <button type="button" className="btn btn--small" onClick={() => setConfirmDelete(null)}>
                  Keep
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="btn btn--small dataset__delete"
                aria-label={`Delete ${dataset.name}`}
                onClick={() => setConfirmDelete(dataset.id)}
              >
                <Trash2 className="icon" aria-hidden="true" focusable="false" />
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
