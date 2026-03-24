import { useEffect, useMemo, useRef, useState } from "react";

const STEP_ORDER = ["validation", "extracting", "chunks", "metadata", "chroma"];

const STEP_META = {
  validation: {
    title: "Validation",
    shortTitle: "Validate PDF",
    subtitle: "Check extracted text quality before indexing continues.",
    empty: "No validation details yet.",
    doneLabel: "Done",
    activeLabel: "Processing",
    idleLabel: "Pending",
    accent: "emerald",
  },
  extracting: {
    title: "Extracting Text",
    shortTitle: "Extract Text",
    subtitle: "Parse PDF pages and pull the source content.",
    empty: "No extraction details yet.",
    doneLabel: "Done",
    activeLabel: "Processing",
    idleLabel: "Pending",
    accent: "emerald",
  },
  chunks: {
    title: "Creating Chunks",
    shortTitle: "Split in Chunks",
    subtitle: "Apply chunking windows and overlap configuration.",
    empty: "No chunk details yet.",
    doneLabel: "Done",
    activeLabel: "Processing",
    idleLabel: "Pending",
    accent: "amber",
  },
  metadata: {
    title: "Creating Metadata",
    shortTitle: "Create Metadata",
    subtitle: "Build metadata records for the generated chunks.",
    empty: "No metadata details yet.",
    doneLabel: "Done",
    activeLabel: "Processing",
    idleLabel: "Pending",
    accent: "blue",
  },
  chroma: {
    title: "Stored In Chroma DB",
    shortTitle: "Store in Vector DB",
    subtitle: "Persist embeddings and vector records to Chroma.",
    empty: "No Chroma details yet.",
    doneLabel: "Done",
    activeLabel: "Processing",
    idleLabel: "Pending",
    accent: "slate",
  },
};

const INITIAL_STEP_STATE = STEP_ORDER.reduce((accumulator, step) => {
  accumulator[step] = { status: "idle", detail: null };
  return accumulator;
}, {});

function prettyJson(value, fallback) {
  if (!value) {
    return fallback;
  }

  return JSON.stringify(value, null, 2);
}

function citationLabel(citation) {
  const title = citation.title || citation.filename || "Document";
  const page = citation.page_number ?? "?";
  return `${title} p.${page}`;
}

function derivePipelineCards(steps) {
  const rawStatuses = {
    validation: steps.validation.status,
    extracting: steps.extracting.status,
    chunks: steps.chunks.status,
    embeddings: steps.metadata.status,
    store: steps.chroma.status,
  };

  const orderedKeys = ["validation", "extracting", "chunks", "embeddings", "store"];
  const derivedStatuses = {};
  let pipelineBlocked = false;

  orderedKeys.forEach((key) => {
    const rawStatus = rawStatuses[key];

    if (pipelineBlocked) {
      derivedStatuses[key] = "idle";
      return;
    }

    if (rawStatus === "done") {
      derivedStatuses[key] = "done";
      return;
    }

    if (rawStatus === "active") {
      derivedStatuses[key] = "active";
      pipelineBlocked = true;
      return;
    }

    derivedStatuses[key] = "idle";
    pipelineBlocked = true;
  });

  return [
    {
      key: "validation",
      title: "Validation",
      description:
        derivedStatuses.validation === "done"
          ? "Document validation finished successfully."
          : "Check extracted text quality before processing continues.",
      status: derivedStatuses.validation,
      detail: steps.validation.detail,
    },
    {
      key: "extracting",
      title: "Extract Data",
      description:
        derivedStatuses.extracting === "done"
          ? "Text & table parsing from source PDF files completed."
          : "Text & table parsing from source PDF files.",
      status: derivedStatuses.extracting,
      detail: steps.extracting.detail,
    },
    {
      key: "chunks",
      title: "Split in Chunks",
      description:
        steps.chunks.detail?.chunk_size && steps.chunks.detail?.chunk_overlap !== undefined
          ? `Applying recursive character splitting (k=${steps.chunks.detail.chunk_size}, overlap=${steps.chunks.detail.chunk_overlap}).`
          : "Applying recursive character splitting to the extracted document.",
      status: derivedStatuses.chunks,
      detail: steps.chunks.detail,
    },
    {
      key: "embeddings",
      title: "Create Embeddings",
      description:
        derivedStatuses.embeddings === "done"
          ? "Metadata prepared and embeddings staged for vector persistence."
          : "Pending vectorization and metadata preparation before persistence.",
      status: derivedStatuses.embeddings,
      detail: steps.metadata.detail,
    },
    {
      key: "store",
      title: "Store in Vector Database",
      description:
        derivedStatuses.store === "done"
          ? "Vector records stored successfully in Chroma."
          : "Awaiting final upsert to the vector database.",
      status: derivedStatuses.store,
      detail: steps.chroma.detail,
    },
  ];
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  return String(value);
}

function MetricTile({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <div className="mt-2 text-lg font-extrabold tracking-[-0.02em] text-slate-900">{formatValue(value)}</div>
    </div>
  );
}

function KeyValueGrid({ entries }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {entries.map(({ label, value }) => (
        <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">{label}</div>
          <div className="mt-2 break-words text-sm font-medium leading-6 text-slate-800">{formatValue(value)}</div>
        </div>
      ))}
    </div>
  );
}

function PreviewBlock({ label, text }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <p className="mt-3 line-clamp-6 text-sm leading-7 text-slate-700">{text || "No preview available."}</p>
    </div>
  );
}

function CitationHighlightPanel({ citation, onClear }) {
  if (!citation) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
        Select a citation to inspect the supporting passage.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-orange-100 bg-orange-50/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-orange-700">Annotated Passage</div>
          <div className="mt-2 text-lg font-extrabold tracking-[-0.03em] text-slate-900">
            {citation.title || citation.filename || "Supporting Passage"}
          </div>
          <div className="mt-1 text-sm text-slate-600">
            Page {citation.page_number ?? "?"}
            {citation.filename ? ` • ${citation.filename}` : ""}
          </div>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="rounded-lg border border-orange-200 bg-white px-3 py-1.5 text-xs font-bold uppercase tracking-[0.12em] text-orange-700"
        >
          Clear
        </button>
      </div>

      <div className="mt-4 rounded-xl border border-orange-200 bg-white px-4 py-4">
        <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">Useful Passage</div>
        <p className="mt-3 text-sm leading-7 text-slate-700">{citation.excerpt || "No excerpt available."}</p>
      </div>
    </div>
  );
}

function renderInspector(step, detail) {
  if (!step || !detail) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
        Select a completed pipeline card to inspect what this step produced.
      </div>
    );
  }

  if (step === "validation") {
    const stats = detail.stats || {};

    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3">
          <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-emerald-700">Validation Result</div>
          <div className="mt-2 text-sm font-semibold text-slate-900">{detail.message || "PDF validation passed."}</div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricTile label="Pages Total" value={stats.pages_total} />
          <MetricTile label="Pages With Text" value={stats.pages_with_text} />
          <MetricTile label="Total Chars" value={stats.total_chars} />
          <MetricTile label="Alpha Ratio" value={stats.alpha_ratio} />
        </div>
      </div>
    );
  }

  if (step === "extracting") {
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <MetricTile label="Pages Loaded" value={detail.pages_loaded ?? detail.pages_extracted} />
          <MetricTile label="Source File" value={detail.file} />
        </div>
        <PreviewBlock label="Extracted Text Preview" text={detail.sample_text} />
      </div>
    );
  }

  if (step === "chunks") {
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <MetricTile label="Chunks Created" value={detail.chunks_created} />
          <MetricTile label="Chunk Size" value={detail.chunk_size} />
          <MetricTile label="Chunk Overlap" value={detail.chunk_overlap} />
        </div>
        <PreviewBlock label="Chunk Preview" text={detail.sample_chunk} />
      </div>
    );
  }

  if (step === "metadata") {
    const metadata = detail.sample_metadata || detail;
    const entries = [
      { label: "Document ID", value: metadata.document_id },
      { label: "Filename", value: metadata.filename },
      { label: "Title", value: metadata.title },
      { label: "Author", value: metadata.author },
      { label: "Chunk ID", value: metadata.chunk_id },
      { label: "Chunk Index", value: metadata.chunk_index },
      { label: "Page Number", value: metadata.page_number },
      { label: "Source Type", value: metadata.source_type },
      { label: "Embedding Model", value: metadata.embedding_model },
      { label: "Extracted At", value: metadata.extracted_at },
    ];

    return (
      <div className="space-y-4">
        {detail.records_built ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <MetricTile label="Records Built" value={detail.records_built} />
            <MetricTile label="Metadata File" value={detail.metadata_output_path} />
          </div>
        ) : null}
        <KeyValueGrid entries={entries} />
      </div>
    );
  }

  if (step === "chroma") {
    const vectorSample = detail.vector_sample || detail;
    const metadata = vectorSample.metadata || {};
    const entries = [
      { label: "Vector ID", value: vectorSample.id },
      { label: "Document ID", value: metadata.document_id },
      { label: "Filename", value: metadata.filename },
      { label: "Title", value: metadata.title },
      { label: "Author", value: metadata.author },
      { label: "Chunk ID", value: metadata.chunk_id },
      { label: "Chunk Index", value: metadata.chunk_index },
      { label: "Page Number", value: metadata.page_number },
      { label: "Embedding Model", value: metadata.embedding_model },
      { label: "Extracted At", value: metadata.extracted_at },
    ];

    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <MetricTile label="Vectors Stored" value={detail.vectors_stored ?? detail.count} />
          <MetricTile label="Collection" value={detail.collection || detail.vector_collection} />
        </div>
        <KeyValueGrid entries={entries} />
        {Array.isArray(vectorSample.embedding_preview) && vectorSample.embedding_preview.length > 0 ? (
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">Embedding Preview</div>
            <p className="mt-3 text-sm leading-7 text-slate-700">
              First values: {vectorSample.embedding_preview.join(", ")}
            </p>
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
      Select a completed pipeline card to inspect what this step produced.
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [steps, setSteps] = useState(INITIAL_STEP_STATE);
  const [openStep, setOpenStep] = useState(null);
  const [statusText, setStatusText] = useState("Waiting for upload.");
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [queryAlert, setQueryAlert] = useState("");
  const [ingestError, setIngestError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [health, setHealth] = useState("Checking backend...");
  const fileInputRef = useRef(null);
  const viewerRef = useRef(null);
  const [pdfObjectUrl, setPdfObjectUrl] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCitation, setSelectedCitation] = useState(null);

  useEffect(() => {
    let ignore = false;

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const body = await response.json();
        if (!ignore) {
          setHealth(body.status === "ok" ? "Backend connected" : "Backend unavailable");
        }
      } catch {
        if (!ignore) {
          setHealth("Backend unavailable");
        }
      }
    }

    checkHealth();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPdfObjectUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPdfObjectUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  const hasIndexedDocument = useMemo(
    () => steps.chroma.status === "done" || messages.some((message) => message.kind === "indexed"),
    [messages, steps.chroma.status],
  );
  const pipelineCards = useMemo(() => derivePipelineCards(steps), [steps]);
  const totalPages = steps.validation.detail?.stats?.pages_total ?? null;
  const viewerSrc = pdfObjectUrl ? `${pdfObjectUrl}#page=${currentPage}&view=FitH` : null;

  function resetPipelineUi() {
    setSteps(INITIAL_STEP_STATE);
    setOpenStep(null);
    setStatusText("Waiting for upload.");
    setMessages([]);
    setQueryAlert("");
    setIngestError(null);
    setCurrentPage(1);
    setSelectedCitation(null);
  }

  function updateStep(step, patch) {
    setSteps((current) => ({
      ...current,
      [step]: {
        ...current[step],
        ...patch,
      },
    }));
  }

  function toggleStep(step) {
    if (steps[step].status !== "done") {
      return;
    }

    setOpenStep((current) => (current === step ? null : step));
  }

  function addMessage(message) {
    setMessages((current) => [message, ...current]);
  }

  function openFilePicker() {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  }

  function navigateToPage(page) {
    if (!page || Number.isNaN(Number(page))) {
      return;
    }

    const numericPage = Math.max(1, Math.floor(Number(page)));
    const nextPage = totalPages ? Math.min(numericPage, totalPages) : numericPage;
    setCurrentPage(nextPage);
    setQueryAlert("");
    viewerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleCitationClick(citation) {
    if (!citation) {
      return;
    }

    if (selectedCitation?.chunk_id === citation.chunk_id && selectedCitation?.page_number === citation.page_number) {
      setSelectedCitation(null);
      return;
    }

    setSelectedCitation(citation);
    navigateToPage(citation.page_number);
  }

  async function parseResponseSafely(response) {
    const raw = await response.text();

    try {
      return JSON.parse(raw);
    } catch {
      return { detail: raw || response.statusText };
    }
  }

  async function readNdjsonStream(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      lines.forEach((line) => {
        if (line.trim()) {
          onEvent(JSON.parse(line));
        }
      });
    }

    if (buffer.trim()) {
      onEvent(JSON.parse(buffer));
    }
  }

  async function handleFileSelection(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setSelectedFile(file);
    resetPipelineUi();
    setStatusText("Uploading PDF and waiting for validation to complete...");
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/ingest/pdf/stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await parseResponseSafely(response);
        setStatusText(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
        return;
      }

      let finalResult = null;

      await readNdjsonStream(response, (streamEvent) => {
        if (streamEvent.type === "step_started") {
          updateStep(streamEvent.step, { status: "active" });
          setStatusText(`Running step: ${STEP_META[streamEvent.step]?.title || streamEvent.step}`);
          return;
        }

        if (streamEvent.type === "step_completed") {
          updateStep(streamEvent.step, { status: "done", detail: streamEvent.payload });

          const progressMessages = {
            validation: "Validation completed. Starting text extraction...",
            extracting: "Text extraction completed. Creating chunks...",
            chunks: "Chunks created. Creating metadata...",
            metadata: "Metadata created. Storing embeddings in Chroma...",
            chroma: "Stored in Chroma DB successfully.",
          };

          setStatusText(progressMessages[streamEvent.step] || "Processing...");
          return;
        }

        if (streamEvent.type === "error") {
          if (streamEvent.status_code === 422) {
            setIngestError({
              title: "PDF Not Valid For Indexing",
              message: streamEvent.detail?.message || "This PDF could not be indexed.",
              stats: streamEvent.detail?.stats || {},
            });
            setStatusText("Validation failed. Review the details below.");
          } else {
            const detail =
              typeof streamEvent.detail === "string"
                ? streamEvent.detail
                : JSON.stringify(streamEvent.detail || streamEvent);
            setStatusText(detail);
          }

          throw new Error("stream_error");
        }

        if (streamEvent.type === "result") {
          finalResult = streamEvent.payload;
        }
      });

      if (finalResult) {
        updateStep("validation", {
          status: "done",
          detail: steps.validation.detail ?? finalResult.validation ?? null,
        });
        updateStep("extracting", {
          status: "done",
          detail:
            steps.extracting.detail ??
            (finalResult.pages_extracted
              ? {
                pages_loaded: finalResult.pages_extracted,
                sample_text: finalResult.sample_chunk,
              }
              : null),
        });
        updateStep("chunks", {
          status: "done",
          detail:
            steps.chunks.detail ??
            {
              chunks_created: finalResult.chunks_created,
              chunk_size: finalResult.chunk_size,
              chunk_overlap: finalResult.chunk_overlap,
              sample_chunk: finalResult.sample_chunk,
            },
        });
        updateStep("metadata", {
          status: "done",
          detail:
            steps.metadata.detail ??
            {
              records_built: finalResult.metadata_records,
              metadata_output_path: finalResult.metadata_output_path,
              sample_metadata: finalResult.metadata_sample,
            },
        });
        updateStep("chroma", {
          status: "done",
          detail:
            steps.chroma.detail ??
            {
              vectors_stored: finalResult.vectors_stored,
              vector_collection: finalResult.vector_collection,
              vector_db_path: finalResult.vector_db_path,
              vector_sample: finalResult.vector_sample,
            },
        });
        setStatusText("Ingestion completed.");
        addMessage({
          id: `indexed-${Date.now()}`,
          role: "assistant",
          kind: "indexed",
          content: "Document indexed successfully.",
          citations: [],
        });
      }
    } catch (error) {
      if (String(error) !== "Error: stream_error") {
        setStatusText(`Request failed: ${error}`);
      }
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  }

  async function handleAsk() {
    const trimmed = query.trim();
    setQueryAlert("");

    if (!trimmed) {
      setQueryAlert("Please write a question first.");
      return;
    }

    addMessage({
      id: `user-${Date.now()}`,
      role: "user",
      kind: "message",
      content: trimmed,
      citations: [],
    });
    setQuery("");
    setIsQuerying(true);
    setSelectedCitation(null);

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmed }),
      });
      const body = await parseResponseSafely(response);

      if (!response.ok) {
        setQueryAlert(body.detail || "Please upload first a valid document.");
        return;
      }

      addMessage({
        id: `assistant-${Date.now()}`,
        role: "assistant",
        kind: "message",
        content: body.answer || "No answer returned.",
        citations: body.citations || [],
      });
    } catch (error) {
      setQueryAlert(`Query failed: ${error}`);
    } finally {
      setIsQuerying(false);
    }
  }

  function handleComposerKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      handleAsk();
    }
  }

  return (
    <div className="min-h-screen bg-[#f7f9fb] text-slate-800">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-slate-50/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="grid h-9 w-9 place-items-center rounded-full border border-[#d7deea] bg-white text-lg text-[#565e74] shadow-sm">
                ☼
              </div>
              <div className="font-display text-xl font-extrabold tracking-[-0.03em] text-slate-900">
                Samy | The Grand Maester
              </div>
            </div>
            <nav className="hidden items-center gap-5 text-sm font-semibold text-slate-500 md:flex">
              <button className="border-b-2 border-slate-900 pb-1 text-slate-900">Processing</button>
            </nav>
          </div>

          <div className="flex items-center gap-3">
            <button className="grid h-9 w-9 place-items-center rounded-full text-slate-500 transition hover:bg-white">
              ☷
            </button>
            <button className="grid h-9 w-9 place-items-center rounded-full text-slate-500 transition hover:bg-white">
              ✦
            </button>
            <div className="grid h-9 w-9 place-items-center rounded-full border border-slate-200 bg-[linear-gradient(180deg,#2b3540,#596472)] text-xs font-bold text-white">
              SG
            </div>
          </div>
        </div>
      </header>

      <div>
        <main className="px-4 py-6 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl space-y-10">
            <section className="mx-auto max-w-3xl pt-2">
              <div className="mb-8 text-center">
                <h1 className="font-display text-[3rem] font-extrabold tracking-[-0.05em] text-slate-900">
                  Data Ingestion
                </h1>
                <p className="mx-auto mt-3 max-w-2xl text-[1.05rem] leading-8 text-slate-500">
                  Entrust a manuscript to Samy so the archive can validate, index, and prepare grounded answers.
                </p>
              </div>

              <input
                ref={fileInputRef}
                className="hidden"
                type="file"
                accept="application/pdf,.pdf"
                onChange={handleFileSelection}
              />

              <div className="rounded-2xl border border-dashed border-slate-200 bg-[#f9fbfd] px-6 py-14 sm:px-10 sm:py-16">
                <div className="flex flex-col items-center text-center">
                  <div className="font-display text-[2rem] font-bold tracking-[-0.04em] text-slate-900">
                    Drag and drop PDFs here
                  </div>
                  <div className="mt-2 text-[1.05rem] text-slate-500">
                    Support for academic papers, manuals, and datasets
                  </div>

                  <button
                    type="button"
                    onClick={openFilePicker}
                    disabled={isUploading}
                    className="mt-8 rounded-md bg-[#5a627a] px-7 py-3 text-lg font-semibold text-white transition hover:opacity-90 disabled:cursor-wait disabled:opacity-70"
                  >
                    {isUploading ? "Uploading..." : "Browse Files"}
                  </button>

                  {selectedFile && (
                    <div className="mt-5 text-sm text-slate-500">
                      Selected file: <span className="font-semibold text-slate-700">{selectedFile.name}</span>
                    </div>
                  )}
                </div>
              </div>
            </section>

            <section className="space-y-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h2 className="font-display text-xl font-bold text-slate-900">Indexing Chronicle</h2>
                  <p className="mt-1 text-sm text-slate-500">{statusText}</p>
                </div>
                <div className="text-xs font-bold uppercase tracking-[0.22em] text-slate-500">
                  Session ID: PDF-RAG-001
                </div>
              </div>

              {ingestError && (
                <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
                  <div className="font-bold">{ingestError.title}</div>
                  <div className="mt-1">{ingestError.message}</div>
                  <pre className="mt-3 overflow-x-auto rounded-2xl border border-rose-200 bg-white p-3 text-xs leading-6 text-rose-800">
                    {prettyJson(ingestError.stats, "No details.")}
                  </pre>
                </div>
              )}

              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
                {pipelineCards.map((card) => {
                  const tone =
                    card.status === "done"
                      ? {
                        card: "border-emerald-100 bg-[#fbfefd]",
                        bar: "border-b-[#62ff86]",
                      }
                      : card.status === "active"
                        ? {
                          card: "border-orange-100 bg-[#fffdfb]",
                          bar: "border-b-orange-200",
                        }
                        : {
                          card: "border-slate-100 bg-slate-50/55 opacity-70",
                          bar: "border-b-slate-200",
                        };

                  return (
                    <article
                      key={card.key}
                      className={`rounded-2xl border border-b-[4px] p-4 shadow-sm ${tone.card} ${tone.bar}`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          if (card.key === "validation") {
                            toggleStep("validation");
                            return;
                          }

                          if (card.key === "extracting") {
                            toggleStep("extracting");
                            return;
                          }

                          if (card.key === "chunks") {
                            toggleStep("chunks");
                            return;
                          }

                          if (card.key === "embeddings") {
                            toggleStep("metadata");
                            return;
                          }

                          toggleStep("chroma");
                        }}
                        className="w-full text-left"
                      >
                        <h3 className="font-display text-[1rem] font-extrabold tracking-[-0.03em] text-slate-800">
                          {card.title}
                        </h3>
                        <p className="mt-3 text-base leading-8 text-slate-600">{card.description}</p>
                      </button>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="h-px flex-1 bg-slate-200" />
                <h3 className="text-[10px] font-black uppercase tracking-[0.28em] text-slate-500">
                  Workspace
                </h3>
                <div className="h-px flex-1 bg-slate-200" />
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
                <section className="overflow-hidden rounded-2xl bg-white shadow-[0_12px_32px_-4px_rgba(42,52,57,0.06)]">
                  <div className="border-b border-slate-200 bg-slate-50/80 px-5 py-4">
                    <div className="font-display text-base font-bold text-slate-900">Grand Maester's Counsel</div>
                    <div className="mt-1 text-sm text-slate-500">
                      Consult the indexed manuscript and receive grounded answers with citations.
                    </div>
                  </div>

                  <div className="border-b border-slate-200 bg-white px-5 py-4">
                    {queryAlert && (
                      <div className="mb-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
                        {queryAlert}
                      </div>
                    )}

                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                      <textarea
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        onKeyDown={handleComposerKeyDown}
                        placeholder="Send message..."
                        rows={4}
                        className="min-h-[112px] w-full border-0 bg-transparent p-0 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400"
                      />
                    </div>

                    <div className="mt-3 flex items-center justify-between gap-3">
                      <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                        {hasIndexedDocument ? "Document ready for QA" : "Upload a PDF to enable grounded answers"}
                      </div>
                      <button
                        type="button"
                        disabled={isQuerying}
                        onClick={handleAsk}
                        className="rounded-lg bg-[#565e74] px-5 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-wait disabled:opacity-70"
                      >
                        {isQuerying ? "Asking..." : "Ask"}
                      </button>
                    </div>
                  </div>

                  <div className="min-h-[560px] bg-slate-50/60 px-5 py-4">
                    {messages.length === 0 && !isQuerying ? (
                      <div className="grid min-h-[220px] place-items-center text-center">
                        <div className="max-w-xs">
                          <div className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-white text-lg font-bold text-slate-600 shadow-sm">
                            SG
                          </div>
                          <div className="mt-4 font-display text-lg font-bold text-slate-900">
                            Seek the archive
                          </div>
                          <p className="mt-2 text-sm leading-6 text-slate-500">
                            Upload a PDF, let the chronicle finish, and ask grounded questions with supporting citations.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col gap-3">
                        {isQuerying && (
                          <article className="flex gap-3">
                            <div className="grid h-8 w-8 place-items-center rounded-xl bg-slate-200 text-xs font-bold text-slate-700">
                              AI
                            </div>
                            <div className="max-w-[calc(100%-44px)] rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3">
                              <div className="mb-1 text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                                Assistant
                              </div>
                              <div className="text-sm leading-6 text-slate-700">
                                Samy is consulting the indexed manuscript and will answer with citations where possible.
                              </div>
                              <div className="mt-3 inline-flex items-center gap-2 text-xs text-slate-500">
                                <span className="h-2 w-2 animate-pulse rounded-full bg-orange-500" />
                                Generating answer...
                              </div>
                            </div>
                          </article>
                        )}

                        {messages.map((message) => (
                          <article
                            key={message.id}
                            className={`flex gap-3 ${message.role === "user" ? "justify-end" : ""}`}
                          >
                            <div
                              className={`grid h-8 w-8 place-items-center rounded-xl text-xs font-bold ${message.role === "user"
                                  ? "order-2 bg-[#dae2fd] text-[#4a5268]"
                                  : "bg-slate-200 text-slate-700"
                                }`}
                            >
                              {message.role === "user" ? "U" : "AI"}
                            </div>

                            <div
                              className={`max-w-[calc(100%-44px)] rounded-2xl border px-4 py-3 ${message.role === "user"
                                  ? "rounded-tr-md border-[#cbd5f5] bg-[#dae2fd] text-slate-800"
                                  : "rounded-tl-md border-slate-200 bg-white text-slate-900"
                                }`}
                            >
                              <div className="mb-1 text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                                {message.role === "user" ? "You" : "Assistant"}
                              </div>
                              <div className="whitespace-pre-wrap text-sm leading-6">{message.content}</div>

                              {message.citations.length > 0 && (
                                <div className="mt-3 flex flex-wrap gap-2">
                                  {message.citations.map((citation, index) => (
                                    <button
                                      key={`${message.id}-${index}`}
                                      type="button"
                                      onClick={() => handleCitationClick(citation)}
                                      className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1.5 text-xs font-bold text-orange-700"
                                    >
                                      {citationLabel(citation)}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          </article>
                        ))}
                      </div>
                    )}
                  </div>
                </section>

                <div className="space-y-4">
                  <section
                    ref={viewerRef}
                    className="overflow-hidden rounded-2xl bg-white shadow-[0_12px_32px_-4px_rgba(42,52,57,0.06)]"
                  >
                    <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50/80 px-5 py-4">
                      <div className="text-sm font-bold text-slate-900">
                        {selectedFile?.name || "Manuscript preview"}
                      </div>
                      <div className="text-xs uppercase tracking-[0.18em] text-slate-500">Manuscript Viewer</div>
                    </div>

                    <div className="border-b border-slate-200 bg-white px-5 py-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="text-sm text-slate-500">
                          {selectedFile ? "Navigate through the uploaded manuscript and jump directly from citations." : "Upload a PDF to open the manuscript viewer."}
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => navigateToPage(currentPage - 1)}
                            disabled={!pdfObjectUrl || currentPage <= 1}
                            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Prev
                          </button>
                          <div className="min-w-[110px] text-center text-sm font-semibold text-slate-700">
                            Page {currentPage}{totalPages ? ` of ${totalPages}` : ""}
                          </div>
                          <button
                            type="button"
                            onClick={() => navigateToPage(currentPage + 1)}
                            disabled={!pdfObjectUrl || (totalPages ? currentPage >= totalPages : false)}
                            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="grid min-h-[640px] place-items-center bg-[linear-gradient(180deg,#f8fafc,#eef2f7)] p-4">
                      {viewerSrc ? (
                        <div className="h-[600px] w-full overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                          <iframe
                            key={viewerSrc}
                            title="Manuscript Viewer"
                            src={viewerSrc}
                            className="h-full w-full border-0"
                          />
                        </div>
                      ) : (
                        <div className="relative w-full max-w-md rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                          <div className="space-y-3">
                            <div className="h-4 w-3/4 rounded bg-slate-100" />
                            <div className="h-4 w-1/2 rounded bg-slate-100" />
                            <div className="h-2 w-full rounded bg-slate-100" />
                            <div className="h-2 w-full rounded bg-slate-100" />
                            <div className="h-2 w-5/6 rounded bg-slate-100" />
                            <div className="grid grid-cols-2 gap-2 pt-4">
                              <div className="h-16 rounded border border-orange-100 bg-orange-50/60" />
                              <div className="h-16 rounded border border-orange-100 bg-orange-50/60" />
                            </div>
                          </div>

                          <div className="absolute bottom-4 right-4 rounded-full border border-white/40 bg-white/80 px-3 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-orange-700 shadow-sm backdrop-blur">
                            Manuscript Viewer
                          </div>
                        </div>
                      )}
                    </div>
                  </section>

                  <section className="overflow-hidden rounded-2xl bg-white shadow-[0_12px_32px_-4px_rgba(42,52,57,0.06)]">
                    <div className="border-b border-slate-200 bg-slate-50/80 px-5 py-4">
                      <div className="text-[10px] font-black uppercase tracking-[0.28em] text-slate-500">
                        Annotated Passage
                      </div>
                    </div>

                    <div className="px-5 py-4">
                      <CitationHighlightPanel
                        citation={selectedCitation}
                        onClear={() => setSelectedCitation(null)}
                      />
                    </div>
                  </section>

                  <section className="overflow-hidden rounded-2xl bg-white shadow-[0_12px_32px_-4px_rgba(42,52,57,0.06)]">
                    <div className="border-b border-slate-200 px-5 py-4">
                      <div className="text-[10px] font-black uppercase tracking-[0.28em] text-slate-500">
                        Extraction Inspector
                      </div>
                    </div>

                    <div className="px-5 py-4">
                      <div className="mb-4 flex gap-4 border-b border-slate-200 pb-3">
                        <button className="border-b-2 border-[#565e74] pb-2 text-xs font-bold text-[#565e74]">
                          Active Step Detail
                        </button>
                      </div>

                      {renderInspector(openStep, openStep ? steps[openStep].detail : null)}
                    </div>
                  </section>
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
