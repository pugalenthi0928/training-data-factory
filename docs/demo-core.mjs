const encoder = new TextEncoder();

export function normalizeText(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .trim();
}

export async function sha256Hex(value) {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    encoder.encode(String(value)),
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function stableContentId(prefix, ...parts) {
  const payload = parts.map((part) => normalizeText(part)).join("\u0000");
  return `${prefix}_${await sha256Hex(payload)}`;
}

function sentences(text) {
  const matches = normalizeText(text).match(/[^.!?]+[.!?]+|[^.!?]+$/g);
  return (matches ?? []).map((sentence) => sentence.trim()).filter(Boolean);
}

function chunkText(text, targetLength = 420) {
  const sourceSentences = sentences(text);
  if (sourceSentences.length === 0) return [];

  const chunks = [];
  let current = [];
  let currentLength = 0;

  for (const sentence of sourceSentences) {
    if (current.length > 0 && currentLength + sentence.length > targetLength) {
      chunks.push(current.join(" "));
      current = [];
      currentLength = 0;
    }
    current.push(sentence);
    currentLength += sentence.length + 1;
  }

  if (current.length > 0) chunks.push(current.join(" "));
  return chunks;
}

async function buildSources(sources) {
  if (!Array.isArray(sources) || sources.length < 2) {
    throw new Error("Add at least two source documents for a safe split.");
  }

  const records = [];
  for (const [sourceIndex, source] of sources.entries()) {
    const name = normalizeText(source.name) || `source-${sourceIndex + 1}.txt`;
    const content = normalizeText(source.content);
    if (content.length < 80) {
      throw new Error(`${name} needs at least 80 characters of source text.`);
    }

    const documentId = await stableContentId("doc", content);
    const chunks = [];
    for (const [chunkIndex, text] of chunkText(content).entries()) {
      chunks.push({
        chunk_id: await stableContentId("chunk", documentId, chunkIndex, text),
        chunk_index: chunkIndex,
        text,
      });
    }

    records.push({
      name,
      document_id: documentId,
      character_count: content.length,
      chunks,
    });
  }

  return records;
}

function firstSentence(text) {
  return sentences(text)[0] ?? normalizeText(text);
}

async function buildExamples(sourceRecords) {
  const examples = [];

  for (const source of sourceRecords) {
    for (const chunk of source.chunks) {
      const sourceSentence = firstSentence(chunk.text);
      const summarySentences = sentences(chunk.text).slice(0, 2).join(" ");
      const candidates = [
        {
          task_name: "summary_smoke",
          input_text: `Summarise the main control described in ${source.name}.`,
          output_text: summarySentences || sourceSentence,
        },
        {
          task_name: "qa_smoke",
          input_text: `What engineering principle is central to ${source.name}?`,
          output_text: sourceSentence,
        },
      ];

      for (const candidate of candidates) {
        const outputWordCount = normalizeText(candidate.output_text).split(" ").length;
        const qualityFlags = [];
        if (outputWordCount < 8) qualityFlags.push("short_output");

        examples.push({
          id: await stableContentId(
            "example",
            source.document_id,
            chunk.chunk_id,
            candidate.task_name,
            candidate.input_text,
            candidate.output_text,
          ),
          document_id: source.document_id,
          chunk_id: chunk.chunk_id,
          source_name: source.name,
          task_name: candidate.task_name,
          input_text: candidate.input_text,
          output_text: candidate.output_text,
          quality: {
            output_word_count: outputWordCount,
            source_grounded: normalizeText(chunk.text).includes(
              normalizeText(candidate.output_text),
            ),
            flags: qualityFlags,
          },
        });
      }
    }
  }

  return examples;
}

function tokens(text) {
  return normalizeText(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function ngrams(text, size) {
  const words = tokens(text);
  const grams = [];
  for (let index = 0; index <= words.length - size; index += 1) {
    grams.push(words.slice(index, index + size).join(" "));
  }
  return grams;
}

function contaminationReport(examples, benchmarkText, ngramSize = 8) {
  const benchmark = normalizeText(benchmarkText);
  if (benchmark.length < 40) {
    throw new Error("The benchmark needs at least 40 characters.");
  }

  const benchmarkIndex = new Set(ngrams(benchmark, ngramSize));
  if (benchmarkIndex.size === 0) {
    throw new Error(`The benchmark needs at least ${ngramSize} words.`);
  }

  const flagged = [];
  for (const example of examples) {
    const candidateText = [example.input_text, example.output_text].join(" ");
    const matches = ngrams(candidateText, ngramSize).filter((gram) =>
      benchmarkIndex.has(gram),
    );
    if (matches.length > 0) {
      flagged.push({
        example_id: example.id,
        source_name: example.source_name,
        matching_ngrams: [...new Set(matches)].slice(0, 3),
      });
    }
  }

  return {
    status: flagged.length === 0 ? "passed" : "blocked",
    ngram_size: ngramSize,
    benchmark_ngram_count: benchmarkIndex.size,
    examples_screened: examples.length,
    flagged_count: flagged.length,
    contamination_rate: Number((flagged.length / examples.length).toFixed(4)),
    flagged_examples: flagged,
  };
}

function chooseTestSources(examples, testFraction, seed) {
  if (!(testFraction > 0 && testFraction < 1)) {
    throw new Error("The requested test fraction must be between 0 and 1.");
  }

  const sourceCounts = new Map();
  for (const example of examples) {
    sourceCounts.set(
      example.document_id,
      (sourceCounts.get(example.document_id) ?? 0) + 1,
    );
  }

  const sourceIds = [...sourceCounts.keys()].sort((left, right) => {
    const leftDigest = left.split("_").at(-1);
    const rightDigest = right.split("_").at(-1);
    const leftRank = (Number.parseInt(leftDigest.slice(0, 8), 16) ^ seed) >>> 0;
    const rightRank = (Number.parseInt(rightDigest.slice(0, 8), 16) ^ seed) >>> 0;
    return leftRank - rightRank || left.localeCompare(right);
  });
  if (sourceIds.length < 2) {
    throw new Error("At least two unique source IDs are required.");
  }

  const targetRows = examples.length * testFraction;
  let best = null;
  const maxMask = 2 ** sourceIds.length;

  for (let mask = 1; mask < maxMask - 1; mask += 1) {
    const selected = [];
    let rows = 0;
    for (let index = 0; index < sourceIds.length; index += 1) {
      if ((mask & (2 ** index)) !== 0) {
        selected.push(sourceIds[index]);
        rows += sourceCounts.get(sourceIds[index]);
      }
    }

    const candidate = {
      selected,
      rows,
      distance: Math.abs(rows - targetRows),
    };
    if (
      best === null ||
      candidate.distance < best.distance ||
      (candidate.distance === best.distance &&
        candidate.selected.join("") < best.selected.join(""))
    ) {
      best = candidate;
    }
  }

  return new Set(best.selected);
}

function overlap(leftRows, rightRows, key) {
  const left = new Set(leftRows.map((row) => row[key]).filter(Boolean));
  const right = new Set(rightRows.map((row) => row[key]).filter(Boolean));
  return [...left].filter((value) => right.has(value));
}

async function splitExamples(examples, testFraction, seed) {
  const testSources = chooseTestSources(examples, testFraction, seed);
  const train = examples.filter((row) => !testSources.has(row.document_id));
  const test = examples.filter((row) => testSources.has(row.document_id));
  const documentOverlap = overlap(train, test, "document_id");
  const chunkOverlap = overlap(train, test, "chunk_id");

  if (documentOverlap.length > 0 || chunkOverlap.length > 0) {
    throw new Error("Unsafe split detected. Source identifiers cross partitions.");
  }

  const trainJsonl = train.map((row) => JSON.stringify(row)).join("\n");
  const testJsonl = test.map((row) => JSON.stringify(row)).join("\n");

  return {
    train,
    test,
    manifest: {
      status: "passed",
      strategy: "whole_source_subset",
      seed,
      requested_test_fraction: testFraction,
      achieved_test_fraction: Number((test.length / examples.length).toFixed(4)),
      counts: {
        total_examples: examples.length,
        train_examples: train.length,
        test_examples: test.length,
        train_sources: new Set(train.map((row) => row.document_id)).size,
        test_sources: new Set(test.map((row) => row.document_id)).size,
      },
      overlap: {
        document_ids: documentOverlap,
        chunk_ids: chunkOverlap,
      },
      artifact_sha256: {
        train_jsonl: await sha256Hex(trainJsonl),
        test_jsonl: await sha256Hex(testJsonl),
      },
    },
  };
}

export async function runBrowserDemo(config) {
  const sourceRecords = await buildSources(config.sources);
  const examples = await buildExamples(sourceRecords);
  const contamination = contaminationReport(examples, config.benchmarkText);
  const artifacts = {
    "provenance.json": {
      mode: "deterministic_browser_smoke",
      source_count: sourceRecords.length,
      sources: sourceRecords,
    },
    "examples.json": examples,
    "contamination_report.json": contamination,
  };

  if (contamination.status === "blocked") {
    return {
      status: "blocked",
      sourceRecords,
      examples,
      contamination,
      split: null,
      artifacts,
    };
  }

  const split = await splitExamples(
    examples,
    Number(config.testFraction ?? 0.33),
    Number(config.seed ?? 42),
  );
  artifacts["split_manifest.json"] = split.manifest;

  return {
    status: "passed",
    sourceRecords,
    examples,
    contamination,
    split,
    artifacts,
  };
}
