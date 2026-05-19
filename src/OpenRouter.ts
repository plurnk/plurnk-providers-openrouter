import { PlurnkParser } from "@plurnk/plurnk-grammar";
import type { PlurnkStatement } from "@plurnk/plurnk-grammar";
import { chatCompletionStream, OpenAiHttpError } from "./openaiStream.ts";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ProviderAssistant = {
    tokens: number;
    content: string;
    ops: PlurnkStatement[];
    reasoning: string | null;
};

export type ProviderResponse = {
    assistant: ProviderAssistant;
    assistantRaw: unknown;
};

// All fields required. Defaults belong in `.env.example`, not in library
// code. Build the config from env explicitly — typically via `fromEnv`.
export type OpenRouterConfig = {
    baseUrl: string;
    apiKey: string;
    model: string;
    contextSize: number;
    fetchTimeoutMs: number;
    // PROVIDERS.md §3.8 universal reasoning budget. OpenRouter exposes a
    // relay-level passthrough toggle (`include_reasoning`) but does not
    // accept a token count directly. v0 translation: any positive budget
    // turns the passthrough on; zero turns it off.
    reasonBudget: number;
    // Ranking headers sent to OpenRouter. Empty string means omit.
    httpReferer: string;
    xTitle: string;
};

export default class OpenRouter {
    #baseUrl: string;
    #apiKey: string;
    #model: string;
    #contextSize: number;
    #fetchTimeoutMs: number;
    #reasonBudget: number;
    #httpReferer: string;
    #xTitle: string;

    constructor(config: OpenRouterConfig) {
        this.#baseUrl = config.baseUrl.replace(/\/v1\/?$/, "");
        this.#apiKey = config.apiKey;
        this.#model = config.model;
        this.#contextSize = config.contextSize;
        this.#fetchTimeoutMs = config.fetchTimeoutMs;
        this.#reasonBudget = config.reasonBudget;
        this.#httpReferer = config.httpReferer;
        this.#xTitle = config.xTitle;
    }

    // PROVIDERS.md §3.7 factory contract. Async: resolves contextSize from
    // OpenRouter's `/models` catalog at construction time. plurnk-service
    // pays the lookup cost once at boot and the provider holds a frozen
    // contextSize for the rest of its life.
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<OpenRouter> {
        const baseUrl = env.OPENROUTER_BASE_URL;
        if (baseUrl === undefined || baseUrl.length === 0) {
            throw new Error("openrouter provider: OPENROUTER_BASE_URL must be set");
        }
        const apiKey = env.OPENROUTER_API_KEY;
        if (apiKey === undefined || apiKey.length === 0) {
            throw new Error("openrouter provider: OPENROUTER_API_KEY must be set");
        }
        const fetchTimeoutMs = Number(env.OPENROUTER_FETCH_TIMEOUT_MS ?? "600000");
        const normalizedBase = baseUrl.replace(/\/v1\/?$/, "");
        const contextSize = await fetchContextSize({
            baseUrl: normalizedBase,
            apiKey,
            model,
            fetchTimeoutMs,
        });
        return new OpenRouter({
            baseUrl,
            apiKey,
            model,
            contextSize,
            fetchTimeoutMs,
            reasonBudget: Number(env.PLURNK_REASON ?? "0"),
            httpReferer: env.OPENROUTER_HTTP_REFERER ?? "",
            xTitle: env.OPENROUTER_X_TITLE ?? "",
        });
    }

    get contextSize(): number { return this.#contextSize; }
    get model(): string { return this.#model; }
    get baseUrl(): string { return this.#baseUrl; }

    async generate({ messages, signal }: { messages: ChatMessage[]; signal?: AbortSignal }): Promise<ProviderResponse> {
        const headers: Record<string, string> = {
            Authorization: `Bearer ${this.#apiKey}`,
        };
        if (this.#httpReferer.length > 0) headers["HTTP-Referer"] = this.#httpReferer;
        if (this.#xTitle.length > 0) headers["X-Title"] = this.#xTitle;

        const body: Record<string, unknown> = { model: this.#model, messages };
        // PROVIDERS.md §3.8: provider owns wire-format translation of the
        // universal PLURNK_REASON token budget. OpenRouter relays the
        // upstream's reasoning when `include_reasoning: true`; the
        // upstream still owns whether it produces any.
        if (this.#reasonBudget > 0) body.include_reasoning = true;

        const timeoutSignal = AbortSignal.timeout(this.#fetchTimeoutMs);
        const effectiveSignal = signal !== undefined ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

        const raw = await chatCompletionStream({
            url: `${this.#baseUrl}/v1/chat/completions`,
            headers,
            body,
            signal: effectiveSignal,
        });

        // PROVIDERS.md §3.3: split parser items into statements (→ ops) and
        // text fragments (→ reasoning). Free-form prose between ops is the
        // model's casual narration; treat it as reasoning per engine policy.
        const parsed = PlurnkParser.parse(raw.content);
        const ops: PlurnkStatement[] = [];
        const textFragments: string[] = [];
        for (const item of parsed.items) {
            if (item.kind === "statement") ops.push(item.statement);
            else if (item.kind === "text") {
                const trimmed = item.text.trim();
                if (trimmed.length > 0) textFragments.push(trimmed);
            }
        }
        const wireReasoning = raw.reasoning_content.length > 0 ? raw.reasoning_content : "";
        const scrapedReasoning = textFragments.join("\n");
        const reasoningParts = [wireReasoning, scrapedReasoning].filter((s) => s.length > 0);
        const reasoning = reasoningParts.length > 0 ? reasoningParts.join("\n\n") : null;

        return {
            assistant: {
                tokens: raw.usage?.completion_tokens ?? 0,
                content: raw.content,
                ops,
                reasoning,
            },
            assistantRaw: raw,
        };
    }
}

// OpenRouter's provider-pinning shorthand (e.g. `google/gemma-4-31b-it:nitro`)
// routes the completion to a specific upstream. The `/models` catalog lists
// the bare model id, so strip the suffix for catalog lookup — it's a routing
// hint, not part of model identity.
const catalogLookupId = (model: string): string => model.split(":")[0]!;

type CatalogEntry = { id: string; context_length?: number };
type CatalogResponse = { data?: CatalogEntry[] };

const fetchContextSize = async ({
    baseUrl, apiKey, model, fetchTimeoutMs,
}: { baseUrl: string; apiKey: string; model: string; fetchTimeoutMs: number }): Promise<number> => {
    const res = await fetch(`${baseUrl}/v1/models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal: AbortSignal.timeout(fetchTimeoutMs),
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`OpenRouter /models returned ${res.status}: ${body}`);
    }
    const data = (await res.json()) as CatalogResponse;
    const lookupId = catalogLookupId(model);
    const entry = data.data?.find((m) => m.id === lookupId);
    if (entry === undefined) throw new Error(`OpenRouter /models has no entry for "${lookupId}"`);
    if (entry.context_length === undefined || entry.context_length <= 0) {
        throw new Error(`OpenRouter /models has no context_length for "${lookupId}"`);
    }
    return entry.context_length;
};

export { OpenAiHttpError };
