import { chatCompletionStream, OpenAiHttpError } from "./openaiStream.ts";

// OpenRouter's API root. There is no second location and no per-region split,
// so this is a code constant rather than required env. `OPENROUTER_BASE_URL`
// remains an optional override for tunnels, proxies, or local replays.
const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";

// PROVIDERS.md §3.9 default — providers MAY pick lower if they know their
// endpoint is fast, but openrouter relays through arbitrary upstreams that
// can take many minutes for long Claude / DeepSeek completions.
const DEFAULT_FETCH_TIMEOUT_MS = 600000;

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ProviderUsage = {
    prompt: number;
    completion: number;
    cached: number;
    total: number;
};

export type ProviderAssistant = {
    content: string;
    reasoning: string | null;
    usage: ProviderUsage;
    finishReason: string | null;
    model: string;
};

export type ProviderResponse = {
    assistant: ProviderAssistant;
    assistantRaw: unknown;
};

// All fields required. Build the config from env explicitly — typically via
// the static `fromEnv` factory.
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
        const apiKey = env.OPENROUTER_API_KEY;
        if (apiKey === undefined || apiKey.length === 0) {
            throw new Error("openrouter provider: OPENROUTER_API_KEY must be set");
        }
        const baseUrl = env.OPENROUTER_BASE_URL !== undefined && env.OPENROUTER_BASE_URL.length > 0
            ? env.OPENROUTER_BASE_URL
            : DEFAULT_BASE_URL;
        const fetchTimeoutMs = env.PLURNK_PROVIDER_FETCH_TIMEOUT !== undefined && env.PLURNK_PROVIDER_FETCH_TIMEOUT.length > 0
            ? Number(env.PLURNK_PROVIDER_FETCH_TIMEOUT)
            : DEFAULT_FETCH_TIMEOUT_MS;
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
        if (this.#reasonBudget > 0) body.include_reasoning = true;

        const timeoutSignal = AbortSignal.timeout(this.#fetchTimeoutMs);
        const effectiveSignal = signal !== undefined ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

        const raw = await chatCompletionStream({
            url: `${this.#baseUrl}/v1/chat/completions`,
            headers,
            body,
            signal: effectiveSignal,
        });

        const usage: ProviderUsage = {
            prompt: raw.usage?.prompt_tokens ?? 0,
            completion: raw.usage?.completion_tokens ?? 0,
            cached: raw.usage?.cached_tokens ?? 0,
            total: raw.usage?.total_tokens ?? 0,
        };

        return {
            assistant: {
                content: raw.content,
                reasoning: raw.reasoning_content.length > 0 ? raw.reasoning_content : null,
                usage,
                finishReason: raw.finish_reason,
                model: raw.model ?? this.#model,
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
