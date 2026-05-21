import { encode as encodeCl100k } from "gpt-tokenizer/encoding/cl100k_base";
import llamaTokenizer from "llama-tokenizer-js";
import { chatCompletionStream, OpenAiHttpError } from "./openaiStream.ts";

// Tokenizer dispatch driven by OpenRouter model-id prefix. OpenRouter relays
// through dozens of upstream families; precision per route requires knowing
// which tokenizer the upstream actually uses. This map covers the major
// publishers; everything else falls through to the chars/4 heuristic.
//
// Prefixes match the first segment of OpenRouter's id (publisher), e.g.
// "anthropic/claude-opus-latest" → "anthropic".
type TokenizerKind = "cl100k" | "llama" | "heuristic";

const TOKENIZER_BY_PUBLISHER: ReadonlyMap<string, TokenizerKind> = new Map([
    // GPT-3.5/4 family — cl100k_base is canonical
    ["openai", "cl100k"],
    // Anthropic's actual tokenizer (Claude BPE) isn't on npm as a sync lib;
    // cl100k_base is a documented close approximation (within ~5% in practice
    // for English). Better than chars/4.
    ["anthropic", "cl100k"],
    ["~anthropic", "cl100k"],
    // xAI Grok — documented cl100k_base
    ["x-ai", "cl100k"],
    // Llama-family — llama-tokenizer-js handles 1/2/3 accurately; mistral
    // shares the BPE family closely enough
    ["meta-llama", "llama"],
    ["mistralai", "llama"],
    ["nousresearch", "llama"],
    // Open-weight Chinese / European models: heuristic for now (qwen uses
    // gpt2 BPE; deepseek uses its own; gemma uses sentencepiece). Per-family
    // wiring is pass-3.
]);

const tokenizerForModel = (model: string): TokenizerKind => {
    const publisher = model.split("/")[0]!;
    return TOKENIZER_BY_PUBLISHER.get(publisher) ?? "heuristic";
};

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

// Per-token pricing in pico-dollars per token. OpenRouter's /v1/models
// exposes prompt/completion rates as USD-per-token strings; sibling converts
// to pico-dollars (USD × 1e12) at fromEnv time. `cached` is OpenRouter's
// upstream-reported subset of prompt — not a separate billing dimension at
// the relay, so engine uses prompt-rate for cached portion (matches how
// upstreams typically include cached_tokens within prompt_tokens).
export type OpenRouterPricing = {
    prompt_pico_per_token: number;
    completion_pico_per_token: number;
    cached_pico_per_token: number;
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
    // Per-token pricing resolved from /v1/models at construction time.
    pricing: OpenRouterPricing;
    // Tokenizer family resolved from the model's publisher prefix at
    // fromEnv. Frozen on the instance.
    tokenizer: TokenizerKind;
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
    #pricing: OpenRouterPricing;
    #tokenizer: TokenizerKind;

    constructor(config: OpenRouterConfig) {
        this.#baseUrl = config.baseUrl.replace(/\/v1\/?$/, "");
        this.#apiKey = config.apiKey;
        this.#model = config.model;
        this.#contextSize = config.contextSize;
        this.#fetchTimeoutMs = config.fetchTimeoutMs;
        this.#reasonBudget = config.reasonBudget;
        this.#httpReferer = config.httpReferer;
        this.#xTitle = config.xTitle;
        this.#pricing = config.pricing;
        this.#tokenizer = config.tokenizer;
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
        const info = await fetchModelInfo({
            baseUrl: normalizedBase,
            apiKey,
            model,
            fetchTimeoutMs,
        });
        return new OpenRouter({
            baseUrl,
            apiKey,
            model,
            contextSize: info.contextSize,
            fetchTimeoutMs,
            reasonBudget: Number(env.PLURNK_REASON ?? "0"),
            httpReferer: env.OPENROUTER_HTTP_REFERER ?? "",
            xTitle: env.OPENROUTER_X_TITLE ?? "",
            pricing: info.pricing,
            tokenizer: tokenizerForModel(model),
        });
    }

    get contextSize(): number { return this.#contextSize; }
    get model(): string { return this.#model; }
    get baseUrl(): string { return this.#baseUrl; }
    get pricing(): OpenRouterPricing { return this.#pricing; }

    // Per-publisher tokenizer dispatch. Decided once at construction from
    // the model id's publisher prefix (resolved in OpenRouterConfig) and
    // frozen on the instance. Unknown publishers fall back to the chars/4
    // heuristic.
    countTokens(text: string): number {
        if (text.length === 0) return 0;
        switch (this.#tokenizer) {
            case "cl100k": return encodeCl100k(text).length;
            case "llama":  return llamaTokenizer.encode(text).length;
            case "heuristic": return Math.ceil(text.length / 4);
        }
    }

    get tokenizer(): TokenizerKind { return this.#tokenizer; }

    // Cost calculation from OpenRouter-reported per-token pricing.
    // cached portion is billed at prompt rate (OpenRouter's upstreams report
    // cached_tokens as a subset of prompt_tokens, already included; pricing
    // is a single prompt rate that covers both).
    costFor(usage: ProviderUsage): number {
        const promptCost = usage.prompt * this.#pricing.prompt_pico_per_token;
        const completionCost = usage.completion * this.#pricing.completion_pico_per_token;
        return Math.round(promptCost + completionCost);
    }

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

type CatalogPricing = { prompt?: string; completion?: string };
type CatalogEntry = { id: string; context_length?: number; pricing?: CatalogPricing };
type CatalogResponse = { data?: CatalogEntry[] };

// USD-string -> pico-dollars-per-token. OpenRouter's catalog reports rates
// as USD-per-token strings (e.g. "0.000003" = $3/M tokens). 1 USD = 1e12
// pico-dollars; multiplying the parsed float by 1e12 gives pico-per-token.
const parsePicoRate = (raw: string | undefined): number => {
    if (raw === undefined) return 0;
    const usd = Number.parseFloat(raw);
    if (!Number.isFinite(usd) || usd <= 0) return 0;
    return usd * 1e12;
};

const fetchModelInfo = async ({
    baseUrl, apiKey, model, fetchTimeoutMs,
}: { baseUrl: string; apiKey: string; model: string; fetchTimeoutMs: number }): Promise<{ contextSize: number; pricing: OpenRouterPricing }> => {
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
    const promptRate = parsePicoRate(entry.pricing?.prompt);
    const completionRate = parsePicoRate(entry.pricing?.completion);
    return {
        contextSize: entry.context_length,
        pricing: {
            prompt_pico_per_token: promptRate,
            completion_pico_per_token: completionRate,
            // OpenRouter doesn't expose a separate cached rate; cached tokens
            // are billed at the prompt rate (they're a subset of prompt).
            cached_pico_per_token: promptRate,
        },
    };
};

export { OpenAiHttpError };
