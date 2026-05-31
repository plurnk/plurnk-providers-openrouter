// OpenRouter provider — a thin fromEnv over the shared OpenAICompatProvider.
// OpenRouter's only bespoke surface is the /v1/models probe (context window +
// per-token pricing) and publisher-prefix tokenizer dispatch; everything else
// (the generate spine, usage mapping, reasoning translation) is the framework's.

import {
    OpenAICompatProvider,
    parseRequiredInt,
    requireEnv,
    tokenizerByPublisher,
    tokenizerFor,
    type Provider,
    type ProviderUsage,
    type TokenizerFamily,
} from "@plurnk/plurnk-providers";

const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";

// Tokenizer dispatch by OpenRouter publisher prefix (first id segment, e.g.
// "anthropic/claude-opus" → "anthropic"). Unlisted publishers fall through to
// the chars/4 heuristic. cl100k_base is a documented close approximation for
// the anthropic/x-ai families; llama-tokenizer covers the BPE-family models.
const TOKENIZER_BY_PUBLISHER: ReadonlyMap<string, TokenizerFamily> = new Map([
    ["openai", "cl100k"],
    ["anthropic", "cl100k"],
    ["~anthropic", "cl100k"],
    ["x-ai", "cl100k"],
    ["meta-llama", "llama"],
    ["mistralai", "llama"],
    ["nousresearch", "llama"],
]);

export default class OpenRouter {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        const apiKey = requireEnv(env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY", "openrouter");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_FETCH_TIMEOUT, "PLURNK_FETCH_TIMEOUT", "openrouter");
        const reasonBudget = parseRequiredInt(env.PLURNK_REASON, "PLURNK_REASON", "openrouter");
        const rawBase = env.OPENROUTER_BASE_URL !== undefined && env.OPENROUTER_BASE_URL.length > 0
            ? env.OPENROUTER_BASE_URL
            : DEFAULT_BASE_URL;
        const base = rawBase.replace(/\/v1\/?$/, "");

        const { contextSize, pricing } = await fetchModelInfo({ base, apiKey, model, fetchTimeoutMs });

        const headers: Record<string, string> = { Authorization: `Bearer ${apiKey}` };
        if (env.OPENROUTER_HTTP_REFERER) headers["HTTP-Referer"] = env.OPENROUTER_HTTP_REFERER;
        if (env.OPENROUTER_X_TITLE) headers["X-Title"] = env.OPENROUTER_X_TITLE;

        const family = tokenizerByPublisher(model, TOKENIZER_BY_PUBLISHER);

        return new OpenAICompatProvider({
            model,
            url: `${base}/v1/chat/completions`,
            fetchTimeoutMs,
            headers,
            contextSize,
            reasonBudget,
            reasoningStyle: "include_reasoning",
            countTokens: tokenizerFor(family),
            // cached tokens are a subset of prompt, billed at the prompt rate.
            costFor: (usage: ProviderUsage) =>
                Math.round(usage.prompt * pricing.prompt + usage.completion * pricing.completion),
        });
    }
}

type Pricing = { prompt: number; completion: number };

// OpenRouter's provider-pinning shorthand (e.g. `google/gemma-4-31b-it:nitro`)
// is a routing hint, not model identity; the /models catalog lists the bare id.
const catalogLookupId = (model: string): string => model.split(":")[0]!;

// USD-per-token string → pico-dollars per token (1 USD = 1e12 pico).
const parsePicoRate = (raw: string | undefined): number => {
    if (raw === undefined) return 0;
    const usd = Number.parseFloat(raw);
    return Number.isFinite(usd) && usd > 0 ? usd * 1e12 : 0;
};

type CatalogEntry = { id: string; context_length?: number; pricing?: { prompt?: string; completion?: string } };
type CatalogResponse = { data?: CatalogEntry[] };

const fetchModelInfo = async ({
    base, apiKey, model, fetchTimeoutMs,
}: { base: string; apiKey: string; model: string; fetchTimeoutMs: number }): Promise<{ contextSize: number; pricing: Pricing }> => {
    const res = await fetch(`${base}/v1/models`, {
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
    return {
        contextSize: entry.context_length,
        pricing: { prompt: parsePicoRate(entry.pricing?.prompt), completion: parsePicoRate(entry.pricing?.completion) },
    };
};
