import test from "node:test";
import assert from "node:assert/strict";
import OpenRouter from "./OpenRouter.ts";

test("fromEnv: throws when OPENROUTER_API_KEY is unset", async () => {
    await assert.rejects(
        () => OpenRouter.fromEnv({}, "anthropic/claude-opus-latest"),
        /OPENROUTER_API_KEY must be set/,
    );
});

test("fromEnv: uses default base URL when OPENROUTER_BASE_URL unset", async (t) => {
    let captured: string | URL | null = null;
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async (url: string | URL) => {
        captured = url;
        return {
            ok: true,
            json: async () => ({ data: [{ id: "anthropic/claude-opus-latest", context_length: 200000 }] }),
        };
    }) as typeof fetch;

    const p = await OpenRouter.fromEnv({ OPENROUTER_API_KEY: "sk-test" }, "anthropic/claude-opus-latest");
    assert.equal(p.contextSize, 200000);
    assert.equal(String(captured), "https://openrouter.ai/api/v1/models");
});

test("fromEnv: resolves contextSize from /models catalog", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async (_url: string | URL) => ({
        ok: true,
        json: async () => ({ data: [{ id: "anthropic/claude-opus-latest", context_length: 200000 }] }),
    })) as typeof fetch;

    const p = await OpenRouter.fromEnv({
        OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
        OPENROUTER_API_KEY: "sk-test",
    }, "anthropic/claude-opus-latest");
    assert.equal(p.contextSize, 200000);
    assert.equal(p.model, "anthropic/claude-opus-latest");
});

test("fromEnv: strips :provider routing suffix for catalog lookup", async (t) => {
    let captured: string | URL | null = null;
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async (url: string | URL) => {
        captured = url;
        return {
            ok: true,
            json: async () => ({ data: [{ id: "google/gemma-4-31b-it", context_length: 131072 }] }),
        };
    }) as typeof fetch;

    const p = await OpenRouter.fromEnv({
        OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
        OPENROUTER_API_KEY: "sk-test",
    }, "google/gemma-4-31b-it:nitro");
    assert.equal(p.contextSize, 131072);
    // Model identity is preserved including the routing hint; only the
    // catalog lookup strips it. The provider sends the full id back to
    // /chat/completions so the routing actually happens.
    assert.equal(p.model, "google/gemma-4-31b-it:nitro");
    assert.match(String(captured), /\/v1\/models$/);
});

test("fromEnv: throws when model is missing from catalog", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ data: [{ id: "other/model", context_length: 1 }] }),
    })) as unknown as typeof fetch;

    await assert.rejects(
        () => OpenRouter.fromEnv({
            OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
            OPENROUTER_API_KEY: "sk-test",
        }, "anthropic/claude-opus-latest"),
        /has no entry for "anthropic\/claude-opus-latest"/,
    );
});

test("fromEnv: throws when /models returns non-2xx", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: false,
        status: 503,
        text: async () => "upstream unavailable",
    })) as unknown as typeof fetch;

    await assert.rejects(
        () => OpenRouter.fromEnv({
            OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
            OPENROUTER_API_KEY: "sk-test",
        }, "anthropic/claude-opus-latest"),
        /\/models returned 503/,
    );
});

const zeroPricing = { prompt_pico_per_token: 0, completion_pico_per_token: 0, cached_pico_per_token: 0 };

test("contextSize, model, baseUrl exposed on instance", () => {
    const p = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "sk-test",
        model: "anthropic/claude-opus-latest",
        contextSize: 200000,
        fetchTimeoutMs: 60000,
        reasonBudget: 0,
        httpReferer: "",
        xTitle: "",
        pricing: zeroPricing,
    });
    assert.equal(p.contextSize, 200000);
    assert.equal(p.model, "anthropic/claude-opus-latest");
    assert.equal(p.baseUrl, "https://openrouter.ai/api");
});

test("baseUrl strips /v1 suffix", () => {
    const a = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "sk", model: "m", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        httpReferer: "", xTitle: "", pricing: zeroPricing,
    });
    assert.equal(a.baseUrl, "https://openrouter.ai/api");
    const b = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1/",
        apiKey: "sk", model: "m", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        httpReferer: "", xTitle: "", pricing: zeroPricing,
    });
    assert.equal(b.baseUrl, "https://openrouter.ai/api");
});

test("costFor: pico-per-token math from catalog rates", () => {
    // Claude Opus-style rates: $15/M input ($1.5e-5/token) = 1.5e7 pico/token;
    // $75/M output ($7.5e-5/token) = 7.5e7 pico/token.
    const p = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "sk", model: "anthropic/claude-opus", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        httpReferer: "", xTitle: "",
        pricing: { prompt_pico_per_token: 1.5e7, completion_pico_per_token: 7.5e7, cached_pico_per_token: 1.5e7 },
    });
    // 1000 prompt × 1.5e7 + 100 completion × 7.5e7 = 1.5e10 + 7.5e9 = 2.25e10 pico = $0.0225
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 0, total: 1100 }), 22500000000);
});

test("costFor: returns 0 when rates are zero (free models)", () => {
    const p = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "sk", model: "free-model", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        httpReferer: "", xTitle: "", pricing: zeroPricing,
    });
    assert.equal(p.costFor({ prompt: 1000, completion: 500, cached: 0, total: 1500 }), 0);
});

test("countTokens: heuristic returns 0 for empty, ceil(len/4) otherwise", () => {
    const p = new OpenRouter({
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "sk", model: "m", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        httpReferer: "", xTitle: "", pricing: zeroPricing,
    });
    assert.equal(p.countTokens(""), 0);
    assert.equal(p.countTokens("abcd"), 1);
    assert.equal(p.countTokens("abcde"), 2);
});
