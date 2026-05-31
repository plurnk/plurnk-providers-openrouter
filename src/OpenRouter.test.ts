import test, { mock } from "node:test";
import assert from "node:assert/strict";
import OpenRouter from "./OpenRouter.ts";

const baseEnv = Object.freeze({
    OPENROUTER_API_KEY: "sk-test",
    OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
    PLURNK_FETCH_TIMEOUT: "600000",
    PLURNK_REASON: "0",
});

// Mock the /models catalog probe. `entry` becomes the single catalog row.
const mockCatalog = (entry: unknown) => {
    const calls: string[] = [];
    mock.method(globalThis, "fetch", async (url: string) => {
        calls.push(String(url));
        return new Response(JSON.stringify({ data: [entry] }), { status: 200 });
    });
    return calls;
};
test.afterEach(() => mock.restoreAll());

const opus = { id: "anthropic/claude-opus-latest", context_length: 200000, pricing: { prompt: "0.000015", completion: "0.000075" } };

// — fromEnv env guards —

test("fromEnv: throws when OPENROUTER_API_KEY is unset", async () => {
    await assert.rejects(() => OpenRouter.fromEnv({}, "anthropic/claude-opus-latest"), /OPENROUTER_API_KEY must be set/);
});

test("fromEnv: throws when PLURNK_FETCH_TIMEOUT is unset", async () => {
    await assert.rejects(
        () => OpenRouter.fromEnv({ OPENROUTER_API_KEY: "sk-test", PLURNK_REASON: "0" }, "m"),
        /PLURNK_FETCH_TIMEOUT must be set/,
    );
});

test("fromEnv: throws when PLURNK_REASON is non-numeric", async () => {
    mockCatalog(opus);
    await assert.rejects(() => OpenRouter.fromEnv({ ...baseEnv, PLURNK_REASON: "lots" }, "m"), /PLURNK_REASON must be a number/);
});

// — catalog probe —

test("fromEnv: resolves contextSize from /models and defaults the base URL", async () => {
    const calls = mockCatalog(opus);
    const p = await OpenRouter.fromEnv({ OPENROUTER_API_KEY: "sk-test", PLURNK_FETCH_TIMEOUT: "600000", PLURNK_REASON: "0" }, "anthropic/claude-opus-latest");
    assert.equal(p.contextSize, 200000);
    assert.equal(p.model, "anthropic/claude-opus-latest");
    assert.equal(calls[0], "https://openrouter.ai/api/v1/models");
});

test("fromEnv: strips :provider routing suffix for catalog lookup but keeps it as model id", async () => {
    mockCatalog({ id: "google/gemma-4-31b-it", context_length: 131072 });
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "google/gemma-4-31b-it:nitro");
    assert.equal(p.contextSize, 131072);
    assert.equal(p.model, "google/gemma-4-31b-it:nitro");
});

test("fromEnv: throws when model is missing from catalog", async () => {
    mockCatalog({ id: "other/model", context_length: 1 });
    await assert.rejects(() => OpenRouter.fromEnv({ ...baseEnv }, "anthropic/claude-opus-latest"), /has no entry for "anthropic\/claude-opus-latest"/);
});

test("fromEnv: throws when /models returns non-2xx", async () => {
    mock.method(globalThis, "fetch", async () => new Response("upstream unavailable", { status: 503 }));
    await assert.rejects(() => OpenRouter.fromEnv({ ...baseEnv }, "anthropic/claude-opus-latest"), /\/models returned 503/);
});

// — Provider surface on the constructed instance —

test("costFor: pico-per-token math from catalog rates", async () => {
    mockCatalog(opus); // $15/M prompt → 1.5e7 pico/tok; $75/M completion → 7.5e7 pico/tok
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "anthropic/claude-opus-latest");
    // 1000×1.5e7 + 100×7.5e7 = 2.25e10
    assert.equal(p.costFor({ prompt: 1000, completion: 100, reasoning: 0, cached: 0, total: 1100 }), 22500000000);
});

test("costFor: bills reasoning at the completion rate", async () => {
    mockCatalog(opus); // completion → 7.5e7 pico/tok
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "anthropic/claude-opus-latest");
    // (completion 100 + reasoning 100) × 7.5e7 = 1.5e10
    assert.equal(p.costFor({ prompt: 0, completion: 100, reasoning: 100, cached: 0, total: 200 }), 15000000000);
});

test("costFor: returns 0 for a free model (no rates)", async () => {
    mockCatalog({ id: "free/model", context_length: 8192 });
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "free/model");
    assert.equal(p.costFor({ prompt: 1000, completion: 500, reasoning: 0, cached: 0, total: 1500 }), 0);
});

test("tokenizer dispatch: anthropic/* → cl100k (hello world = 2)", async () => {
    mockCatalog(opus);
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "anthropic/claude-opus-latest");
    assert.equal(p.countTokens("hello world"), 2);
});

test("tokenizer dispatch: meta-llama/* → llama (hello world = 3)", async () => {
    mockCatalog({ id: "meta-llama/llama-3.3-70b-instruct", context_length: 131072 });
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "meta-llama/llama-3.3-70b-instruct");
    assert.equal(p.countTokens("hello world"), 3);
});

test("tokenizer dispatch: unknown publisher → heuristic", async () => {
    mockCatalog({ id: "obscure/model", context_length: 4096 });
    const p = await OpenRouter.fromEnv({ ...baseEnv }, "obscure/model");
    assert.equal(p.countTokens(""), 0);
    assert.equal(p.countTokens("abcde"), 2); // ceil(5/4)
});
